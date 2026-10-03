"""Numerically stable logistic regression and a leakage-free three-seed experiment.

The module implements binary logistic regression in NumPy: an overflow-free
sigmoid, a finite ``logaddexp`` cross-entropy with its analytic gradient,
full-batch gradient descent that returns a ``FitResult`` with the fitted
parameters and the per-step history, and a z-score scaler that stores training
statistics, keeps scale 1.0 for zero-variance columns, and never refits. The
loss and gradient functions stay public so callers can check the gradient
against central finite differences of the loss.

The experiment reruns the CSC781 Module 6 Assignment 6 protocol with fixed
seeds and a stratified split into train, dev and test parts. It fits one model
per learning rate, chooses the learning rate and decision threshold together on
the dev split, and scores the test split only after that. An unregularized
``sklearn.linear_model.LogisticRegression`` baseline runs on identical data.
Run it from the repository root::

    uv run --locked python -m mlshowcase.logistic --output-dir results --seeds 42 43 44

The run writes ``logistic.json`` and two PNG figures into the output directory,
one showing training loss per seed and one comparing the selected model with
the baseline. See ``docs/logistic.md`` and ``results/logistic.json`` for the
protocol and the recorded evidence.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from mlshowcase.common import environment, stratified_split, write_results

# Figures are always written to PNG files, so select the headless Agg backend,
# which behaves the same in a terminal and in headless runs.
plt.switch_backend("Agg")

DEFAULT_SEEDS: tuple[int, ...] = (42, 43, 44)
LEARNING_RATES: tuple[float, ...] = (0.01, 0.05, 0.1, 0.5)
MAX_STEPS = 3000
GRADIENT_TOLERANCE = 1e-6
THRESHOLD_GRID: tuple[float, ...] = tuple(round(0.10 + 0.05 * index, 2) for index in range(17))
GRADIENT_CURVE_STRIDE = 10
POSITIVE_LABEL = 1
RESULT_NAME = "logistic"
LOSS_FIGURE = "logistic_loss.png"
COMPARISON_FIGURE = "logistic_comparison.png"


# %% Stable numerics
def sigmoid(z: np.ndarray) -> np.ndarray:
    """Logistic function ``1 / (1 + exp(-z))`` without overflow.

    Non-negative scores use ``1 / (1 + exp(-z))`` and negative scores use
    ``exp(z) / (1 + exp(z))``. Each branch only exponentiates a value at most
    zero, so the output stays finite for inputs of any magnitude and no
    overflow warning is suppressed.
    """
    values = np.asarray(z, dtype=float)
    flat = values.ravel()
    result = np.empty_like(flat)
    positive = flat >= 0.0
    result[positive] = 1.0 / (1.0 + np.exp(-flat[positive]))
    negative = ~positive
    exponentials = np.exp(flat[negative])
    result[negative] = exponentials / (1.0 + exponentials)
    return result.reshape(values.shape)


def add_intercept(X: np.ndarray) -> np.ndarray:
    """Prepend a column of ones to a 2-D feature matrix."""
    features = np.asarray(X, dtype=float)
    if features.ndim != 2:
        raise ValueError("X must be a 2-D array of shape (n_samples, n_features)")
    return np.column_stack((np.ones(features.shape[0]), features))


def _scores(design: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Linear scores ``design @ theta`` with matching-dimension validation."""
    features = np.asarray(design, dtype=float)
    weights = np.asarray(theta, dtype=float).ravel()
    if features.ndim != 2:
        raise ValueError("the design matrix must be a 2-D array")
    if features.shape[1] != weights.shape[0]:
        raise ValueError(
            f"theta has {weights.shape[0]} entries but the design matrix has "
            f"{features.shape[1]} columns"
        )
    return features @ weights


def cross_entropy(y: np.ndarray, logits: np.ndarray) -> float:
    """Mean binary cross-entropy for 0/1 labels and real-valued scores.

    Each sample contributes ``logaddexp(0, -z)`` when ``y == 1`` and
    ``logaddexp(0, z)`` when ``y == 0``. That equals the usual
    ``-log(sigmoid(z))`` for ``y == 1`` and ``-log(1 - sigmoid(z))`` for
    ``y == 0``, but stays finite for large-magnitude ``z``.
    """
    labels = np.asarray(y, dtype=float).ravel()
    scores = np.asarray(logits, dtype=float).ravel()
    if labels.shape != scores.shape:
        raise ValueError("y and logits must have the same number of elements")
    if labels.size == 0:
        raise ValueError("y and logits must not be empty")
    if not np.isin(labels, (0.0, 1.0)).all():
        raise ValueError("y must contain only 0 and 1 labels")
    terms = np.where(
        labels == 1.0,
        np.logaddexp(0.0, -scores),
        np.logaddexp(0.0, scores),
    )
    return float(np.mean(terms))


def logistic_loss(X: np.ndarray, y: np.ndarray, theta: np.ndarray) -> float:
    """Mean cross-entropy of the linear model ``X @ theta``.

    ``X`` is the design matrix including the intercept column (see
    ``add_intercept``).
    """
    return cross_entropy(y, _scores(X, theta))


def logistic_gradient(X: np.ndarray, y: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Gradient of ``logistic_loss`` with respect to ``theta``.

    Returns the analytic expression
    ``X.T @ (sigmoid(X @ theta) - y) / n_samples``; the tests compare it
    against central finite differences of ``logistic_loss``.
    """
    design = np.asarray(X, dtype=float)
    labels = np.asarray(y, dtype=float).ravel()
    if labels.shape != (design.shape[0],):
        raise ValueError("y must supply one label per design-matrix row")
    return design.T @ (sigmoid(_scores(design, theta)) - labels) / design.shape[0]


def _loss_and_gradient(
    design: np.ndarray, labels: np.ndarray, theta: np.ndarray
) -> tuple[float, np.ndarray]:
    """Mean cross-entropy and its gradient from one shared ``design @ theta``.

    ``fit`` calls this once per optimization state, so it forms the logits once
    and derives both quantities from that array instead of computing a second
    ``design @ theta`` inside ``logistic_gradient``. The public
    ``logistic_loss`` and ``logistic_gradient`` stay independent entry points
    for callers and for the finite-difference tests.
    """
    logits = design @ theta
    loss = cross_entropy(labels, logits)
    gradient = design.T @ (sigmoid(logits) - labels) / design.shape[0]
    return loss, gradient


# %% Fitting
@dataclass(frozen=True)
class FitResult:
    """Outcome of full-batch gradient descent.

    ``loss_history[k]`` and ``gradient_norm_history[k]`` hold the mean
    cross-entropy and the largest absolute gradient component measured at the
    parameters after exactly ``k`` updates. Index 0 describes the zero
    initialization, ``steps`` counts the parameter updates performed, and the
    histories hold ``steps + 1`` entries. The last entry always describes the
    returned ``theta``, including when ``max_steps`` capped the run after its
    final update.
    """

    theta: np.ndarray
    loss_history: tuple[float, ...]
    gradient_norm_history: tuple[float, ...]
    steps: int
    converged: bool
    diverged: bool


def fit(
    X: np.ndarray,
    y: np.ndarray,
    learning_rate: float = 0.01,
    max_steps: int = MAX_STEPS,
    tolerance: float = GRADIENT_TOLERANCE,
) -> FitResult:
    """Fit logistic regression by handwritten full-batch gradient descent.

    Weights start at zero. Every optimization state is evaluated with a single
    ``design @ theta`` product that yields both the stable mean
    cross-entropy and its gradient. The run stops when ``max|gradient|`` falls
    below ``tolerance`` (``converged``) or after ``max_steps`` parameter
    updates, whichever comes first. A non-finite loss, gradient or parameter
    update stops the run immediately and marks it ``diverged`` so callers can
    exclude it from model selection.

    The returned ``theta`` is always the parameters of the last recorded
    history entry. The zero initialization is recorded first, and a candidate
    update is committed only once its loss and gradient are known to be
    finite. A run capped by ``max_steps`` therefore reports the state produced
    by its final update rather than a stale pre-update one.
    """
    features = np.asarray(X, dtype=float)
    labels = np.asarray(y, dtype=float).ravel()
    if features.ndim != 2:
        raise ValueError("X must be a 2-D array of shape (n_samples, n_features)")
    if features.shape[0] == 0:
        raise ValueError("X must contain at least one sample")
    if labels.shape != (features.shape[0],):
        raise ValueError("y must supply one label per sample")
    if not np.isin(labels, (0.0, 1.0)).all():
        raise ValueError("y must contain only 0 and 1 labels")
    if not np.isfinite(features).all():
        raise ValueError("X must contain only finite values")
    rate = float(learning_rate)
    if not rate > 0.0:
        raise ValueError("learning_rate must be positive")
    steps_limit = int(max_steps)
    if steps_limit < 1:
        raise ValueError("max_steps must be at least 1")
    limit = float(tolerance)
    if not limit > 0.0:
        raise ValueError("tolerance must be positive")

    design = add_intercept(features)
    theta = np.zeros(design.shape[1], dtype=float)
    losses: list[float] = []
    norms: list[float] = []
    converged = False
    diverged = False
    steps = 0

    # ``loss_history[k]`` and ``gradient_norm_history[k]`` describe ``theta``
    # after exactly ``k`` updates, starting with the zero initialization at
    # index 0. An update is committed only after its loss and gradient are
    # known to be finite, so the returned ``theta`` is always the state of the
    # final history entry, including the update that a ``max_steps`` cap leaves
    # as the last one.
    loss, gradient = _loss_and_gradient(design, labels, theta)
    while True:
        norm = float(np.max(np.abs(gradient)))
        if not (np.isfinite(loss) and np.isfinite(norm)):
            diverged = True
            break
        losses.append(float(loss))
        norms.append(norm)
        if norm < limit:
            converged = True
            break
        if steps >= steps_limit:
            break
        candidate = theta - rate * gradient
        if not np.isfinite(candidate).all():
            diverged = True
            break
        candidate_loss, candidate_gradient = _loss_and_gradient(design, labels, candidate)
        if not (np.isfinite(candidate_loss) and np.isfinite(candidate_gradient).all()):
            diverged = True
            break
        theta = candidate
        loss, gradient = candidate_loss, candidate_gradient
        steps += 1
    return FitResult(
        theta=theta,
        loss_history=tuple(losses),
        gradient_norm_history=tuple(norms),
        steps=steps,
        converged=converged,
        diverged=diverged,
    )


# %% Train-only scaling
@dataclass(frozen=True)
class FeatureScaler:
    """Z-score scaler with statistics stored from one training fit.

    ``mean`` and ``scale`` are the population mean and standard deviation of
    the fitting data. Zero-variance columns store a scale of 1.0 so they
    transform to exact zeros instead of dividing by zero. Later data is
    transformed with the stored statistics and never refits them.
    """

    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, X: np.ndarray) -> FeatureScaler:
        """Learn the per-column mean and standard deviation from ``X``."""
        features = np.asarray(X, dtype=float)
        if features.ndim != 2:
            raise ValueError("X must be a 2-D array of shape (n_samples, n_features)")
        if features.shape[0] == 0:
            raise ValueError("X must contain at least one sample")
        if not np.isfinite(features).all():
            raise ValueError("X must contain only finite values")
        mean = features.mean(axis=0)
        scale = features.std(axis=0)
        scale = np.where(scale > 0.0, scale, 1.0)
        return cls(mean=mean, scale=scale)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Scale ``X`` with the stored statistics."""
        features = np.asarray(X, dtype=float)
        if features.ndim != 2:
            raise ValueError("X must be a 2-D array of shape (n_samples, n_features)")
        if features.shape[1] != self.mean.shape[0]:
            raise ValueError(
                f"X has {features.shape[1]} columns but the scaler was fitted on "
                f"{self.mean.shape[0]}"
            )
        return (features - self.mean) / self.scale


# %% Prediction and metrics
def predict_proba(X: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Positive-class probabilities for a fitted ``theta``."""
    return sigmoid(_scores(add_intercept(X), theta))


def predict(X: np.ndarray, theta: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Hard 0/1 labels, where a probability at or above ``threshold`` predicts 1."""
    cutoff = float(threshold)
    if not 0.0 <= cutoff <= 1.0:
        raise ValueError("threshold must lie in [0, 1]")
    return (predict_proba(X, theta) >= cutoff).astype(int)


def binary_metrics(
    y_true: np.ndarray, probabilities: np.ndarray, threshold: float = 0.5
) -> dict[str, Any]:
    """Metrics for positive class ``1`` at a fixed decision threshold.

    Accuracy, precision, recall, F1 and the confusion matrix use the
    thresholded labels; AUC uses the probabilities and ignores the threshold.
    The confusion matrix is ordered
    ``[[true negatives, false positives], [false negatives, true positives]]``.
    """
    labels = np.asarray(y_true, dtype=int).ravel()
    scores = np.asarray(probabilities, dtype=float).ravel()
    if labels.shape != scores.shape:
        raise ValueError("y_true and probabilities must have the same number of elements")
    if labels.size == 0:
        raise ValueError("y_true and probabilities must not be empty")
    predictions = (scores >= float(threshold)).astype(int)
    true_negatives, false_positives, false_negatives, true_positives = confusion_matrix(
        labels, predictions, labels=[0, 1]
    ).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(labels, predictions)),
        "positive_precision": float(
            precision_score(labels, predictions, pos_label=POSITIVE_LABEL, zero_division=0)
        ),
        "positive_recall": float(
            recall_score(labels, predictions, pos_label=POSITIVE_LABEL, zero_division=0)
        ),
        "positive_f1": float(
            f1_score(labels, predictions, pos_label=POSITIVE_LABEL, zero_division=0)
        ),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "confusion_matrix": [
            [int(true_negatives), int(false_positives)],
            [int(false_negatives), int(true_positives)],
        ],
    }


# %% Experiment helpers
def _class_counts(labels: np.ndarray) -> dict[str, int]:
    """Counts of benign (0) and malignant (1) samples."""
    return {
        "benign": int(np.sum(labels == 0)),
        "malignant": int(np.sum(labels == 1)),
    }


def _summarize(values: Sequence[float]) -> dict[str, float]:
    """Mean and population standard deviation across repeats."""
    samples = np.asarray(values, dtype=float)
    return {"mean": float(samples.mean()), "std": float(samples.std())}


def _elapsed_ms(start: float) -> float:
    """Milliseconds elapsed since a ``time.perf_counter`` reading."""
    return float(round((time.perf_counter() - start) * 1000.0, 3))


def _dev_f1_curve(y_dev: np.ndarray, probabilities: np.ndarray) -> list[float]:
    """Development positive-class F1 for every threshold in the grid."""
    return [
        float(
            f1_score(
                y_dev,
                (probabilities >= cutoff).astype(int),
                pos_label=POSITIVE_LABEL,
                zero_division=0,
            )
        )
        for cutoff in THRESHOLD_GRID
    ]


# %% Experiment protocol
def _build_protocol(
    features: np.ndarray, labels: np.ndarray, seeds: Sequence[int]
) -> dict[str, Any]:
    """Describe the dataset, splits, preprocessing, optimization and selection."""
    return {
        "provenance": {
            "dataset": "scikit-learn bundled breast-cancer dataset",
            "loader": "sklearn.datasets.load_breast_cancer()",
            "network_access_required": False,
            "n_samples": int(features.shape[0]),
            "n_features": int(features.shape[1]),
            "class_labels": {"benign": 0, "malignant": 1},
            "class_counts": _class_counts(labels),
            "label_remap": {
                "source_malignant": 0,
                "source_benign": 1,
                "note": (
                    "The bundled dataset encodes malignant as 0. This experiment remaps "
                    "malignant to the positive class 1 and benign to 0 before splitting."
                ),
            },
            "coursework": (
                "This experiment reimplements the CSC781 Module 6 Assignment 6 protocol. "
                "It corrects normalization and selection of the learning rate and threshold. "
                "It reuses no historical outputs or report scores."
            ),
        },
        "preprocessing": {
            "scaler": "z-score using the population mean and standard deviation",
            "fit_on": (
                "The scaler fits training data only. It uses the stored statistics "
                "to transform development and test data."
            ),
            "constant_features": (
                "A zero-variance training column keeps scale 1.0, so its transformed values "
                "are 0.0."
            ),
        },
        "splits": {
            "strategy": "stratified by class label into 60% train, 20% dev and 20% test",
            "train_fraction": 0.6,
            "dev_fraction": 0.2,
            "test_fraction": 0.2,
            "seed_policy": (
                "Each seed independently re-splits all samples. Selection uses only the "
                "dev split, and the held-out test split is scored after selection."
            ),
            "seeds": [int(seed) for seed in seeds],
        },
        "hyperparameters": {
            "method": "full-batch gradient descent on the mean binary cross-entropy",
            "weights_initialization": "all zeros",
            "learning_rates": [float(rate) for rate in LEARNING_RATES],
            "max_steps": MAX_STEPS,
            "tolerance": GRADIENT_TOLERANCE,
            "convergence_policy": (
                "Each step computes the full-batch gradient. The run stops when "
                "max|gradient| < tolerance or after max_steps, whichever comes first. "
                "A non-finite loss or update stops the run, marks it diverged and "
                "excludes it from model selection."
            ),
        },
        "selection": {
            "criterion": (
                "Highest development positive-class F1 across learning rates and "
                "thresholds among runs that did not diverge."
            ),
            "threshold_grid": {
                "start": 0.1,
                "stop": 0.9,
                "step": 0.05,
                "values": [float(value) for value in THRESHOLD_GRID],
            },
            "positive_class": {
                "label": POSITIVE_LABEL,
                "meaning": "malignant",
            },
            "tie_break": "smallest threshold, then smallest learning rate",
            "held_out": "Each model evaluates the test split once per seed after selection.",
        },
        "metrics": {
            "accuracy": "fraction of correctly labelled held-out samples",
            "positive_precision": "precision for malignant cases, positive class 1",
            "positive_recall": "recall for malignant cases, positive class 1",
            "positive_f1": "F1 for malignant cases, positive class 1",
            "roc_auc": "threshold-independent AUC over the predicted probabilities",
            "confusion_matrix": (
                "[[true negatives, false positives], [false negatives, true positives]]"
            ),
        },
        "baseline": {
            "estimator": "sklearn.linear_model.LogisticRegression",
            "parameters": {
                "C": "np.inf (unregularized)",
                "fit_intercept": True,
                "solver": "lbfgs",
                "max_iter": 10000,
                "tol": 1e-10,
            },
            "training_data": (
                "The same scaled training split and label remap as the NumPy implementation."
            ),
            "threshold_selection": (
                "An independent development positive-class F1 sweep over the same threshold grid."
            ),
            "role": "independent implementation check on identical data",
        },
        "repeats": {
            "seeds": [int(seed) for seed in seeds],
            "purpose": "report seed-to-seed variation of the selected protocol",
        },
        "intended_use": (
            "This experiment uses a small teaching dataset. "
            "Do not use its outputs for diagnosis or treatment."
        ),
    }


def run_seed(features: np.ndarray, labels: np.ndarray, seed: int) -> dict[str, Any]:
    """Run the split, fit, selection and evaluation stages for one seed.

    The scaler is fitted on the training split only. Every learning rate is
    fitted on the training split by full-batch gradient descent. The learning
    rate and decision threshold are then chosen jointly by the highest
    unrounded development positive-class F1. Only afterwards is the selected
    model scored on the untouched test split. An unregularized
    ``LogisticRegression`` baseline runs the same preprocessing and its own
    independent development threshold.
    """
    split = stratified_split(labels, seed)
    train_index, dev_index, test_index = split.train, split.dev, split.test
    y_train = labels[train_index]
    y_dev = labels[dev_index]
    y_test = labels[test_index]

    scaler = FeatureScaler.fit(features[train_index])
    X_train = scaler.transform(features[train_index])
    X_dev = scaler.transform(features[dev_index])
    X_test = scaler.transform(features[test_index])

    rate_runs: list[dict[str, Any]] = []
    dev_probabilities: dict[float, np.ndarray] = {}
    dev_f1_curves: dict[float, list[float]] = {}
    thetas: dict[float, np.ndarray] = {}
    for rate in LEARNING_RATES:
        start = time.perf_counter()
        result = fit(
            X_train,
            y_train,
            learning_rate=rate,
            max_steps=MAX_STEPS,
            tolerance=GRADIENT_TOLERANCE,
        )
        fit_ms = _elapsed_ms(start)

        start = time.perf_counter()
        probabilities = predict_proba(X_dev, result.theta)
        curve = _dev_f1_curve(y_dev, probabilities)
        selection_ms = _elapsed_ms(start)

        best_index = int(np.argmax(curve))
        rate_runs.append(
            {
                "learning_rate": float(rate),
                "steps": int(result.steps),
                "converged": bool(result.converged),
                "diverged": bool(result.diverged),
                "eligible_for_selection": bool(
                    not result.diverged and np.isfinite(result.theta).all()
                ),
                "initial_loss": float(result.loss_history[0]),
                "final_loss": float(result.loss_history[-1]),
                "final_gradient_norm": float(result.gradient_norm_history[-1]),
                "loss_curve": [float(value) for value in result.loss_history],
                "gradient_norm_curve": {
                    "stride": GRADIENT_CURVE_STRIDE,
                    "values": [
                        float(value)
                        for value in result.gradient_norm_history[::GRADIENT_CURVE_STRIDE]
                    ],
                },
                "dev_f1_by_threshold": [float(value) for value in curve],
                "dev_best_threshold": float(THRESHOLD_GRID[best_index]),
                "dev_best_f1": float(curve[best_index]),
                "fit_ms": fit_ms,
                "selection_ms": selection_ms,
            }
        )
        dev_probabilities[float(rate)] = probabilities
        dev_f1_curves[float(rate)] = curve
        thetas[float(rate)] = result.theta

    eligible = [entry for entry in rate_runs if entry["eligible_for_selection"]]
    if not eligible:
        raise RuntimeError(f"seed {seed}: every learning rate diverged, no model can be selected")
    # Ranking uses the unrounded development F1: rounding to a fixed number of
    # decimals is a presentation concern applied only after the (learning
    # rate, threshold) pair is fixed, never part of the comparison itself.
    candidates = [
        (
            dev_f1_curves[entry["learning_rate"]][index],
            -THRESHOLD_GRID[index],
            -entry["learning_rate"],
            entry["learning_rate"],
            THRESHOLD_GRID[index],
        )
        for entry in eligible
        for index in range(len(THRESHOLD_GRID))
    ]
    _, _, _, selected_rate, selected_threshold = max(candidates)

    # The selected learning rate's parameters are evaluated; the last fitted
    # model is never used by accident. ``selection.dev_f1`` is read back from
    # these metrics so the reported score and the selected run cannot drift.
    selected_dev_metrics = binary_metrics(
        y_dev, dev_probabilities[selected_rate], selected_threshold
    )

    start = time.perf_counter()
    test_probabilities = predict_proba(X_test, thetas[selected_rate])
    final_metrics = binary_metrics(y_test, test_probabilities, selected_threshold)
    eval_ms = _elapsed_ms(start)

    start = time.perf_counter()
    baseline = LogisticRegression(
        C=np.inf,
        fit_intercept=True,
        solver="lbfgs",
        max_iter=10000,
        tol=1e-10,
    )
    baseline.fit(X_train, y_train)
    baseline_fit_ms = _elapsed_ms(start)

    start = time.perf_counter()
    positive_column = list(baseline.classes_).index(POSITIVE_LABEL)
    baseline_dev = baseline.predict_proba(X_dev)[:, positive_column]
    baseline_curve = _dev_f1_curve(y_dev, baseline_dev)
    baseline_best_index = int(np.argmax(baseline_curve))
    baseline_threshold = float(THRESHOLD_GRID[baseline_best_index])
    baseline_selection_ms = _elapsed_ms(start)

    start = time.perf_counter()
    baseline_test = baseline.predict_proba(X_test)[:, positive_column]
    baseline_final = binary_metrics(y_test, baseline_test, baseline_threshold)
    baseline_eval_ms = _elapsed_ms(start)

    return {
        "seed": int(seed),
        "split": {
            "sizes": {
                "train": int(train_index.size),
                "dev": int(dev_index.size),
                "test": int(test_index.size),
            },
            "class_counts": {
                "full": _class_counts(labels),
                "train": _class_counts(y_train),
                "dev": _class_counts(y_dev),
                "test": _class_counts(y_test),
            },
        },
        "rates": rate_runs,
        "selection": {
            "learning_rate": float(selected_rate),
            "threshold": float(selected_threshold),
            "dev_f1": float(selected_dev_metrics["positive_f1"]),
            "dev_metrics": selected_dev_metrics,
        },
        "final": final_metrics,
        "baseline": {
            "n_iter": int(baseline.n_iter_[0]),
            "positive_column": int(positive_column),
            "selected_threshold": baseline_threshold,
            "dev_f1": float(baseline_curve[baseline_best_index]),
            "dev_f1_by_threshold": [float(value) for value in baseline_curve],
            "final": baseline_final,
        },
        "timings_ms": {
            "fits": {str(entry["learning_rate"]): entry["fit_ms"] for entry in rate_runs},
            "fit_total": float(round(sum(entry["fit_ms"] for entry in rate_runs), 3)),
            "threshold_selection": float(
                round(sum(entry["selection_ms"] for entry in rate_runs), 3)
            ),
            "final_evaluation": eval_ms,
            "baseline_fit": baseline_fit_ms,
            "baseline_selection": baseline_selection_ms,
            "baseline_evaluation": baseline_eval_ms,
        },
    }


def summarize_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Mean and standard deviation of the held-out metrics across seeds."""
    metric_names = (
        "accuracy",
        "positive_precision",
        "positive_recall",
        "positive_f1",
        "roc_auc",
    )
    return {
        "seeds": [int(run["seed"]) for run in runs],
        "custom": {name: _summarize([run["final"][name] for run in runs]) for name in metric_names},
        "baseline": {
            name: _summarize([run["baseline"]["final"][name] for run in runs])
            for name in metric_names
        },
        "selections": [
            {
                "seed": int(run["seed"]),
                "learning_rate": run["selection"]["learning_rate"],
                "threshold": run["selection"]["threshold"],
                "baseline_threshold": run["baseline"]["selected_threshold"],
            }
            for run in runs
        ],
    }


def run_experiment(seeds: Sequence[int] = DEFAULT_SEEDS) -> dict[str, Any]:
    """Run the protocol for every seed and return the JSON-ready evidence payload."""
    seed_list = [int(seed) for seed in seeds]
    if not seed_list:
        raise ValueError("at least one seed is required")
    dataset = load_breast_cancer()
    features = np.asarray(dataset.data, dtype=float)
    observed = np.asarray(dataset.target, dtype=int)
    # The bundled dataset encodes malignant as 0 and benign as 1. The remap
    # makes malignant the positive class, so precision, recall and F1 describe
    # malignant cases.
    labels = (observed == 0).astype(int)
    runs = [run_seed(features, labels, seed) for seed in seed_list]
    return {
        "protocol": _build_protocol(features, labels, seed_list),
        "environment": environment(),
        "runs": runs,
        "summary": summarize_runs(runs),
    }


# %% Figures
def save_figure(figure: plt.Figure, output_path: Path) -> None:
    """Save a figure as a PNG and release it."""
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def plot_loss_curves(runs: Sequence[dict[str, Any]], output_path: Path) -> None:
    """Plot training loss against gradient-descent step, one panel per seed.

    ``loss_curve[k]`` is the training loss after ``k`` parameter updates, so
    the x axis counts completed updates and starts at the initial weights.
    """
    figure, axes = plt.subplots(
        1, len(runs), figsize=(5.0 * len(runs), 4.0), sharey=True, squeeze=False
    )
    for column, run in enumerate(runs):
        axis = axes[0][column]
        for entry in run["rates"]:
            curve = np.maximum(np.asarray(entry["loss_curve"], dtype=float), 1e-12)
            steps = np.arange(curve.size)
            axis.plot(steps, curve, linewidth=1.4, label=f"rate {entry['learning_rate']}")
        axis.set_title(f"seed {run['seed']}")
        axis.set_xlabel("Completed gradient-descent steps\nStep 0 uses the initial weights")
        axis.set_yscale("log")
        axis.grid(alpha=0.3)
    axes[0][0].set_ylabel("Mean binary cross-entropy on training data")
    axes[0][-1].legend(fontsize=8)
    figure.suptitle("Training loss by learning rate\nAt most 3000 full-batch steps")
    figure.tight_layout()
    save_figure(figure, output_path)


def plot_comparison(
    runs: Sequence[dict[str, Any]],
    summary: dict[str, Any],
    output_path: Path,
) -> None:
    """Plot development threshold selection and test metrics versus the baseline."""
    figure, (left, right) = plt.subplots(1, 2, figsize=(11.0, 4.5))

    thresholds = np.asarray(THRESHOLD_GRID)
    rates = sorted({entry["learning_rate"] for run in runs for entry in run["rates"]})
    for rate in rates:
        curves = [
            np.asarray(entry["dev_f1_by_threshold"], dtype=float)
            for run in runs
            for entry in run["rates"]
            if entry["learning_rate"] == rate
        ]
        stacked = np.vstack(curves)
        left.plot(thresholds, stacked.mean(axis=0), linewidth=1.8, label=f"rate {rate}")
        left.plot(thresholds, stacked.T, linewidth=0.6, alpha=0.35)
    left.scatter(
        [run["selection"]["threshold"] for run in runs],
        [run["selection"]["dev_f1"] for run in runs],
        marker="*",
        s=90,
        color="black",
        zorder=5,
        label="selected",
    )
    left.set_xlabel("decision threshold")
    left.set_ylabel("development positive-class F1")
    left.set_title("Threshold selection on the development split")
    left.grid(alpha=0.3)
    left.legend(fontsize=8)

    metric_names = (
        "accuracy",
        "positive_precision",
        "positive_recall",
        "positive_f1",
        "roc_auc",
    )
    display_names = ("accuracy", "precision", "recall", "F1", "AUC")
    positions = np.arange(len(metric_names))
    width = 0.38
    series = (
        (-width / 2.0, "custom", "NumPy gradient descent"),
        (width / 2.0, "baseline", "sklearn (C=inf)"),
    )
    for offset, name, label in series:
        means = [summary[name][metric]["mean"] for metric in metric_names]
        errors = [summary[name][metric]["std"] for metric in metric_names]
        right.bar(positions + offset, means, width, yerr=errors, capsize=3, label=label)
    right.set_xticks(positions, display_names)
    right.set_ylim(0.0, 1.05)
    right.set_ylabel("Test metric\nMean and population standard deviation across seeds")
    right.set_title("Selected model versus unregularized baseline")
    right.grid(axis="y", alpha=0.3)
    right.legend(fontsize=8)

    figure.tight_layout()
    save_figure(figure, output_path)


def write_figures(
    runs: Sequence[dict[str, Any]],
    summary: dict[str, Any],
    output_dir: Path,
) -> dict[str, Path]:
    """Write both PNG figures and return their paths keyed by role."""
    output_dir.mkdir(parents=True, exist_ok=True)
    loss_path = output_dir / LOSS_FIGURE
    comparison_path = output_dir / COMPARISON_FIGURE
    plot_loss_curves(runs, loss_path)
    plot_comparison(runs, summary, comparison_path)
    return {"loss": loss_path, "comparison": comparison_path}


# %% Entry point
def main(argv: list[str] | None = None) -> int:
    """Run the seeded protocol and write JSON evidence plus PNG figures."""
    parser = argparse.ArgumentParser(
        prog="python -m mlshowcase.logistic",
        description="Stable logistic regression on the scikit-learn breast-cancer dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="directory for logistic.json and the PNG figures (default: results/)",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_SEEDS),
        help="seeds; each one repeats the full split, selection and evaluation protocol",
    )
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = run_experiment(args.seeds)
    figure_paths = write_figures(payload["runs"], payload["summary"], output_dir)
    payload["figures"] = {role: path.name for role, path in figure_paths.items()}
    write_results(output_dir, RESULT_NAME, payload)

    for run in payload["runs"]:
        selection = run["selection"]
        print(
            f"seed {run['seed']}: rate={selection['learning_rate']} "
            f"threshold={selection['threshold']} dev_f1={selection['dev_f1']:.4f} "
            f"test_f1={run['final']['positive_f1']:.4f}"
        )
    print(f"wrote {RESULT_NAME}.json and {len(figure_paths)} figures to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
