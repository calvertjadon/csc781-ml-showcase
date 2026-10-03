"""Handwritten k-nearest neighbours on the scikit-learn digits dataset.

The module computes Euclidean, Manhattan and Minkowski distances in NumPy and
provides a ``KNNClassifier`` that stores the training split and ranks its rows
at prediction time. It returns the original label values, so class labels need
not be contiguous, resolves vote ties toward the smallest class label, keeps
equal distances in training-row order, and ranks every training row once, which
lets a caller sweep ``k`` without recomputing distances.

The experiment reruns the CSC781 Module 2 Assignment 2 protocol with fixed
seeds and a stratified split into train, dev and test parts. It selects one
distance and ``k`` on the dev split, scores the test split only afterwards, and
compares the handwritten classifier with
``sklearn.neighbors.KNeighborsClassifier`` on identical rows. Run it from the
repository root::

    uv run --locked python -m mlshowcase.knn --output-dir results --seeds 42 43 44

The run writes ``knn.json`` and two PNG figures into the output directory.
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
from sklearn.datasets import load_digits
from sklearn.exceptions import NotFittedError
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.neighbors import KNeighborsClassifier

from mlshowcase.common import environment, stratified_split, write_results

# Figures are always written to PNG files, so select the headless Agg backend,
# which behaves the same in a terminal and in headless runs.
plt.switch_backend("Agg")

__all__ = [
    "BASELINE_FIGURE",
    "DEFAULT_SEEDS",
    "DEV_FRACTION",
    "DISTANCE_GRID",
    "K_GRID",
    "MINKOWSKI_P",
    "RESULT_NAME",
    "SWEEP_FIGURE",
    "TRAIN_FRACTION",
    "DistanceSpec",
    "KNNClassifier",
    "NeighborRanking",
    "classification_summary",
    "euclidean_distance",
    "load_digits_data",
    "main",
    "majority_vote",
    "manhattan_distance",
    "minkowski_distance",
    "rank_neighbors",
    "run_experiment",
    "run_seed",
    "summarize_runs",
    "write_figures",
]

DEFAULT_SEEDS: tuple[int, ...] = (42, 43, 44)
K_GRID: tuple[int, ...] = (1, 3, 5, 7, 9, 11, 15)
MINKOWSKI_P = 1.5
TRAIN_FRACTION = 0.6
DEV_FRACTION = 0.2
RESULT_NAME = "knn"
SWEEP_FIGURE = "knn_dev_sweep.png"
BASELINE_FIGURE = "knn_baseline.png"

_DISTANCE_METRICS: tuple[str, ...] = ("euclidean", "manhattan", "minkowski")
_QUERY_CHUNK = 128


# %% Distances
def _validated_p(metric: str, p: float | None) -> float | None:
    """Validate a metric name and its Minkowski exponent, returning the exponent."""
    if metric not in _DISTANCE_METRICS:
        raise ValueError(
            f"unknown metric {metric!r}; expected one of {', '.join(_DISTANCE_METRICS)}"
        )
    if metric == "minkowski":
        if p is None:
            raise ValueError("minkowski distance requires an explicit exponent p >= 1")
        exponent = float(p)
        if exponent < 1.0:
            raise ValueError(f"minkowski exponent must be >= 1, got {exponent}")
        return exponent
    if p is not None:
        raise ValueError(f"metric {metric!r} does not take a Minkowski exponent")
    return None


def _distance_from_diff(differences: np.ndarray, metric: str, exponent: float | None) -> np.ndarray:
    """Reduce component-wise differences along the last axis with ``metric``.

    ``exponent`` is validated by ``_validated_p`` and only used by Minkowski.
    """
    if metric == "euclidean":
        return np.sqrt(np.sum(differences * differences, axis=-1))
    if metric == "manhattan":
        return np.sum(np.abs(differences), axis=-1)
    if exponent is None:
        raise ValueError("minkowski distance requires an explicit exponent p >= 1")
    return np.sum(np.abs(differences) ** exponent, axis=-1) ** (1.0 / exponent)


def _paired_difference(a: Any, b: Any) -> np.ndarray:
    """Component-wise difference of two equally shaped feature vectors."""
    first = np.asarray(a, dtype=float)
    second = np.asarray(b, dtype=float)
    if first.shape != second.shape:
        raise ValueError(f"vectors must have matching shapes, got {first.shape} and {second.shape}")
    return first - second


def euclidean_distance(a: Any, b: Any) -> float:
    """Euclidean (L2) distance over every component of two feature vectors."""
    return float(_distance_from_diff(_paired_difference(a, b), "euclidean", None))


def manhattan_distance(a: Any, b: Any) -> float:
    """Manhattan (L1) distance over every component of two feature vectors."""
    return float(_distance_from_diff(_paired_difference(a, b), "manhattan", None))


def minkowski_distance(a: Any, b: Any, p: float | None = None) -> float:
    """Minkowski distance ``(sum |a - b| ** p) ** (1 / p)`` for ``p >= 1``."""
    exponent = _validated_p("minkowski", p)
    return float(_distance_from_diff(_paired_difference(a, b), "minkowski", exponent))


# %% Neighbour ranking
@dataclass(frozen=True, eq=False)
class NeighborRanking:
    """Training neighbours for every query row, sorted by distance.

    ``indices[i, j]`` is the training row at rank ``j`` for query ``i`` and
    ``distances[i, j]`` is its distance. Distances are sorted with a stable
    sort, so equal distances keep their original training-row order.
    """

    indices: np.ndarray
    distances: np.ndarray


def _feature_matrix(values: Any, name: str) -> np.ndarray:
    """Convert values to a finite 2-D float array."""
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be 2-D (n_samples, n_features), got {matrix.ndim}-D")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} must contain only finite values")
    return matrix


def _label_vector(values: Any, name: str, expected: int | None = None) -> np.ndarray:
    """Convert values to a 1-D label array, optionally checking its length."""
    labels = np.asarray(values)
    if labels.ndim != 1:
        raise ValueError(f"{name} must be 1-D (n_samples,), got {labels.ndim}-D")
    if expected is not None and labels.shape[0] != expected:
        raise ValueError(f"{name} has {labels.shape[0]} labels but {expected} samples were given")
    return labels


def _rank_validated(
    train: np.ndarray,
    query: np.ndarray,
    metric: str,
    exponent: float | None,
) -> NeighborRanking:
    """Rank already-validated feature matrices by distance."""
    if train.shape[0] == 0:
        raise ValueError("at least one training row is required")
    if train.shape[1] != query.shape[1]:
        raise ValueError(
            f"training rows have {train.shape[1]} features but queries have {query.shape[1]}"
        )
    distances = np.empty((query.shape[0], train.shape[0]), dtype=float)
    # Broadcasting one query chunk at a time keeps the temporary
    # (chunk, n_train, n_features) difference array small.
    for start in range(0, query.shape[0], _QUERY_CHUNK):
        stop = start + _QUERY_CHUNK
        differences = query[start:stop, None, :] - train[None, :, :]
        distances[start:stop] = _distance_from_diff(differences, metric, exponent)
    order = np.argsort(distances, axis=1, kind="stable")
    return NeighborRanking(
        indices=order,
        distances=np.take_along_axis(distances, order, axis=1),
    )


def rank_neighbors(
    X_train: Any,
    X_query: Any,
    *,
    metric: str = "euclidean",
    p: float | None = None,
) -> NeighborRanking:
    """Rank every training row for every query row by distance.

    Distances are computed once per call, so a caller sweeping ``k`` can reuse
    a single ranking for every neighbourhood size.
    """
    exponent = _validated_p(metric, p)
    train = _feature_matrix(X_train, "X_train")
    query = _feature_matrix(X_query, "X_query")
    return _rank_validated(train, query, metric, exponent)


# %% Majority vote
def majority_vote(ranking: NeighborRanking, y_train: Any, k: int) -> np.ndarray:
    """Predict each query by majority vote among its ``k`` nearest neighbours.

    Vote-count ties resolve to the smallest class label (``argmax`` over
    class-sorted counts returns the first maximum). Equal-distance neighbours
    retain training-row order; this can affect membership at the ``k`` boundary.
    """
    train_labels = _label_vector(y_train, "y_train", ranking.indices.shape[1])
    size = int(k)
    available = ranking.indices.shape[1]
    if not 1 <= size <= available:
        raise ValueError(f"k must be between 1 and {available}, got {size}")

    classes = np.unique(train_labels)
    selected = train_labels[ranking.indices[:, :size]]
    codes = np.searchsorted(classes, selected)
    counts = np.zeros((selected.shape[0], classes.size), dtype=np.int64)
    rows = np.repeat(np.arange(selected.shape[0]), size)
    np.add.at(counts, (rows, codes.ravel()), 1)
    # np.argmax keeps the first maximum and classes_ are sorted, so an exact
    # vote tie deterministically resolves to the smallest class label.
    return classes[np.argmax(counts, axis=1)]


class KNNClassifier:
    """Handwritten k-nearest-neighbour classifier.

    Parameters
    ----------
    n_neighbors:
        Number of neighbours that vote on each prediction (``k``).
    metric:
        ``"euclidean"``, ``"manhattan"`` or ``"minkowski"``.
    p:
        Minkowski exponent, required for ``"minkowski"`` and rejected for the
        other metrics; values below 1 are rejected.
    """

    def __init__(
        self,
        n_neighbors: int = 5,
        *,
        metric: str = "euclidean",
        p: float | None = None,
    ) -> None:
        self.n_neighbors = int(n_neighbors)
        if self.n_neighbors < 1:
            raise ValueError(f"n_neighbors must be >= 1, got {n_neighbors}")
        self.metric = metric
        self.p = p
        self._exponent = _validated_p(metric, p)
        self._train_features: np.ndarray | None = None
        self._train_labels: np.ndarray | None = None
        self.classes_: np.ndarray | None = None
        self.n_features_in_: int | None = None

    def fit(self, X: Any, y: Any) -> KNNClassifier:
        """Store the training split; features and labels stay separate arrays."""
        features = _feature_matrix(X, "X")
        labels = _label_vector(y, "y", features.shape[0])
        if features.shape[0] == 0:
            raise ValueError("fit requires at least one training sample")
        self._train_features = features
        self._train_labels = labels
        self.classes_ = np.unique(labels)
        self.n_features_in_ = features.shape[1]
        return self

    def rank(self, X: Any) -> NeighborRanking:
        """Rank training rows for each query row; reuse one ranking across ``k``."""
        train_features, _ = self._check_fitted()
        query = _feature_matrix(X, "X")
        if query.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {query.shape[1]} features, expected {self.n_features_in_}")
        return _rank_validated(train_features, query, self.metric, self._exponent)

    def predict(self, X: Any, *, k: int | None = None) -> np.ndarray:
        """Predict labels for ``X`` using ``k`` neighbours (default ``n_neighbors``)."""
        train_features, train_labels = self._check_fitted()
        query = _feature_matrix(X, "X")
        if query.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {query.shape[1]} features, expected {self.n_features_in_}")
        ranking = _rank_validated(train_features, query, self.metric, self._exponent)
        return majority_vote(ranking, train_labels, self.n_neighbors if k is None else k)

    def _check_fitted(self) -> tuple[np.ndarray, np.ndarray]:
        if self._train_features is None or self._train_labels is None:
            raise NotFittedError("KNNClassifier must be fitted before it can predict")
        return self._train_features, self._train_labels


# %% Experiment protocol
@dataclass(frozen=True)
class DistanceSpec:
    """One distance metric in the sweep and its optional Minkowski exponent."""

    name: str
    p: float | None = None

    @property
    def label(self) -> str:
        """Label shown in figures and printouts."""
        if self.p is None:
            return self.name
        return f"{self.name} (p={self.p:g})"


DISTANCE_GRID: tuple[DistanceSpec, ...] = (
    DistanceSpec("euclidean"),
    DistanceSpec("manhattan"),
    DistanceSpec("minkowski", MINKOWSKI_P),
)


def load_digits_data() -> tuple[np.ndarray, np.ndarray]:
    """Load the bundled 8x8 digit images as float features and int64 labels."""
    bunch = load_digits()
    features = np.asarray(bunch.data, dtype=float)
    labels = np.asarray(bunch.target, dtype=np.int64)
    if features.shape[0] != labels.shape[0]:
        raise ValueError("digits features and labels disagree on sample count")
    return features, labels


def _class_counts(labels: np.ndarray, indices: np.ndarray) -> dict[str, int]:
    """Per-class sample counts for a split, keyed by class label as text."""
    classes, counts = np.unique(labels[indices], return_counts=True)
    return {str(int(value)): int(count) for value, count in zip(classes, counts, strict=True)}


def classification_summary(truth: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    """Accuracy, macro F1 and the confusion matrix as plain Python values."""
    classes = np.unique(truth)
    matrix = confusion_matrix(truth, predictions, labels=classes)
    return {
        "accuracy": float(accuracy_score(truth, predictions)),
        "macro_f1": float(f1_score(truth, predictions, average="macro", zero_division=0.0)),
        "confusion_matrix": [[int(value) for value in row] for row in matrix],
        "confusion_matrix_labels": [int(value) for value in classes],
    }


def _sklearn_baseline(n_neighbors: int, metric: str, p: float | None) -> KNeighborsClassifier:
    """Build the reference estimator with the selected hyperparameters.

    ``algorithm="brute"`` keeps the reference search exhaustive like the
    handwritten implementation and avoids tree-ordering differences.
    """
    exponent = _validated_p(metric, p)
    parameters: dict[str, Any] = {
        "n_neighbors": int(n_neighbors),
        "metric": metric,
        "algorithm": "brute",
    }
    if exponent is not None:
        parameters["p"] = exponent
    return KNeighborsClassifier(**parameters)


def run_seed(features: np.ndarray, labels: np.ndarray, seed: int) -> dict[str, Any]:
    """Run the full protocol for one seed: split, dev sweep, selection, held-out scoring."""
    split = stratified_split(labels, seed, train_fraction=TRAIN_FRACTION, dev_fraction=DEV_FRACTION)
    train_index = np.asarray(split.train)
    dev_index = np.asarray(split.dev)
    test_index = np.asarray(split.test)
    train_x, train_y = features[train_index], labels[train_index]
    dev_x, dev_y = features[dev_index], labels[dev_index]
    test_x, test_y = features[test_index], labels[test_index]

    # Development distances are computed once per metric; every k in the sweep
    # reuses the same ranking instead of recomputing distances.
    selection_start = time.perf_counter()
    rankings: dict[str, NeighborRanking] = {}
    ranking_seconds: dict[str, float] = {}
    for spec in DISTANCE_GRID:
        ranking_start = time.perf_counter()
        rankings[spec.name] = rank_neighbors(train_x, dev_x, metric=spec.name, p=spec.p)
        ranking_seconds[spec.name] = time.perf_counter() - ranking_start

    dev_curves: dict[str, dict[str, dict[str, float]]] = {spec.name: {} for spec in DISTANCE_GRID}
    selection: dict[str, Any] | None = None
    # k-major order: an exact dev-macro-F1 tie keeps the smallest k, and within
    # one k the first metric in DISTANCE_GRID wins (euclidean, manhattan, minkowski).
    for k in K_GRID:
        for spec in DISTANCE_GRID:
            predictions = majority_vote(rankings[spec.name], train_y, k)
            accuracy = float(accuracy_score(dev_y, predictions))
            macro_f1 = float(f1_score(dev_y, predictions, average="macro", zero_division=0.0))
            dev_curves[spec.name][str(k)] = {"accuracy": accuracy, "macro_f1": macro_f1}
            if selection is None or macro_f1 > selection["dev_macro_f1"]:
                selection = {
                    "metric": spec.name,
                    "p": spec.p,
                    "k": int(k),
                    "dev_accuracy": accuracy,
                    "dev_macro_f1": macro_f1,
                }
    selection_seconds = time.perf_counter() - selection_start
    if selection is None:
        raise RuntimeError("no sweep configuration was evaluated")

    # The held-out split is used only now, after the configuration is fixed.
    model = KNNClassifier(n_neighbors=selection["k"], metric=selection["metric"], p=selection["p"])
    fit_start = time.perf_counter()
    model.fit(train_x, train_y)
    fit_seconds = time.perf_counter() - fit_start

    evaluation_start = time.perf_counter()
    handwritten_predictions = model.predict(test_x)
    handwritten = classification_summary(test_y, handwritten_predictions)
    evaluation_seconds = time.perf_counter() - evaluation_start

    baseline = _sklearn_baseline(selection["k"], selection["metric"], selection["p"])
    baseline_fit_start = time.perf_counter()
    baseline.fit(train_x, train_y)
    baseline_fit_seconds = time.perf_counter() - baseline_fit_start
    baseline_predict_start = time.perf_counter()
    baseline_predictions = baseline.predict(test_x)
    baseline_predict_seconds = time.perf_counter() - baseline_predict_start

    return {
        "seed": int(seed),
        "split": {
            "train_fraction": TRAIN_FRACTION,
            "dev_fraction": DEV_FRACTION,
            "counts": {
                "train": int(train_index.size),
                "dev": int(dev_index.size),
                "test": int(test_index.size),
            },
            "class_counts": {
                "train": _class_counts(labels, train_index),
                "dev": _class_counts(labels, dev_index),
                "test": _class_counts(labels, test_index),
            },
        },
        "dev_curves": dev_curves,
        "selection": selection,
        "heldout": {
            "handwritten": handwritten,
            "sklearn": classification_summary(test_y, baseline_predictions),
            "predictions_match": bool(
                np.array_equal(handwritten_predictions, baseline_predictions)
            ),
        },
        "timings_seconds": {
            "handwritten_fit": fit_seconds,
            "dev_ranking": ranking_seconds,
            "dev_selection": selection_seconds,
            "handwritten_heldout_evaluation": evaluation_seconds,
            "sklearn_fit": baseline_fit_seconds,
            "sklearn_predict": baseline_predict_seconds,
        },
    }


def _selection_label(selection: dict[str, Any]) -> str:
    """Compact label for a selected configuration, e.g. ``manhattan, k=5``."""
    p = selection["p"]
    metric = selection["metric"] if p is None else f"{selection['metric']}(p={p:g})"
    return f"{metric}, k={selection['k']}"


def summarize_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate held-out scores, selections and dev curves across seed runs."""

    def collapsed(values: list[float]) -> dict[str, Any]:
        array = np.asarray(values, dtype=float)
        return {
            "values": [float(value) for value in array],
            "mean": float(array.mean()),
            "std": float(array.std()),
            "min": float(array.min()),
            "max": float(array.max()),
        }

    def mean_dev_curve(metric: str) -> dict[str, float]:
        return {
            str(k): float(np.mean([run["dev_curves"][metric][str(k)]["macro_f1"] for run in runs]))
            for k in K_GRID
        }

    counts: dict[str, int] = {}
    for run in runs:
        label = _selection_label(run["selection"])
        counts[label] = counts.get(label, 0) + 1
    modal = min(label for label, count in counts.items() if count == max(counts.values()))

    return {
        "n_seeds": len(runs),
        "handwritten": {
            "accuracy": collapsed([run["heldout"]["handwritten"]["accuracy"] for run in runs]),
            "macro_f1": collapsed([run["heldout"]["handwritten"]["macro_f1"] for run in runs]),
        },
        "sklearn": {
            "accuracy": collapsed([run["heldout"]["sklearn"]["accuracy"] for run in runs]),
            "macro_f1": collapsed([run["heldout"]["sklearn"]["macro_f1"] for run in runs]),
        },
        "selections": {
            "per_seed": {str(run["seed"]): _selection_label(run["selection"]) for run in runs},
            "counts": counts,
            "modal": modal,
            "modal_tie_break": "highest count, then lexicographic order of the label",
        },
        "predictions_match": {
            "per_seed": {str(run["seed"]): run["heldout"]["predictions_match"] for run in runs},
            "all_seeds": all(run["heldout"]["predictions_match"] for run in runs),
        },
        "dev_macro_f1_mean": {spec.name: mean_dev_curve(spec.name) for spec in DISTANCE_GRID},
        "std_kind": "population standard deviation across the recorded seeds",
    }


def _build_protocol(
    features: np.ndarray,
    labels: np.ndarray,
    seeds: list[int],
    runs: list[dict[str, Any]],
) -> dict[str, Any]:
    """Describe the dataset, preprocessing, splits, selection and baseline."""
    return {
        "provenance": {
            "dataset": "scikit-learn bundled handwritten digits with 8-by-8 grayscale images",
            "loader": "sklearn.datasets.load_digits()",
            "network_access_required": False,
            "n_samples": int(features.shape[0]),
            "n_features": int(features.shape[1]),
            "classes": [int(value) for value in np.unique(labels)],
            "coursework": (
                "This experiment reimplements the CSC781 Module 2 Assignment 2 k-NN protocol. "
                "It reuses no historical outputs or report scores."
            ),
        },
        "preprocessing": {
            "steps": "none",
            "detail": (
                "The code casts raw pixel intensities from 0 to 16 to float64. "
                "It applies no scaling, centering or feature selection. "
                "Every distance uses all 64 features."
            ),
        },
        "splits": {
            "strategy": "stratified by class label into 60% train, 20% dev and 20% test",
            "train_fraction": TRAIN_FRACTION,
            "dev_fraction": DEV_FRACTION,
            "test_fraction": round(1.0 - TRAIN_FRACTION - DEV_FRACTION, 10),
            "seed_policy": (
                "Each seed splits all samples again. Development data selects the "
                "configuration. Each model then evaluates the test split."
            ),
            "per_seed": {
                str(run["seed"]): {
                    "counts": run["split"]["counts"],
                    "class_counts": run["split"]["class_counts"],
                }
                for run in runs
            },
        },
        "seeds": seeds,
        "hyperparameters": {
            "k_grid": list(K_GRID),
            "distances": [{"metric": spec.name, "p": spec.p} for spec in DISTANCE_GRID],
            "vote_tie_break": "smallest class label",
            "neighbor_ordering": "stable sort, so equal distances keep training-row order",
        },
        "selection": {
            "criterion": "maximum macro F1 on the dev split",
            "tie_break": "smallest k, then metric order euclidean, manhattan, minkowski (p=1.5)",
            "held_out": "Each model evaluates the test split once per seed after selection.",
        },
        "metrics": {
            "accuracy": "fraction of correctly predicted held-out samples",
            "macro_f1": "unweighted mean of per-class F1 (zero_division=0)",
        },
        "baseline": {
            "estimator": "sklearn.neighbors.KNeighborsClassifier",
            "algorithm": "brute",
            "hyperparameters": "identical to the configuration selected on the dev split",
            "role": "independent implementation check on identical data, not a tuned competitor",
        },
    }


def run_experiment(seeds: Sequence[int] = DEFAULT_SEEDS) -> dict[str, Any]:
    """Run the protocol for every seed and return the JSON-ready evidence payload."""
    seed_list = [int(seed) for seed in seeds]
    if not seed_list:
        raise ValueError("at least one seed is required")
    features, labels = load_digits_data()
    runs = [run_seed(features, labels, seed) for seed in seed_list]
    return {
        "protocol": _build_protocol(features, labels, seed_list, runs),
        "environment": environment(),
        "runs": runs,
        "summary": summarize_runs(runs),
    }


# %% Figures
_METRIC_COLORS = {
    "euclidean": "tab:blue",
    "manhattan": "tab:orange",
    "minkowski": "tab:green",
}

# Every dev score sits in a narrow band near 1, so the sweep figure zooms its
# y-axis to the observed scores (individual seeds, metric means and selection
# stars) instead of showing a full 0-1 span that would compress every difference.
_ZOOM_PADDING_FRACTION = 0.05
_ZOOM_MIN_PADDING = 0.01


def _zoomed_axis_limits(values: Sequence[float]) -> tuple[float, float]:
    """Y-limits zoomed around ``values``, with modest padding and clipped to [0, 1]."""
    low = float(np.min(values))
    high = float(np.max(values))
    padding = _ZOOM_PADDING_FRACTION * (high - low)
    if padding <= 0.0:
        padding = _ZOOM_MIN_PADDING
    return max(0.0, low - padding), min(1.0, high + padding)


def save_figure(fig: plt.Figure, output_path: Path) -> None:
    """Save a figure as a PNG and release it."""
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_dev_curves(runs: list[dict[str, Any]], output_path: Path) -> None:
    """Plot dev macro F1 and accuracy against k, one colour per distance.

    The y-axes are zoomed to the observed scores because the sweep values sit in
    a narrow band near 1. The titles and axis labels state the restriction, and
    the baseline figure keeps the full 0-1 scale so effect sizes stay comparable.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), sharex=True)
    k_keys = [str(k) for k in K_GRID]
    panels = (
        (axes[0], "macro_f1", "macro F1"),
        (axes[1], "accuracy", "accuracy"),
    )
    plotted_values: dict[str, list[float]] = {score_key: [] for _, score_key, _ in panels}
    for spec in DISTANCE_GRID:
        color = _METRIC_COLORS[spec.name]
        for panel, score_key, _ in panels:
            for run in runs:
                curve = [run["dev_curves"][spec.name][key][score_key] for key in k_keys]
                panel.plot(K_GRID, curve, color=color, alpha=0.25, linewidth=1.0)
                plotted_values[score_key].extend(curve)
            mean_curve = [
                float(np.mean([run["dev_curves"][spec.name][key][score_key] for run in runs]))
                for key in k_keys
            ]
            panel.plot(K_GRID, mean_curve, color=color, marker="o", label=spec.label)
            plotted_values[score_key].extend(mean_curve)
    for run in runs:
        selection = run["selection"]
        color = _METRIC_COLORS[selection["metric"]]
        axes[0].plot(
            [selection["k"]],
            [selection["dev_macro_f1"]],
            marker="*",
            markersize=13,
            color=color,
            markeredgecolor="black",
            linestyle="none",
        )
        axes[1].plot(
            [selection["k"]],
            [selection["dev_accuracy"]],
            marker="*",
            markersize=13,
            color=color,
            markeredgecolor="black",
            linestyle="none",
        )
        plotted_values["macro_f1"].append(selection["dev_macro_f1"])
        plotted_values["accuracy"].append(selection["dev_accuracy"])
    for panel, score_key, score_name in panels:
        low, high = _zoomed_axis_limits(plotted_values[score_key])
        panel.set_title(f"Development {score_name} by k\nY-axis zoomed to {low:.3f} to {high:.3f}")
        panel.set_xlabel("Number of neighbors, k")
        panel.set_ylabel(f"{score_name}\nZoomed y-axis")
        panel.set_xticks(list(K_GRID))
        panel.set_ylim(low, high)
        panel.grid(True, axis="y", color="0.85", linewidth=0.6)
    axes[0].legend(title="Distance\nStars mark selections", fontsize=8)
    save_figure(fig, output_path)


def plot_baseline_comparison(runs: list[dict[str, Any]], output_path: Path) -> None:
    """Compare held-out scores and show the first seed's confusion matrices."""
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.0))
    positions = np.arange(len(runs), dtype=float)
    width = 0.36
    seed_labels = [f"seed {run['seed']}" for run in runs]
    bars = (
        (axes[0, 0], "macro_f1", "Held-out macro F1"),
        (axes[0, 1], "accuracy", "Held-out accuracy"),
    )
    for panel, score_key, title in bars:
        handwritten = [run["heldout"]["handwritten"][score_key] for run in runs]
        reference = [run["heldout"]["sklearn"][score_key] for run in runs]
        panel.bar(positions - width / 2, handwritten, width, label="handwritten NumPy")
        panel.bar(
            positions + width / 2,
            reference,
            width,
            label="sklearn KNeighborsClassifier",
        )
        panel.set_xticks(positions, seed_labels)
        panel.set_ylim(0.0, 1.05)
        panel.set_title(f"{title}\nBoth models use the selected configuration")
        panel.legend(fontsize=8, loc="lower right")
    first = runs[0]
    matrices = (
        (axes[1, 0], "handwritten", "handwritten NumPy"),
        (axes[1, 1], "sklearn", "sklearn KNeighborsClassifier"),
    )
    for panel, model_key, label in matrices:
        summary = first["heldout"][model_key]
        matrix = np.asarray(summary["confusion_matrix"])
        artist = panel.imshow(matrix, cmap="Blues")
        ticks = summary["confusion_matrix_labels"]
        panel.set_title(f"{label}\nSeed {first['seed']} confusion matrix")
        panel.set_xlabel("predicted")
        panel.set_ylabel("true")
        panel.set_xticks(range(len(ticks)), ticks)
        panel.set_yticks(range(len(ticks)), ticks)
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                panel.text(
                    column,
                    row,
                    str(int(matrix[row, column])),
                    ha="center",
                    va="center",
                    fontsize=8,
                )
        fig.colorbar(artist, ax=panel, fraction=0.046, pad=0.04)
    fig.tight_layout()
    save_figure(fig, output_path)


def write_figures(runs: list[dict[str, Any]], output_dir: Path) -> dict[str, Path]:
    """Write both PNG figures and return their paths keyed by role."""
    output_dir.mkdir(parents=True, exist_ok=True)
    sweep_path = output_dir / SWEEP_FIGURE
    baseline_path = output_dir / BASELINE_FIGURE
    plot_dev_curves(runs, sweep_path)
    plot_baseline_comparison(runs, baseline_path)
    return {"dev_sweep": sweep_path, "baseline": baseline_path}


# %% Entry point
def main(argv: list[str] | None = None) -> int:
    """Run the seeded digits protocol and write JSON evidence plus PNG figures."""
    parser = argparse.ArgumentParser(
        description="Handwritten k-nearest neighbours on the scikit-learn digits dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="directory for knn.json and the PNG figures (default: results/)",
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
    figure_paths = write_figures(payload["runs"], output_dir)
    payload["figures"] = {role: path.name for role, path in figure_paths.items()}
    write_results(output_dir, RESULT_NAME, payload)

    for run in payload["runs"]:
        selection = run["selection"]
        print(
            f"seed {run['seed']}: {_selection_label(selection)} "
            f"dev_macro_f1={selection['dev_macro_f1']:.4f} "
            f"test_macro_f1={run['heldout']['handwritten']['macro_f1']:.4f}"
        )
    print(f"wrote {RESULT_NAME}.json and {len(figure_paths)} figures to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
