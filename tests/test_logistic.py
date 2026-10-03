"""Consumer-visible behavior of the public logistic-regression interface."""

from __future__ import annotations

import math
import warnings

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression

from mlshowcase.logistic import (
    THRESHOLD_GRID,
    FeatureScaler,
    add_intercept,
    cross_entropy,
    fit,
    logistic_gradient,
    logistic_loss,
    predict,
    predict_proba,
    run_seed,
    sigmoid,
)


def test_sigmoid_saturates_at_extreme_scores_without_overflow() -> None:
    """Scores around +/-1000 stay finite, monotone and warning-free."""
    scores = np.array([-1000.0, -40.0, -1.0, 0.0, 1.0, 40.0, 1000.0])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        probabilities = sigmoid(scores)

    assert np.isfinite(probabilities).all()
    assert (probabilities >= 0.0).all() and (probabilities <= 1.0).all()
    assert np.all(np.diff(probabilities) >= 0.0)
    assert probabilities[0] == 0.0
    assert probabilities[0] < probabilities[1] < probabilities[3] < probabilities[4]
    assert probabilities[3] == 0.5
    assert probabilities[-1] == 1.0
    np.testing.assert_allclose(sigmoid(np.array([-1.0])), [1.0 / (1.0 + math.e)], rtol=1e-15)
    np.testing.assert_allclose(sigmoid(np.array([2.0])), [1.0 / (1.0 + math.exp(-2.0))], rtol=1e-15)


def test_cross_entropy_stays_finite_at_extreme_logits() -> None:
    """The logaddexp cross-entropy cannot overflow for large-magnitude logits."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        extreme = cross_entropy(np.array([0.0, 1.0]), np.array([1000.0, -1000.0]))

    assert math.isfinite(extreme)
    np.testing.assert_allclose(extreme, 1000.0, rtol=1e-12)
    np.testing.assert_allclose(
        cross_entropy(np.array([1.0, 0.0]), np.array([0.0, 0.0])),
        math.log(2.0),
        rtol=1e-15,
    )
    manual = cross_entropy(np.array([0.0, 1.0]), np.array([-0.5, 1.5]))
    expected = (math.log1p(math.exp(-0.5)) + math.log1p(math.exp(-1.5))) / 2.0
    np.testing.assert_allclose(manual, expected, rtol=1e-15)


def test_analytic_gradient_matches_central_finite_differences() -> None:
    """Finite differences of the loss agree with the exposed analytic gradient."""
    rng = np.random.default_rng(7)
    design = add_intercept(rng.normal(size=(9, 3)))
    labels = (rng.random(9) < 0.5).astype(float)
    theta = rng.normal(size=design.shape[1])
    epsilon = 1e-6

    analytic = logistic_gradient(design, labels, theta)
    numeric = np.empty_like(theta)
    for index in range(theta.size):
        step = np.zeros_like(theta)
        step[index] = epsilon
        numeric[index] = (
            logistic_loss(design, labels, theta + step)
            - logistic_loss(design, labels, theta - step)
        ) / (2.0 * epsilon)

    np.testing.assert_allclose(analytic, numeric, rtol=1e-6, atol=1e-8)


def test_fit_learns_a_linear_boundary_between_two_clouds() -> None:
    """A stable rate converges and the linear score separates the two clouds."""
    rng = np.random.default_rng(11)
    negative = rng.normal(loc=-1.5, scale=1.0, size=(60, 2))
    positive = rng.normal(loc=1.5, scale=1.0, size=(60, 2))
    features = np.vstack((negative, positive))
    labels = np.concatenate((np.zeros(60), np.ones(60)))

    result = fit(features, labels, learning_rate=0.5, max_steps=3000, tolerance=1e-7)

    assert not result.diverged
    assert result.steps == len(result.loss_history) - 1
    assert result.steps == len(result.gradient_norm_history) - 1
    history = np.asarray(result.loss_history)
    assert history[-1] < history[0]
    assert np.all(np.diff(history) <= 1e-12)

    # The last history entry always describes the returned parameters.
    design = add_intercept(features)
    np.testing.assert_allclose(
        result.loss_history[-1],
        logistic_loss(design, labels, result.theta),
        rtol=1e-15,
        atol=0.0,
    )
    np.testing.assert_allclose(
        result.gradient_norm_history[-1],
        float(np.max(np.abs(logistic_gradient(design, labels, result.theta)))),
        rtol=1e-15,
        atol=0.0,
    )

    predictions = predict(features, result.theta)
    assert (predictions == labels).mean() >= 0.85

    start = np.array([-1.5, -1.5])
    end = np.array([1.5, 1.5])
    segment = start + np.linspace(0.0, 1.0, 101)[:, None] * (end - start)
    probabilities = predict_proba(segment, result.theta)
    assert probabilities[0] < 0.5 < probabilities[-1]
    segment_predictions = predict(segment, result.theta)
    assert segment_predictions[0] == 0
    assert segment_predictions[-1] == 1
    assert int(np.count_nonzero(np.diff(segment_predictions))) == 1


def test_fit_records_the_state_produced_by_the_capped_final_update() -> None:
    """``max_steps=1`` returns the updated theta, not a stale pre-update state.

    On ``x = [-1, 2]`` with balanced labels ``[0, 1]`` the gradient at the
    zero initialization is ``[mean(0.5 - y), mean((0.5 - y) * x)] =
    [0, -0.75]``, so one full-batch step at rate 0.2 lands on
    ``theta = [0, 0.15]``; the step cap stops the run before convergence. The
    expected final loss and gradient norm below are derived from the sigmoid
    and cross-entropy definitions evaluated at that returned state.
    """
    features = np.array([[-1.0], [2.0]])
    labels = np.array([0.0, 1.0])
    learning_rate = 0.2

    result = fit(features, labels, learning_rate=learning_rate, max_steps=1)

    assert not result.converged
    assert not result.diverged
    assert result.steps == 1
    assert len(result.loss_history) == len(result.gradient_norm_history) == 2

    # One step from theta = 0, derived here from the dataset and the step rule.
    initial_gradient = np.array(
        [float(np.mean(0.5 - labels)), float(np.mean((0.5 - labels) * features[:, 0]))]
    )
    expected_theta = -learning_rate * initial_gradient
    np.testing.assert_allclose(result.theta, expected_theta, rtol=0.0, atol=1e-15)

    # Loss and gradient of the returned state, from the definitions.
    z_negative = float(expected_theta[0] - expected_theta[1])
    z_positive = float(expected_theta[0] + 2.0 * expected_theta[1])
    negative_probability = 1.0 / (1.0 + math.exp(-z_negative))
    positive_probability = 1.0 / (1.0 + math.exp(-z_positive))
    expected_loss = (math.log1p(math.exp(z_negative)) + math.log1p(math.exp(-z_positive))) / 2.0
    expected_gradient = np.array(
        [
            (negative_probability + positive_probability - 1.0) / 2.0,
            (-negative_probability + 2.0 * (positive_probability - 1.0)) / 2.0,
        ]
    )

    np.testing.assert_allclose(result.loss_history[0], math.log(2.0), rtol=1e-15, atol=0.0)
    np.testing.assert_allclose(result.loss_history[-1], expected_loss, rtol=1e-12, atol=0.0)
    assert result.loss_history[-1] < result.loss_history[0]
    np.testing.assert_allclose(
        result.gradient_norm_history[-1],
        float(np.max(np.abs(expected_gradient))),
        rtol=1e-12,
        atol=0.0,
    )

    # The recorded state is also the state of the returned parameters for the
    # public loss/gradient interfaces.
    design = add_intercept(features)
    np.testing.assert_allclose(
        result.loss_history[-1],
        logistic_loss(design, labels, result.theta),
        rtol=1e-15,
        atol=0.0,
    )
    np.testing.assert_allclose(
        result.gradient_norm_history[-1],
        float(np.max(np.abs(logistic_gradient(design, labels, result.theta)))),
        rtol=1e-15,
        atol=0.0,
    )


def test_predict_threshold_controls_hard_labels_at_fixed_theta() -> None:
    """Thresholds 0.1 and 0.9 move the boundary without refitting theta."""
    features = np.linspace(-3.0, 3.0, 13).reshape(-1, 1)
    theta = np.array([0.0, 2.0])

    probabilities = predict_proba(features, theta)
    low = predict(features, theta, threshold=0.1)
    high = predict(features, theta, threshold=0.9)

    assert np.all(high <= low)
    assert int(high.sum()) < int(low.sum())
    midpoint = 6
    assert features[midpoint, 0] == 0.0
    assert probabilities[midpoint] == 0.5
    assert low[midpoint] == 1
    assert high[midpoint] == 0


def test_fitted_threshold_changes_positive_counts_monotonically() -> None:
    """A fitted model predicts fewer positives as the threshold rises."""
    rng = np.random.default_rng(5)
    features = rng.normal(size=(600, 2))
    scores = 1.4 * features[:, 0] - 1.1 * features[:, 1]
    labels = (rng.random(600) < 1.0 / (1.0 + np.exp(-scores))).astype(float)

    result = fit(features, labels, learning_rate=0.5, max_steps=3000, tolerance=1e-7)
    counts = [
        int(predict(features, result.theta, threshold=cutoff).sum())
        for cutoff in (0.1, 0.3, 0.5, 0.7, 0.9)
    ]

    assert counts == sorted(counts, reverse=True)
    assert counts[0] > counts[-1]


def test_scaler_transforms_later_data_with_stored_training_statistics() -> None:
    """Held-out data is shifted with the stored statistics, never refitted."""
    rng = np.random.default_rng(3)
    training = rng.normal(loc=5.0, scale=2.0, size=(40, 3))
    scaler = FeatureScaler.fit(training)
    stored_mean = scaler.mean.copy()
    stored_scale = scaler.scale.copy()

    held_out = rng.normal(loc=-1.0, scale=0.5, size=(12, 3))
    shift = 7.5
    shifted = scaler.transform(held_out + shift) - scaler.transform(held_out)
    training_scale = training.std(axis=0)
    expected_shift = np.broadcast_to(shift / training_scale, held_out.shape)
    held_out_scale = held_out.std(axis=0)
    assert not np.allclose(expected_shift, np.broadcast_to(shift / held_out_scale, held_out.shape))
    np.testing.assert_allclose(shifted, expected_shift, rtol=1e-12, atol=1e-12)

    transformed = scaler.transform(held_out)
    assert np.abs(transformed.mean(axis=0)).max() > 1.0
    np.testing.assert_allclose(scaler.transform(training).mean(axis=0), np.zeros(3), atol=1e-12)
    np.testing.assert_allclose(scaler.transform(training).std(axis=0), np.ones(3), atol=1e-12)
    np.testing.assert_array_equal(scaler.mean, stored_mean)
    np.testing.assert_array_equal(scaler.scale, stored_scale)


def test_scaler_keeps_constant_columns_finite_and_zero() -> None:
    """A zero-variance column keeps scale 1.0 and transforms to exact zeros."""
    training = np.array([[1.0, 5.0], [1.0, 7.0], [1.0, 9.0]])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        scaler = FeatureScaler.fit(training)
        transformed = scaler.transform(training)

    assert np.isfinite(transformed).all()
    assert scaler.scale[0] == 1.0
    np.testing.assert_array_equal(transformed[:, 0], np.zeros(3))
    np.testing.assert_allclose(transformed[:, 1].mean(), 0.0, atol=1e-12)
    np.testing.assert_allclose(transformed[:, 1].std(), 1.0, atol=1e-12)

    later = scaler.transform(np.array([[1.0, 11.0], [1.0, -3.0]]))
    assert np.isfinite(later).all()
    np.testing.assert_array_equal(later[:, 0], np.zeros(2))


def test_handwritten_fit_agrees_with_unregularized_sklearn_solution() -> None:
    """Both implementations minimize the same objective on identical data."""
    rng = np.random.default_rng(17)
    features = rng.normal(size=(300, 2))
    scores = 1.3 * features[:, 0] - 0.7 * features[:, 1] + 0.4
    labels = (rng.random(300) < 1.0 / (1.0 + np.exp(-scores))).astype(float)

    result = fit(features, labels, learning_rate=0.5, max_steps=10000, tolerance=1e-8)
    assert result.converged

    baseline = LogisticRegression(C=np.inf, solver="lbfgs", max_iter=10000, tol=1e-12)
    baseline.fit(features, labels)
    expected = np.concatenate((baseline.intercept_, baseline.coef_.ravel()))

    np.testing.assert_allclose(result.theta, expected, atol=1e-3)


def test_run_seed_selects_on_full_precision_development_f1() -> None:
    """The joint selection maximizes the development F1 and reports the selected score.

    The chosen (learning rate, threshold) pair is the maximum of the
    development F1 grid over the eligible runs under the documented tie-break
    (smallest threshold, then smallest learning rate), and ``selection.dev_f1``
    is the selected run's own development F1.
    """
    dataset = load_breast_cancer()
    features = np.asarray(dataset.data, dtype=float)
    labels = (np.asarray(dataset.target, dtype=int) == 0).astype(int)

    run = run_seed(features, labels, seed=42)

    eligible = [entry for entry in run["rates"] if entry["eligible_for_selection"]]
    best_f1 = max(
        entry["dev_f1_by_threshold"][index]
        for entry in eligible
        for index in range(len(THRESHOLD_GRID))
    )
    selection = run["selection"]
    assert selection["dev_f1"] == best_f1
    assert selection["dev_f1"] == selection["dev_metrics"]["positive_f1"]

    achieving = [
        (THRESHOLD_GRID[index], entry["learning_rate"])
        for entry in eligible
        for index in range(len(THRESHOLD_GRID))
        if entry["dev_f1_by_threshold"][index] == best_f1
    ]
    expected_threshold, expected_rate = min(achieving)
    assert selection["threshold"] == expected_threshold
    assert selection["learning_rate"] == expected_rate
