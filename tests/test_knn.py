"""Consumer-visible behaviour of the handwritten k-NN implementation.

The fixtures are hand-checked geometries plus tie-free random data, so the
expected values come from the maths (or from scikit-learn as an independent
implementation) rather than from the module under test.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.neighbors import KNeighborsClassifier

from mlshowcase.knn import (
    KNNClassifier,
    euclidean_distance,
    majority_vote,
    manhattan_distance,
    minkowski_distance,
    rank_neighbors,
)


# %% Distances
def test_euclidean_distance_matches_worked_example() -> None:
    # sqrt(3^2 + 4^2 + 12^2) = sqrt(169) = 13; dropping the final feature gives 5.
    assert euclidean_distance([3.0, 4.0, 12.0], [0.0, 0.0, 0.0]) == pytest.approx(13.0)


def test_manhattan_distance_matches_worked_example() -> None:
    # 3 + 4 + 12 = 19; dropping the final feature gives 7.
    assert manhattan_distance([3.0, 4.0, 12.0], [0.0, 0.0, 0.0]) == pytest.approx(19.0)


def test_minkowski_distance_matches_worked_example() -> None:
    distance = minkowski_distance([1.0, 2.0, 2.0], [0.0, 0.0, 0.0], p=3.0)
    # (1 + 8 + 8) ** (1/3); cubing keeps the check independent of the module.
    assert distance**3 == pytest.approx(17.0, rel=1e-12)


def test_distances_use_the_final_feature() -> None:
    # The two vectors differ only in their last component; the historical
    # loop over ``len(row) - 1`` would report 0.0 instead of 7.0.
    a = [0.0, 0.0, 0.0, 0.0]
    b = [0.0, 0.0, 0.0, 7.0]
    assert euclidean_distance(a, b) == pytest.approx(7.0)
    assert manhattan_distance(a, b) == pytest.approx(7.0)
    assert minkowski_distance(a, b, p=1.5) == pytest.approx(7.0)


@pytest.mark.parametrize("p", [0.0, 0.5, 0.99])
def test_minkowski_rejects_exponents_below_one(p: float) -> None:
    with pytest.raises(ValueError, match="must be >= 1"):
        minkowski_distance([1.0, 2.0], [0.0, 0.0], p=p)


def test_minkowski_requires_an_explicit_exponent() -> None:
    with pytest.raises(ValueError, match="explicit exponent"):
        minkowski_distance([1.0, 2.0], [0.0, 0.0])
    with pytest.raises(ValueError, match="explicit exponent"):
        rank_neighbors([[0.0]], [[0.0]], metric="minkowski")


def test_rank_neighbors_rejects_unknown_metric() -> None:
    with pytest.raises(ValueError, match="unknown metric"):
        rank_neighbors([[0.0]], [[0.0]], metric="cosine")


# %% Neighbour ranking and voting
def test_one_ranking_supports_a_k_sweep() -> None:
    # Query at the origin: nearest neighbour labels 7, the next two label 3.
    # A single ranking votes 7 at k=1 and 3 at k=3 without recomputing distances.
    train_x = np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    train_y = np.array([7, 3, 3])
    ranking = rank_neighbors(train_x, np.array([[0.0, 0.0]]), metric="euclidean")

    assert ranking.indices.shape == (1, 3)
    assert np.array_equal(ranking.distances, [[1.0, 2.0, 3.0]])
    assert majority_vote(ranking, train_y, 1)[0] == 7
    assert majority_vote(ranking, train_y, 3)[0] == 3


def test_majority_vote_rejects_k_larger_than_the_training_set() -> None:
    ranking = rank_neighbors([[0.0], [1.0]], [[0.5]], metric="euclidean")
    with pytest.raises(ValueError, match="k must be between"):
        majority_vote(ranking, np.array([0, 1]), 3)


# %% Classifier
def test_classifier_uses_the_final_feature_when_it_is_decisive() -> None:
    # Training points differ only in the last feature, so the nearest
    # neighbour is decided entirely by that final component.
    train_x = np.array([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 10.0]])
    train_y = np.array([0, 1])
    model = KNNClassifier(n_neighbors=1).fit(train_x, train_y)

    assert model.predict(np.array([[0.0, 0.0, 0.0, 1.0]]))[0] == 0
    assert model.predict(np.array([[0.0, 0.0, 0.0, 9.0]]))[0] == 1


def test_noncontiguous_labels_survive_the_majority_vote() -> None:
    # Two of three neighbours vote 7; a returned class index would be 0.
    train_x = np.array([[1.0, 0.0], [2.0, 0.0], [20.0, 0.0]])
    train_y = np.array([7, 40, 7])
    model = KNNClassifier(n_neighbors=3).fit(train_x, train_y)

    predictions = model.predict(np.array([[0.0, 0.0]]))
    assert predictions[0] == 7
    assert predictions.dtype == train_y.dtype  # labels stay integers, never floats


def test_vote_ties_resolve_to_the_smallest_class_label() -> None:
    # The query sits exactly between a neighbour labelled 9 and one labelled 2,
    # so the 2-neighbour vote is a 1-1 tie.
    train_x = np.array([[1.0, 0.0], [-1.0, 0.0]])
    train_y = np.array([9, 2])
    query = np.array([[0.0, 0.0]])

    assert KNNClassifier(n_neighbors=2).fit(train_x, train_y).predict(query)[0] == 2
    flipped = KNNClassifier(n_neighbors=2).fit(train_x[::-1], train_y[::-1])
    assert flipped.predict(query)[0] == 2


def test_equal_distances_keep_training_row_order() -> None:
    # Two identical training points: stable ordering makes the first row win a
    # 1-neighbour vote, and reversing the rows flips the deterministic winner.
    train_x = np.array([[1.0, 0.0], [1.0, 0.0]])
    query = np.array([[0.0, 0.0]])

    first = KNNClassifier(n_neighbors=1).fit(train_x, np.array([4, 6])).predict(query)[0]
    flipped = KNNClassifier(n_neighbors=1).fit(train_x[::-1], np.array([6, 4])).predict(query)[0]
    assert first == 4
    assert flipped == 6


# %% Reference comparison
@pytest.mark.parametrize(
    ("metric", "p"),
    [
        pytest.param("euclidean", None, id="euclidean"),
        pytest.param("manhattan", None, id="manhattan"),
        pytest.param("minkowski", 1.5, id="minkowski-p1.5"),
    ],
)
def test_predictions_match_sklearn_on_a_tie_free_fixture(metric: str, p: float | None) -> None:
    rng = np.random.default_rng(20251002)
    train_x = rng.normal(size=(30, 5))
    train_y = rng.choice([7, 40, 99], size=30)  # noncontiguous labels as well
    query_x = rng.normal(size=(12, 5))
    neighbors = 5

    reference = KNeighborsClassifier(
        n_neighbors=neighbors,
        metric=metric,
        p=p if p is not None else 2,
        algorithm="brute",
    ).fit(train_x, train_y)
    reference_distances, _ = reference.kneighbors(query_x, n_neighbors=neighbors + 1)
    # The fixture is tie-free at the k-th boundary, so the voting multiset is
    # unique and both implementations must return the same labels.
    assert np.all(reference_distances[:, neighbors] > reference_distances[:, neighbors - 1])

    ours = KNNClassifier(n_neighbors=neighbors, metric=metric, p=p).fit(train_x, train_y)
    assert np.array_equal(ours.predict(query_x), reference.predict(query_x))
