"""Behavioral tests for the shared showcase infrastructure."""

from __future__ import annotations

import getpass
import json
import platform
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from mlshowcase.common import environment, stratified_split, write_results

CLASS_COUNTS = {"alpha": 50, "beta": 30, "gamma": 20}


def _labels() -> np.ndarray:
    """A deterministic, imbalanced three-class label vector of 100 entries."""
    return np.concatenate([np.full(count, label) for label, count in CLASS_COUNTS.items()])


def test_partitions_are_disjoint_and_cover_every_source_index():
    labels = _labels()
    split = stratified_split(labels, seed=7)

    train, dev, test = np.asarray(split.train), np.asarray(split.dev), np.asarray(split.test)

    for part in (train, dev, test):
        assert part.ndim == 1
        assert np.issubdtype(part.dtype, np.integer)

    assert len(np.intersect1d(train, dev)) == 0
    assert len(np.intersect1d(train, test)) == 0
    assert len(np.intersect1d(dev, test)) == 0
    assert sorted(np.concatenate([train, dev, test]).tolist()) == list(range(labels.size))


def test_partition_sizes_follow_absolute_fractions_of_the_full_label_array():
    # 100 observations: treating dev as a fraction of the remainder would give
    # about 8 dev indices instead of the specified 20.
    labels = _labels()
    split = stratified_split(labels, seed=11)

    assert abs(split.train.size - 0.6 * labels.size) <= 1
    assert abs(split.dev.size - 0.2 * labels.size) <= 1
    assert split.test.size == labels.size - split.train.size - split.dev.size


def test_every_partition_preserves_class_representation():
    labels = _labels()
    split = stratified_split(labels, seed=23)

    for part, fraction in ((split.train, 0.6), (split.dev, 0.2), (split.test, 0.2)):
        counts = Counter(labels[part].tolist())
        assert set(counts) == set(CLASS_COUNTS)
        for label, total in CLASS_COUNTS.items():
            assert abs(counts[label] - fraction * total) <= 1


def test_fractions_summing_to_one_leave_an_empty_test_partition():
    labels = _labels()

    split = stratified_split(labels, seed=42, train_fraction=0.9, dev_fraction=0.1)

    assert split.train.size == 90
    assert split.dev.size == 10
    assert split.test.size == 0
    assert np.issubdtype(split.test.dtype, np.integer)
    assert sorted(np.concatenate([split.train, split.dev, split.test]).tolist()) == list(
        range(labels.size)
    )

    counts = Counter(labels[split.dev].tolist())
    for label, total in CLASS_COUNTS.items():
        assert abs(counts[label] - 0.1 * total) <= 1


def test_boundary_split_matches_a_large_held_out_protocol():
    # 50,000 observations over 100 classes: the CIFAR-100 train-set protocol
    # carves 45,000 train / 5,000 dev while an external test set is held out.
    labels = np.repeat(np.arange(100), 500)

    split = stratified_split(labels, seed=42, train_fraction=0.9, dev_fraction=0.1)

    assert split.train.size == 45_000
    assert split.dev.size == 5_000
    assert split.test.size == 0


def test_same_seed_reproduces_the_split_and_different_seeds_change_it():
    labels = _labels()
    first = stratified_split(labels, seed=1234)
    repeat = stratified_split(labels, seed=1234)

    assert np.array_equal(first.train, repeat.train)
    assert np.array_equal(first.dev, repeat.dev)
    assert np.array_equal(first.test, repeat.test)

    other = stratified_split(labels, seed=4321)
    assert not np.array_equal(first.train, other.train)


@pytest.mark.parametrize(
    ("train_fraction", "dev_fraction"),
    [
        (-0.1, 0.2),
        (0.6, -0.1),
        (0.0, 0.2),
        (0.6, 0.0),
        (1.0, 0.1),
        (0.6, 1.0),
        (0.9, 0.2),
        (0.8, 0.4),
        (float("nan"), 0.2),
        (0.6, float("inf")),
        (0.6, float("-inf")),
    ],
)
def test_invalid_fractions_are_rejected(train_fraction, dev_fraction):
    with pytest.raises(ValueError):
        stratified_split(
            _labels(), seed=0, train_fraction=train_fraction, dev_fraction=dev_fraction
        )


@pytest.mark.parametrize("bad_fraction", ["0.6", None])
def test_non_numeric_fractions_are_rejected(bad_fraction):
    with pytest.raises(TypeError):
        stratified_split(_labels(), seed=0, train_fraction=bad_fraction)


def test_write_results_round_trips_concrete_metrics(tmp_path):
    payload = {
        "protocol": {"seed": 7, "train_fraction": 0.6, "dev_fraction": 0.2},
        "metrics": {"accuracy": 0.9375, "macro_f1": 0.925, "log_loss": 0.1875},
        "counts": [60, 20, 20],
    }

    path = write_results(tmp_path / "nested" / "results", "knn", payload)

    assert path == tmp_path / "nested" / "results" / "knn.json"
    assert json.loads(path.read_text(encoding="utf-8")) == payload


@pytest.mark.parametrize(
    "payload",
    [
        {"accuracy": float("nan")},
        {"per_class": [0.5, float("inf")]},
        {"history": {"loss": [0.5, float("-inf")]}},
    ],
)
def test_write_results_rejects_nonfinite_values(tmp_path, payload):
    with pytest.raises(ValueError):
        write_results(tmp_path, "knn", payload)

    assert not (tmp_path / "knn.json").exists()


def test_environment_exposes_public_metadata_and_installed_packages():
    info = environment()

    assert info["python"] == {
        "version": platform.python_version(),
        "implementation": platform.python_implementation(),
    }
    assert info["os"]["system"] == platform.system()
    assert set(info["os"]) == {"system", "release", "machine"}
    assert info["cpu_count"] >= 1

    packages = info["packages"]
    assert {"numpy", "scipy", "scikit-learn", "matplotlib"} <= set(packages)
    assert all(isinstance(version, str) and version for version in packages.values())


def test_environment_metadata_is_json_serializable_without_personal_details():
    info = environment()

    serialized = json.dumps(info, allow_nan=False, sort_keys=True)

    # A CPU model may legitimately contain a slash (for example
    # "w/ Radeon graphics"), so check for this machine's actual private
    # locations rather than the "/" character.
    for private_location in (Path.home(), Path.cwd()):
        assert str(private_location) not in serialized
    hostname = platform.node()
    if hostname:
        assert hostname not in serialized


def test_environment_cpu_model_is_recorded_without_private_identifiers():
    model = environment()["cpu_model"]

    assert isinstance(model, str)
    for private_detail in (platform.node(), getpass.getuser(), str(Path.home())):
        if private_detail:
            assert private_detail not in model
