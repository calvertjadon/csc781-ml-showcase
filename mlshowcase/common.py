"""Shared helpers for the CSC 781 experiments.

Experiment modules import their split, environment metadata, and results writer
from this module so that protocols stay comparable across experiments. The
module needs numpy and scikit-learn. It reports optional distributions, such as
the CPU-only PyTorch extras, when they are installed and never assumes they are
present.
"""

from __future__ import annotations

import json
import math
import numbers
import os
import platform
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from sklearn.model_selection import train_test_split

__all__ = ["Split", "environment", "stratified_split", "write_results"]

_METADATA_PACKAGES = (
    "numpy",
    "scipy",
    "scikit-learn",
    "matplotlib",
    "torch",
    "torchvision",
)


@dataclass(frozen=True)
class Split:
    """Disjoint train/development/test partitions as integer index arrays.

    Each attribute is a one-dimensional ``numpy.intp`` array of positions into
    the source label array. The three partitions are pairwise disjoint and
    together cover every source position.
    """

    train: np.ndarray
    dev: np.ndarray
    test: np.ndarray


def _validate_fraction(value: float, name: str) -> float:
    """Return ``value`` as a finite float strictly between 0 and 1."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a real number, got {value!r}")
    fraction = float(value)
    if not math.isfinite(fraction):
        raise ValueError(f"{name} must be finite, got {value!r}")
    if not 0.0 < fraction < 1.0:
        raise ValueError(f"{name} must be strictly between 0 and 1, got {value!r}")
    return fraction


def stratified_split(
    labels: ArrayLike,
    seed: int,
    train_fraction: float = 0.6,
    dev_fraction: float = 0.2,
) -> Split:
    """Partition label indices into stratified train/development/test sets.

    ``train_fraction`` and ``dev_fraction`` are absolute fractions of the full
    label array and must not sum to more than 1; the test partition receives
    the remainder, which is empty when the two fractions sum to exactly 1. Both
    stages are stratified by label and deterministic for a given ``seed``.
    """
    fraction_train = _validate_fraction(train_fraction, "train_fraction")
    fraction_dev = _validate_fraction(dev_fraction, "dev_fraction")
    fraction_test = 1.0 - math.fsum((fraction_train, fraction_dev))
    if fraction_test < 0.0:
        raise ValueError(
            "train_fraction + dev_fraction must not exceed 1; "
            f"got {train_fraction!r} and {dev_fraction!r}"
        )

    labels_array = np.asarray(labels)
    if labels_array.ndim != 1:
        raise ValueError(f"labels must be one-dimensional, got shape {labels_array.shape}")
    if labels_array.size == 0:
        raise ValueError("labels must contain at least one observation")

    indices = np.arange(labels_array.size, dtype=np.intp)
    train_indices, remainder_indices = train_test_split(
        indices,
        train_size=fraction_train,
        random_state=seed,
        shuffle=True,
        stratify=labels_array,
    )

    if fraction_test == 0.0:
        # The test remainder is empty; the whole remainder is development.
        return Split(
            train=np.asarray(train_indices, dtype=np.intp),
            dev=np.asarray(remainder_indices, dtype=np.intp),
            test=np.empty(0, dtype=np.intp),
        )

    # The development share is relative to the remainder so that the absolute
    # dev fraction is a fraction of the full label array, not of the remainder.
    dev_share = fraction_dev / (fraction_dev + fraction_test)
    dev_offsets, test_offsets = train_test_split(
        np.arange(remainder_indices.size, dtype=np.intp),
        train_size=dev_share,
        random_state=seed + 1,
        shuffle=True,
        stratify=labels_array[remainder_indices],
    )
    return Split(
        train=np.asarray(train_indices, dtype=np.intp),
        dev=np.asarray(remainder_indices[dev_offsets], dtype=np.intp),
        test=np.asarray(remainder_indices[test_offsets], dtype=np.intp),
    )


def _cpu_model() -> str:
    """Return the CPU model name, or ``""`` when it is unavailable.

    ``platform.processor()`` is the portable source and is commonly empty on
    Linux, so the function then reads the ``model name`` field of
    ``/proc/cpuinfo``. It extracts only that one field, so serial numbers and
    other machine identifiers in ``/proc/cpuinfo`` never reach the result.
    """
    model = platform.processor().strip()
    if model or platform.system() != "Linux":
        return model
    try:
        contents = Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    for line in contents.splitlines():
        field, separator, value = line.partition(":")
        if separator and field.strip() == "model name":
            return value.strip()
    return ""


def environment() -> dict[str, Any]:
    """Return interpreter, OS, CPU, and package-version metadata without personal identifiers.

    The result holds aggregate platform data only: no absolute paths, hostnames,
    usernames, or machine identifiers such as serial numbers. The CPU appears as
    its logical ``cpu_count`` and a ``cpu_model`` name, which is the processor
    name, or the Linux ``/proc/cpuinfo`` ``model name`` when the interpreter
    reports no processor. Optional distributions, for example the CPU-only
    PyTorch extras, appear only when installed.
    """
    packages: dict[str, str] = {}
    for distribution in _METADATA_PACKAGES:
        try:
            packages[distribution] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            continue

    return {
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
        },
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "cpu_count": os.cpu_count(),
        "cpu_model": _cpu_model(),
        "packages": packages,
    }


def write_results(output_dir: Path, name: str, payload: dict) -> Path:
    """Write ``payload`` as strict, finite JSON to ``output_dir/<name>.json``.

    The writer creates the output directory and its parents when missing. It
    raises :class:`ValueError` for non-finite values such as NaN or Infinity
    instead of writing non-standard JSON tokens, and it never leaves a partial
    file behind.
    """
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.json"
    path.write_text(
        json.dumps(payload, indent=2, allow_nan=False, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path
