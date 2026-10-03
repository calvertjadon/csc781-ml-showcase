"""Matched DenseNet-161 transfer-learning comparison on CIFAR-100.

The experiment rebuilds the CSC781 Module 8 Assignment 9 exercise as a matched,
selection-safe protocol:

* ``scratch`` - DenseNet-161 trained from random initialisation (``weights=None``),
* ``pretrained`` - DenseNet-161 initialised from the ImageNet-1K weights, with the
  backbone parameters *and* the BatchNorm running buffers frozen and a fresh
  100-class head.

Both variants see the same stratified 45000/5000 carve of the official CIFAR-100
training split, the same 64 px preprocessing derived from the DenseNet-161
weights recipe (normalisation and bilinear interpolation inherited, standard
224/256 resolution overridden), and the same fixed-budget SGD settings. The
official 10000-image test split is scored once, after the best development
macro-F1 checkpoint has been selected.

Run from the repository root::

    uv run --locked --extra deep-learning python -m mlshowcase.transfer --device cpu

The first run downloads CIFAR-100 through torchvision and the DenseNet-161
IMAGENET1K_V1 weights. The run writes ``transfer.json`` and two PNG figures into
the output directory (default ``results/``). Selected checkpoints are written to
``<output-dir>/checkpoints/`` for local inspection only: they are git-ignored and
are not referenced by the JSON evidence.

The CLI prints flushed progress lines while it works: run and epoch starts,
periodic batch and sample counts, the development and final-test phases, the
selected checkpoint and the elapsed times. The reusable helpers accept an
optional progress callback and otherwise stay quiet.
"""

from __future__ import annotations

import argparse
import random
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

from mlshowcase.common import environment, stratified_split, write_results

# Figures are always written to PNG files, so pin a headless backend that
# behaves identically in a terminal and in headless renders.
plt.switch_backend("Agg")

RESULT_NAME = "transfer"
LEARNING_CURVE_FIGURE = "transfer_learning_curves.png"
COMPARISON_FIGURE = "transfer_comparison.png"
CHECKPOINT_DIRECTORY = "checkpoints"

NUM_CLASSES = 100
VARIANTS: tuple[str, ...] = ("scratch", "pretrained")
BACKBONE_MODULE = "features"

DEFAULT_DATA_DIR = Path("data")
DEFAULT_OUTPUT_DIR = Path("results")
DEFAULT_EPOCHS = 1
DEFAULT_SEED = 42
DEFAULT_BATCH_SIZE = 32
DEFAULT_THREADS = 4
LEARNING_RATE = 0.01
MOMENTUM = 0.9
SPLIT_SEED = 42
TRAIN_FRACTION = 0.9
DEV_FRACTION = 0.1
CROP_SIZE = 64
RESIZE_SIZE = 74

_VARIANT_COLORS = {"scratch": "tab:orange", "pretrained": "tab:blue"}


# %% Environment and preprocessing
def resolve_device(specification: str) -> torch.device:
    """Resolve ``"auto"`` (CUDA when available, else CPU) or an explicit device."""
    name = str(specification).strip().lower()
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("device 'cuda' was requested but CUDA is not available")
    return device


def seed_everything(seed: int) -> torch.Generator:
    """Seed Python, NumPy and torch; return the DataLoader shuffle generator.

    Both variants receive a generator seeded with the same value, so they see
    the same shuffling order and the comparison stays matched.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return torch.Generator().manual_seed(seed)


def build_transform(crop_size: int = CROP_SIZE, resize_size: int = RESIZE_SIZE) -> Any:
    """Build the DenseNet-161 weights recipe at the experiment's input size.

    ``weights.transforms(...)`` inherits the checkpoint's normalisation constants,
    interpolation mode and antialiasing while overriding the standard 224/256
    resolution; both variants use this exact transform.
    """
    from torchvision.models import DenseNet161_Weights

    return DenseNet161_Weights.IMAGENET1K_V1.transforms(
        crop_size=crop_size, resize_size=resize_size
    )


def describe_transform(transform: Any) -> dict[str, Any]:
    """Return the public, JSON-ready description of a weights transform instance."""
    return {
        "source": "DenseNet161_Weights.IMAGENET1K_V1.transforms",
        "resize_size": int(transform.resize_size[0]),
        "crop_size": int(transform.crop_size[0]),
        "interpolation": str(
            getattr(transform.interpolation, "name", transform.interpolation)
        ).lower(),
        "antialias": bool(transform.antialias),
        "mean": [float(value) for value in transform.mean],
        "std": [float(value) for value in transform.std],
        "note": (
            "crop_size=64 / resize_size=74 override the weights' standard 224/256 "
            "resolution; normalisation constants and bilinear interpolation are "
            "inherited from the weights recipe, and both variants share this transform"
        ),
    }


# %% Model construction and freezing
def build_model(variant: str, num_classes: int = NUM_CLASSES) -> nn.Module:
    """Build one variant: ``scratch`` (random init) or ``pretrained`` (frozen backbone)."""
    from torchvision.models import DenseNet161_Weights, densenet161

    if variant == "scratch":
        return densenet161(weights=None, num_classes=num_classes)
    if variant == "pretrained":
        model = densenet161(weights=DenseNet161_Weights.IMAGENET1K_V1)
        return freeze_backbone_for_transfer(model, num_classes=num_classes)
    raise ValueError(f"unknown variant {variant!r}; expected one of {', '.join(VARIANTS)}")


def freeze_backbone_for_transfer(model: nn.Module, num_classes: int = NUM_CLASSES) -> nn.Module:
    """Freeze every existing parameter and install a fresh trainable classifier.

    Frozen ``requires_grad`` alone is not enough for BatchNorm: ``model.train()``
    would still update the running buffers. :func:`train_epoch` therefore holds
    the ``features`` module in ``eval()`` after every ``train()`` call.
    """
    for parameter in model.parameters():
        parameter.requires_grad = False
    head = model.classifier
    model.classifier = nn.Linear(head.in_features, num_classes)
    return model


def trainable_parameters(model: nn.Module) -> list[nn.Parameter]:
    """Return the parameters an optimiser should own for this model."""
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def _hold_frozen_modules_in_eval(model: nn.Module, frozen_prefixes: Sequence[str]) -> None:
    """Put the named backbone submodules back into evaluation mode."""
    for prefix in frozen_prefixes:
        model.get_submodule(prefix).eval()


# %% Progress reporting
# One periodic progress line every this many batches, plus one for the final batch.
PROGRESS_EVERY_BATCHES = 25

# A progress sink receives one finished line at a time. The reusable helpers
# never print by themselves: they stay quiet unless a sink is passed, which is
# how the tests exercise them and how the CLI opts into visible progress.
ProgressCallback = Callable[[str], None]


def _print_progress(message: str) -> None:
    """CLI sink: flush every line so a long run reports while it is running."""
    print(message, flush=True)


def _emit_progress(progress: ProgressCallback | None, message: str) -> None:
    """Send one status line to ``progress``; no sink means no output."""
    if progress is not None:
        progress(message)


def _scoped_progress(progress: ProgressCallback | None, scope: str) -> ProgressCallback | None:
    """Prefix every message with ``scope``, or stay ``None`` when there is no sink."""
    if progress is None:
        return None

    def emit(message: str) -> None:
        progress(f"{scope}: {message}")

    return emit


def _known_length(collection: Any) -> int | None:
    """Length of a loader or dataset, or ``None`` when it cannot report one."""
    try:
        return len(collection)
    except TypeError:
        return None


def _count_text(value: int | None) -> str:
    """Render a known length, or ``?`` when the collection cannot report one."""
    return "?" if value is None else str(value)


def _fraction(done: int, total: int | None) -> str:
    """``done/total`` when the total is known, just ``done`` otherwise."""
    return f"{done}/{total}" if total is not None else str(done)


def _format_seconds(seconds: float) -> str:
    """Compact elapsed time for progress lines: ``48s``, ``2m03s``, ``1h02m03s``."""
    whole_seconds = int(seconds)
    hours, remainder = divmod(whole_seconds, 3600)
    minutes, seconds_remaining = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{seconds_remaining:02d}s"
    if minutes:
        return f"{minutes}m{seconds_remaining:02d}s"
    return f"{seconds_remaining}s"


# %% Training and evaluation
def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device | str,
    *,
    frozen_prefixes: Sequence[str] = (),
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Train for one epoch and return the sample-count-weighted mean loss.

    Batch losses are weighted by each batch's actual sample count, so an
    undersized final batch cannot distort the epoch loss. ``frozen_prefixes``
    names submodules kept in ``eval()`` (frozen BatchNorm buffers) after
    ``model.train()`` puts the rest of the model into training mode.

    ``progress`` receives one line every ``PROGRESS_EVERY_BATCHES`` batches and
    one for the final batch, each with the batches and samples seen so far and
    the elapsed time. Leaving it ``None`` keeps the helper quiet for library use.
    """
    model.train()
    _hold_frozen_modules_in_eval(model, frozen_prefixes)

    total_loss = 0.0
    total_samples = 0
    batches = 0
    training_start = time.perf_counter()
    total_batches = _known_length(loader)
    dataset_samples = _known_length(getattr(loader, "dataset", None))
    for features, targets in loader:
        features = features.to(device)
        targets = targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(features), targets)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach()) * int(targets.size(0))
        total_samples += int(targets.size(0))
        batches += 1
        if progress is not None and (
            batches % PROGRESS_EVERY_BATCHES == 0 or batches == total_batches
        ):
            progress(
                f"training batch {_fraction(batches, total_batches)} "
                f"({_fraction(total_samples, dataset_samples)} samples, "
                f"{_format_seconds(time.perf_counter() - training_start)} elapsed)"
            )
    if total_samples == 0:
        raise ValueError("the training loader produced no samples")
    return {"loss": total_loss / total_samples, "samples": total_samples, "batches": batches}


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device | str,
    *,
    num_classes: int = NUM_CLASSES,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Evaluate a split: sample-weighted loss, accuracy and all-class macro F1.

    Macro F1 requests every class index from 0 to ``num_classes - 1``, so a class
    absent from the split still contributes a zero to the macro average.

    ``progress`` receives the same periodic batch/sample updates as
    :func:`train_epoch`; leaving it ``None`` keeps the helper quiet.
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0
    batches = 0
    evaluation_start = time.perf_counter()
    total_batches = _known_length(loader)
    dataset_samples = _known_length(getattr(loader, "dataset", None))
    predicted_batches: list[np.ndarray] = []
    label_batches: list[np.ndarray] = []
    with torch.no_grad():
        for features, targets in loader:
            features = features.to(device)
            targets = targets.to(device)
            logits = model(features)
            total_loss += float(criterion(logits, targets).detach()) * int(targets.size(0))
            total_samples += int(targets.size(0))
            predicted_batches.append(logits.argmax(dim=1).cpu().numpy())
            label_batches.append(targets.cpu().numpy())
            batches += 1
            if progress is not None and (
                batches % PROGRESS_EVERY_BATCHES == 0 or batches == total_batches
            ):
                progress(
                    f"evaluation batch {_fraction(batches, total_batches)} "
                    f"({_fraction(total_samples, dataset_samples)} samples, "
                    f"{_format_seconds(time.perf_counter() - evaluation_start)} elapsed)"
                )
    if total_samples == 0:
        raise ValueError("the evaluation loader produced no samples")
    predicted = np.concatenate(predicted_batches)
    truth = np.concatenate(label_batches)
    return {
        "loss": total_loss / total_samples,
        "accuracy": float(np.mean(predicted == truth)),
        "macro_f1": float(
            f1_score(
                truth,
                predicted,
                labels=list(range(num_classes)),
                average="macro",
                zero_division=0,
            )
        ),
        "samples": total_samples,
    }


# %% Data and splits
def load_cifar100(data_dir: Path, transform: Any) -> tuple[Dataset, Dataset]:
    """Load the official CIFAR-100 train and test splits through torchvision."""
    from torchvision.datasets import CIFAR100

    train = CIFAR100(root=str(data_dir), train=True, download=True, transform=transform)
    test = CIFAR100(root=str(data_dir), train=False, download=True, transform=transform)
    return train, test


def split_train_dev(labels: np.ndarray, seed: int = SPLIT_SEED) -> tuple[np.ndarray, np.ndarray]:
    """Stratified 90/10 carve of the official training labels (45000/5000).

    The shared splitter's test remainder is empty by design here: the official
    CIFAR-100 test images are a separate held-out dataset, not a remainder of
    the training split.
    """
    split = stratified_split(
        labels, seed=seed, train_fraction=TRAIN_FRACTION, dev_fraction=DEV_FRACTION
    )
    return np.asarray(split.train, dtype=np.int64), np.asarray(split.dev, dtype=np.int64)


# %% Experiment protocol
def save_checkpoint(
    model: nn.Module,
    checkpoint_dir: Path,
    variant: str,
    seed: int,
    epoch: int,
    dev_macro_f1: float,
) -> Path:
    """Save the selected state dict to the local, git-ignored checkpoint directory."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / f"{RESULT_NAME}_{variant}_seed{seed}.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "variant": variant,
            "seed": seed,
            "epoch": epoch,
            "dev_macro_f1": dev_macro_f1,
        },
        path,
    )
    return path


def run_variant(
    variant: str,
    train_dataset: Dataset,
    dev_dataset: Dataset,
    test_dataset: Dataset,
    *,
    seed: int,
    epochs: int,
    batch_size: int,
    device: torch.device,
    checkpoint_dir: Path,
    num_classes: int = NUM_CLASSES,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Train one variant for one seed and return its evidence record.

    ``progress`` receives the run's status lines: the start of each epoch, the
    periodic batch/sample updates from :func:`train_epoch` and :func:`evaluate`,
    the development and final-test phase boundaries, the selected checkpoint and
    the total run time. Leaving it ``None`` keeps the run quiet for library use.
    """
    run_start = time.perf_counter()
    emit = _scoped_progress(progress, f"{variant} seed {seed}")
    generator = seed_everything(seed)
    model = build_model(variant, num_classes=num_classes).to(device)
    frozen_prefixes: tuple[str, ...] = (BACKBONE_MODULE,) if variant == "pretrained" else ()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(trainable_parameters(model), lr=LEARNING_RATE, momentum=MOMENTUM)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, generator=generator
    )
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_count = sum(parameter.numel() for parameter in trainable_parameters(model))
    _emit_progress(
        emit,
        f"start: epochs={epochs} batch_size={batch_size} device={device} "
        f"train={_count_text(_known_length(train_dataset))} "
        f"dev={_count_text(_known_length(dev_dataset))} "
        f"test={_count_text(_known_length(test_dataset))} "
        f"trainable_parameters={trainable_count:,}",
    )

    history: list[dict[str, Any]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    best_dev_macro_f1 = -1.0
    train_seconds = 0.0
    dev_seconds = 0.0
    for epoch in range(1, epochs + 1):
        epoch_emit = _scoped_progress(emit, f"epoch {epoch}/{epochs}")
        _emit_progress(
            epoch_emit,
            f"training started ({_count_text(_known_length(train_loader))} batches, "
            f"up to {batch_size} samples each)",
        )
        start = time.perf_counter()
        train_stats = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            frozen_prefixes=frozen_prefixes,
            progress=epoch_emit,
        )
        epoch_train_seconds = time.perf_counter() - start
        _emit_progress(
            epoch_emit,
            f"training finished: loss={train_stats['loss']:.4f} "
            f"({_format_seconds(epoch_train_seconds)})",
        )
        _emit_progress(
            epoch_emit,
            f"development evaluation started ({_count_text(_known_length(dev_loader))} batches, "
            f"{_count_text(_known_length(dev_dataset))} samples)",
        )
        start = time.perf_counter()
        dev_stats = evaluate(
            model, dev_loader, criterion, device, num_classes=num_classes, progress=epoch_emit
        )
        epoch_dev_seconds = time.perf_counter() - start
        _emit_progress(
            epoch_emit,
            f"development evaluation finished: loss={dev_stats['loss']:.4f} "
            f"accuracy={dev_stats['accuracy']:.4f} macro_f1={dev_stats['macro_f1']:.4f} "
            f"({_format_seconds(epoch_dev_seconds)})",
        )
        train_seconds += epoch_train_seconds
        dev_seconds += epoch_dev_seconds
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_stats["loss"],
                "train_seconds": epoch_train_seconds,
                "dev_loss": dev_stats["loss"],
                "dev_accuracy": dev_stats["accuracy"],
                "dev_macro_f1": dev_stats["macro_f1"],
                "dev_seconds": epoch_dev_seconds,
            }
        )
        if dev_stats["macro_f1"] > best_dev_macro_f1:
            best_dev_macro_f1 = dev_stats["macro_f1"]
            best_epoch = epoch
            best_state = {
                name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError("no epoch completed, so no checkpoint could be selected")
    model.load_state_dict(best_state)
    checkpoint_path = save_checkpoint(
        model, checkpoint_dir, variant, seed, best_epoch, best_dev_macro_f1
    )
    _emit_progress(
        emit,
        f"selected checkpoint: epoch {best_epoch}/{epochs} "
        f"dev_macro_f1={best_dev_macro_f1:.4f} -> {checkpoint_path}",
    )

    _emit_progress(
        emit,
        f"final test evaluation started ({_count_text(_known_length(test_loader))} batches, "
        f"{_count_text(_known_length(test_dataset))} samples)",
    )
    test_start = time.perf_counter()
    test_stats = evaluate(
        model, test_loader, criterion, device, num_classes=num_classes, progress=emit
    )
    test_seconds = time.perf_counter() - test_start
    _emit_progress(
        emit,
        f"final test evaluation finished: loss={test_stats['loss']:.4f} "
        f"accuracy={test_stats['accuracy']:.4f} macro_f1={test_stats['macro_f1']:.4f} "
        f"({_format_seconds(test_seconds)})",
    )
    total_seconds = time.perf_counter() - run_start
    _emit_progress(emit, f"run finished ({_format_seconds(total_seconds)} total)")

    return {
        "variant": variant,
        "seed": seed,
        "device": str(device),
        "epochs": epochs,
        "backbone": {
            "module": BACKBONE_MODULE,
            "frozen": variant == "pretrained",
            "frozen_eval_prefixes": list(frozen_prefixes),
        },
        "parameters": {
            "total": total_parameters,
            "trainable": trainable_count,
            "frozen": total_parameters - trainable_count,
        },
        "history": history,
        "selection": {
            "criterion": "dev_macro_f1",
            "epoch": best_epoch,
            "dev_macro_f1": best_dev_macro_f1,
            "tie_break": "first epoch with the best development macro F1",
        },
        "final_test": {
            "loss": test_stats["loss"],
            "accuracy": test_stats["accuracy"],
            "macro_f1": test_stats["macro_f1"],
            "samples": test_stats["samples"],
        },
        "timings_seconds": {
            "train": train_seconds,
            "dev": dev_seconds,
            "test": test_seconds,
            "total": total_seconds,
        },
    }


def _count_classes(labels: np.ndarray) -> dict[str, int]:
    """Per-class sample counts for every class index, including absent ones."""
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=NUM_CLASSES)
    return {str(index): int(count) for index, count in enumerate(counts)}


def _weights_metadata() -> dict[str, Any]:
    """Public provenance for the ImageNet checkpoint used by the pretrained variant."""
    from torchvision.models import DenseNet161_Weights

    weights = DenseNet161_Weights.IMAGENET1K_V1
    metrics = weights.meta.get("_metrics", {}).get("ImageNet-1K", {})
    return {
        "name": str(weights),
        "url": weights.url,
        "source_imagenet_1k_metrics": {str(key): float(value) for key, value in metrics.items()},
        "note": (
            "publicly reported metadata of the checkpoint, not a result of this "
            "experiment; the pretrained variant loads these weights and replaces the "
            "1000-class head with a fresh 100-class head"
        ),
    }


def _build_protocol(
    *,
    train_index: np.ndarray,
    dev_index: np.ndarray,
    labels: np.ndarray,
    test_labels: np.ndarray,
    transform_spec: dict[str, Any],
    seeds: Sequence[int],
    epochs: int,
    batch_size: int,
    threads: int,
    device: str,
    runs: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """Describe the dataset, splits, preprocessing, settings and selection policy."""
    parameter_counts = {
        variant: {
            "total": records[0]["parameters"]["total"],
            "trainable": records[0]["parameters"]["trainable"],
        }
        for variant, records in runs.items()
    }
    return {
        "provenance": {
            "dataset": "CIFAR-100 (100 classes, 600 images per class)",
            "loader": "torchvision.datasets.CIFAR100",
            "official_train_samples": int(labels.size),
            "official_test_samples": int(test_labels.size),
            "coursework": (
                "Protocol rebuild of the CSC781 Module 8 Assignment 9 transfer-learning "
                "exercise; no historical outputs or report scores are reused."
            ),
            "initial_weights": {
                "scratch": "weights=None (random initialisation)",
                "pretrained": _weights_metadata(),
            },
        },
        "splits": {
            "strategy": "stratified 90/10 carve of the official CIFAR-100 training split",
            "seed": SPLIT_SEED,
            "policy": (
                "both variants train on the same 45000 images and select checkpoints on "
                "the same 5000 development images; the official 10000-image test split is "
                "a separate dataset scored once after selection"
            ),
            "counts": {
                "train": int(train_index.size),
                "dev": int(dev_index.size),
                "test": int(test_labels.size),
            },
            "class_counts": {
                "train": _count_classes(labels[train_index]),
                "dev": _count_classes(labels[dev_index]),
                "test": _count_classes(test_labels),
            },
        },
        "preprocessing": transform_spec,
        "seeds": {
            "model_and_shuffle": [int(seed) for seed in seeds],
            "split": SPLIT_SEED,
            "python_random": "random.seed(seed)",
            "numpy": "np.random.seed(seed)",
            "torch": "torch.manual_seed(seed)",
            "dataloader_shuffle": "torch.Generator().manual_seed(seed), shared by both variants",
            "cudnn": "deterministic=True, benchmark=False when CUDA is used",
        },
        "hyperparameters": {
            "optimizer": "torch.optim.SGD",
            "learning_rate": LEARNING_RATE,
            "momentum": MOMENTUM,
            "weight_decay": 0.0,
            "batch_size": batch_size,
            "epochs": epochs,
            "loss": "torch.nn.CrossEntropyLoss",
            "loader_workers": 0,
            "torch_threads": threads,
        },
        "device": device,
        "selection": {
            "criterion": "best development macro F1 over all 100 classes",
            "tie_break": "earliest epoch",
            "held_out": (
                "the official test split is evaluated once per run after reloading the "
                "selected checkpoint"
            ),
        },
        "budget": {
            "policy": (
                "fixed epoch budget (default 1 epoch); this is a disclosed compute "
                "budget, not a convergence criterion"
            )
        },
        "freezing": {
            "scratch": "every parameter is trainable",
            "pretrained": (
                "all backbone parameters require no gradient and the features module is "
                "re-applied to eval() after every model.train(), so BatchNorm running "
                "buffers stay frozen; only the replacement classifier is optimised"
            ),
            "optimizer_parameters": "only parameters with requires_grad=True",
            "caveat": (
                "the pretrained variant therefore has far fewer trainable parameters "
                "than the scratch variant; the comparison measures initialisation plus "
                "freezing as deployed, not an equal-parameter-count control"
            ),
            "parameter_counts": parameter_counts,
        },
        "metrics": {
            "loss": "sample-count-weighted mean cross-entropy across batches",
            "accuracy": "fraction of correctly classified samples",
            "macro_f1": (
                "unweighted mean of per-class F1 over all 100 CIFAR-100 classes (zero_division=0)"
            ),
        },
    }


def run_experiment(
    *,
    epochs: int = DEFAULT_EPOCHS,
    seeds: Sequence[int] = (DEFAULT_SEED,),
    batch_size: int = DEFAULT_BATCH_SIZE,
    device: str = "auto",
    threads: int = DEFAULT_THREADS,
    data_dir: Path = DEFAULT_DATA_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Run the matched protocol for every variant and seed; return the evidence payload.

    ``progress`` is forwarded to every :func:`run_variant` call so the CLI can
    stream status lines; leaving it ``None`` keeps the function quiet.
    """
    if epochs < 1:
        raise ValueError(f"epochs must be >= 1, got {epochs}")
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    if threads < 1:
        raise ValueError(f"threads must be >= 1, got {threads}")
    seed_list = [int(seed) for seed in seeds]
    if not seed_list:
        raise ValueError("at least one seed is required")

    torch.set_num_threads(threads)
    resolved_device = resolve_device(device)
    transform = build_transform()
    _emit_progress(progress, f"loading CIFAR-100 from {Path(data_dir)}")
    train_source, test_source = load_cifar100(Path(data_dir), transform)
    labels = np.asarray(train_source.targets, dtype=np.int64)
    test_labels = np.asarray(test_source.targets, dtype=np.int64)
    train_index, dev_index = split_train_dev(labels)
    train_dataset = Subset(train_source, train_index.tolist())
    dev_dataset = Subset(train_source, dev_index.tolist())
    checkpoint_dir = Path(output_dir) / CHECKPOINT_DIRECTORY
    _emit_progress(
        progress,
        f"split: {train_index.size} train / {dev_index.size} dev samples; "
        f"{test_labels.size} official test samples",
    )

    runs: dict[str, list[dict[str, Any]]] = {}
    for variant in VARIANTS:
        runs[variant] = [
            run_variant(
                variant,
                train_dataset,
                dev_dataset,
                test_source,
                seed=seed,
                epochs=epochs,
                batch_size=batch_size,
                device=resolved_device,
                checkpoint_dir=checkpoint_dir,
                progress=progress,
            )
            for seed in seed_list
        ]

    return {
        "protocol": _build_protocol(
            train_index=train_index,
            dev_index=dev_index,
            labels=labels,
            test_labels=test_labels,
            transform_spec=describe_transform(transform),
            seeds=seed_list,
            epochs=epochs,
            batch_size=batch_size,
            threads=threads,
            device=str(resolved_device),
            runs=runs,
        ),
        "environment": environment(),
        "runs": runs,
        "summary": summarize_runs(runs),
    }


def _collapsed(values: list[float]) -> dict[str, Any]:
    """Per-run values plus their mean, population standard deviation and range."""
    array = np.asarray(values, dtype=float)
    return {
        "values": [float(value) for value in array],
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def summarize_runs(runs: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Aggregate final metrics across seeds per variant, plus the paired delta."""
    per_variant: dict[str, Any] = {}
    for variant, records in runs.items():
        per_variant[variant] = {
            "seeds": [record["seed"] for record in records],
            "final_test": {
                metric: _collapsed([record["final_test"][metric] for record in records])
                for metric in ("loss", "accuracy", "macro_f1")
            },
            "selected_dev_macro_f1": _collapsed(
                [record["selection"]["dev_macro_f1"] for record in records]
            ),
            "selected_epoch": _collapsed(
                [float(record["selection"]["epoch"]) for record in records]
            ),
            "parameters": records[0]["parameters"],
        }
    return {
        "std_kind": "population standard deviation across the recorded seeds",
        "per_variant": per_variant,
        "delta_pretrained_minus_scratch": {
            metric: per_variant["pretrained"]["final_test"][metric]["mean"]
            - per_variant["scratch"]["final_test"][metric]["mean"]
            for metric in ("loss", "accuracy", "macro_f1")
        },
    }


# %% Figures
# Learning-curve panels label every recorded epoch while the run is short; longer
# runs fall back to a locator that is still pinned to whole epochs.
MAX_EPOCH_TICKS = 12


def save_figure(fig: plt.Figure, output_path: Path) -> None:
    """Save a figure as a PNG and release it."""
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_learning_curves(runs: dict[str, list[dict[str, Any]]], output_path: Path) -> None:
    """Plot development macro F1 and loss per epoch for each variant and seed.

    Series are drawn against the whole epochs that were actually recorded. A run
    that recorded a single epoch is a single measured point: it is drawn as an
    isolated marker on the integer tick ``1``, so the panel cannot be read as a
    curve over a fractional epoch axis.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.5))
    panels = (
        (axes[0], "dev_macro_f1", "Development macro F1", "macro F1"),
        (axes[1], "dev_loss", "Development loss", "cross-entropy"),
    )
    repeats = max(len(records) for records in runs.values())
    observed_epochs = sorted(
        {
            int(entry["epoch"])
            for records in runs.values()
            for record in records
            for entry in record["history"]
        }
    )
    single_epoch = len(observed_epochs) == 1
    for variant, records in runs.items():
        color = _VARIANT_COLORS[variant]
        for panel, key, _, _ in panels:
            for record in records:
                epochs = [int(entry["epoch"]) for entry in record["history"]]
                values = [entry[key] for entry in record["history"]]
                panel.plot(
                    epochs,
                    values,
                    color=color,
                    marker="o",
                    # A one-epoch run is one measured point, so draw no line at
                    # all: there is no measured trajectory to connect.
                    linestyle="none" if len(epochs) == 1 else "-",
                    markersize=7.0 if len(epochs) == 1 else 6.0,
                    linewidth=1.4,
                    alpha=0.35 if repeats > 1 else 0.9,
                    label=f"{variant} (seed {record['seed']})",
                )
        if repeats > 1:
            epochs = [int(entry["epoch"]) for entry in records[0]["history"]]
            for panel, key, _, _ in panels:
                panel.plot(
                    epochs,
                    [
                        float(np.mean([entry[key] for entry in record["history"]]))
                        for record in records
                    ],
                    color=color,
                    linestyle="--",
                    marker="s" if len(epochs) == 1 else None,
                    markersize=6.0,
                    linewidth=2.0,
                    label=f"{variant} mean",
                )
    for panel, _, title, ylabel in panels:
        panel.set_title(f"{title} (1 epoch)" if single_epoch else title)
        panel.set_xlabel("epoch")
        panel.set_ylabel(ylabel)
        panel.grid(alpha=0.3)
        # Ticks sit on the whole epochs that were recorded: a one-epoch run gets
        # a single tick at 1 instead of a fractional 0.96-1.04 axis.
        if len(observed_epochs) <= MAX_EPOCH_TICKS:
            panel.set_xticks(observed_epochs)
        else:
            panel.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=MAX_EPOCH_TICKS))
        span = observed_epochs[-1] - observed_epochs[0]
        padding = 0.5 if span == 0 else max(0.4, 0.03 * span)
        panel.set_xlim(observed_epochs[0] - padding, observed_epochs[-1] + padding)
    axes[0].legend(fontsize=8)
    fig.tight_layout(pad=1.0, w_pad=2.0)
    save_figure(fig, output_path)


def plot_comparison(runs: dict[str, list[dict[str, Any]]], output_path: Path) -> None:
    """Compare final test scores and parameter counts between the two variants.

    Error bars are drawn only when more than one seed was recorded: a single-seed
    run shows its measured scores as plain bars whose title says so, instead of a
    zero-length population-std interval that would imply a variability estimate
    the run does not contain.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.5))
    seed_count = max(len(records) for records in runs.values())
    metric_specs = (("accuracy", "accuracy"), ("macro_f1", "macro F1"))
    positions = np.arange(len(metric_specs), dtype=float)
    width = 0.35
    for variant_index, variant in enumerate(VARIANTS):
        records = runs[variant]
        means = [
            float(np.mean([record["final_test"][metric] for record in records]))
            for metric, _ in metric_specs
        ]
        errors = (
            [
                float(np.std([record["final_test"][metric] for record in records]))
                for metric, _ in metric_specs
            ]
            if seed_count > 1
            else None
        )
        axes[0].bar(
            positions + (variant_index - (len(VARIANTS) - 1) / 2) * width,
            means,
            width * 0.9,
            yerr=errors,
            capsize=4 if errors is not None else 0,
            color=_VARIANT_COLORS[variant],
            label=variant,
        )
    axes[0].set_xticks(positions, [label for _, label in metric_specs])
    axes[0].set_ylim(0.0, 1.05)
    axes[0].set_ylabel("held-out test score")
    axes[0].set_title(
        "Official test split (1 seed)\nno error bars"
        if seed_count == 1
        else f"Official test split ({seed_count} seeds)\nerror bars: population std"
    )
    axes[0].legend(fontsize=8)

    parameter_positions = np.arange(len(VARIANTS), dtype=float)
    totals = [runs[variant][0]["parameters"]["total"] for variant in VARIANTS]
    trainable = [runs[variant][0]["parameters"]["trainable"] for variant in VARIANTS]
    axes[1].bar(parameter_positions - 0.2, totals, 0.4, label="total", color="tab:gray")
    axes[1].bar(parameter_positions + 0.2, trainable, 0.4, label="trainable", color="tab:green")
    bar_series = (
        (parameter_positions - 0.2, totals),
        (parameter_positions + 0.2, trainable),
    )
    for bar_positions, values in bar_series:
        for position, value in zip(bar_positions, values, strict=True):
            axes[1].text(position, value, f"{value:,}", ha="center", va="bottom", fontsize=8)
    axes[1].set_yscale("log")
    axes[1].set_xticks(parameter_positions, list(VARIANTS))
    axes[1].set_ylabel("parameters (log scale)")
    axes[1].set_title("Unequal trainable parameters\n(frozen backbone vs new head)")
    axes[1].legend(fontsize=8)
    # Reserved inter-panel spacing keeps both two-line titles fully inside their
    # own panel instead of colliding across the gap.
    fig.tight_layout(pad=1.0, w_pad=2.0)
    save_figure(fig, output_path)


def write_figures(runs: dict[str, list[dict[str, Any]]], output_dir: Path) -> dict[str, Path]:
    """Write the learning-curve and comparison PNGs, returning their paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    curve_path = output_dir / LEARNING_CURVE_FIGURE
    comparison_path = output_dir / COMPARISON_FIGURE
    plot_learning_curves(runs, curve_path)
    plot_comparison(runs, comparison_path)
    return {"learning_curves": curve_path, "comparison": comparison_path}


# %% Entry point
def main(argv: Sequence[str] | None = None) -> int:
    """Run the matched DenseNet-161 protocol and write JSON evidence plus figures."""
    parser = argparse.ArgumentParser(
        description="Matched DenseNet-161 scratch vs ImageNet transfer on CIFAR-100",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help="fixed training budget per variant (default: 1)",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[DEFAULT_SEED],
        help="model/shuffle seeds; each one reruns both variants (default: 42)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="'auto', 'cpu' or 'cuda' (default: auto)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="batch size for train/dev/test (default: 32)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=DEFAULT_THREADS,
        help="torch CPU threads (default: 4)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="torchvision CIFAR-100 root (default: data/)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="directory for transfer.json and the PNG figures (default: results/)",
    )
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    payload = run_experiment(
        epochs=args.epochs,
        seeds=args.seeds,
        batch_size=args.batch_size,
        device=args.device,
        threads=args.threads,
        data_dir=Path(args.data_dir),
        output_dir=output_dir,
        progress=_print_progress,
    )
    figure_paths = write_figures(payload["runs"], output_dir)
    payload["figures"] = {role: path.name for role, path in figure_paths.items()}
    write_results(output_dir, RESULT_NAME, payload)

    for variant, records in payload["runs"].items():
        for record in records:
            final = record["final_test"]
            print(
                f"{variant} seed {record['seed']}: "
                f"dev_macro_f1={record['selection']['dev_macro_f1']:.4f} "
                f"test_accuracy={final['accuracy']:.4f} "
                f"test_macro_f1={final['macro_f1']:.4f} "
                f"trainable_parameters={record['parameters']['trainable']:,}",
                flush=True,
            )
    print(f"wrote {RESULT_NAME}.json and {len(figure_paths)} figures to {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
