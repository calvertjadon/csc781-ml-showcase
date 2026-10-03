"""Consumer-visible behaviour of mlshowcase.transfer.

Every fixture is a tiny real ``nn.Module`` graph, the backbone fixtures sharing
DenseNet's ``features`` / ``classifier`` split, with hand-set weights whose
gradient flow can be checked by hand and hand-written logits whose per-sample
losses can be worked out by hand, so the expected values come from worked
arithmetic rather than from the module under test. Nothing here
downloads model weights, loads CIFAR-100, or trains a full model. Torch is an
optional dependency, so this whole module is skipped when it is not installed.
"""

from __future__ import annotations

import importlib
import math
from typing import Any

import pytest

torch = pytest.importorskip("torch")

transfer = importlib.import_module("mlshowcase.transfer")


class TinyBatchNormNet(torch.nn.Module):
    """Minimal real network with the same ``features``/``classifier`` split as DenseNet."""

    def __init__(self, num_classes: int = 100) -> None:
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 1, kernel_size=1, bias=False),
            torch.nn.BatchNorm2d(1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
        )
        self.classifier = torch.nn.Linear(1, num_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(features))


class FixedLogits(torch.nn.Module):
    """Return a prescribed logit row for every integer index in the batch."""

    def __init__(self, table: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("table", table)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        return self.table[indices]


class TrainableLogitBias(torch.nn.Module):
    """Add a zero-initialised trainable per-class bias to per-sample logits."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros(num_classes))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits + self.bias


def _tiny_model(num_classes: int = 100) -> TinyBatchNormNet:
    """Build the fixture with hand-set weights that keep every gradient path active.

    The three fixture images light one input channel each with value 1, 2 or 4,
    the BatchNorm bias is 2 with unit running statistics, and the convolution has
    no bias, so every pre-activation and BatchNorm output is positive and ReLU
    never masks a gradient.
    """
    torch.manual_seed(0)
    model = TinyBatchNormNet(num_classes=num_classes)
    with torch.no_grad():
        model.features[0].weight.copy_(torch.tensor([1.0, 2.0, 4.0]).reshape(1, 3, 1, 1))
        model.features[1].weight.fill_(1.0)
        model.features[1].bias.fill_(2.0)
        model.features[1].running_mean.zero_()
        model.features[1].running_var.fill_(1.0)
    return model


def _one_hot_images() -> torch.Tensor:
    """Three 1x1 images, each lighting one input channel: values 1, 2 and 4."""
    return torch.eye(3).reshape(3, 3, 1, 1)


def _score_classes_zero_and_one(model: TinyBatchNormNet) -> None:
    """Score class 0 as ``+h`` and class 1 as ``-h``; all other logits stay flat.

    The hand-checked backbone gradient in the scratch test uses these exact head
    weights: other head rows would rescale the BatchNorm input gradients.
    """
    with torch.no_grad():
        model.classifier.weight.zero_()
        model.classifier.bias.zero_()
        model.classifier.weight[0, 0] = 1.0
        model.classifier.weight[1, 0] = -1.0


def _loader(images: torch.Tensor, labels: list[int], batch_size: int = 3) -> Any:
    """Deterministic loader over the fixture tensors (no shuffling, no workers)."""
    dataset = torch.utils.data.TensorDataset(images, torch.tensor(labels))
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)


def test_transfer_training_updates_the_head_but_freezes_backbone_and_bn_buffers() -> None:
    model = _tiny_model()
    transfer.freeze_backbone_for_transfer(model, num_classes=100)
    trainable_ids = {id(parameter) for parameter in transfer.trainable_parameters(model)}
    assert {
        name for name, parameter in model.named_parameters() if id(parameter) in trainable_ids
    } == {"classifier.weight", "classifier.bias"}

    conv_before = model.features[0].weight.detach().clone()
    running_mean_before = model.features[1].running_mean.detach().clone()
    running_var_before = model.features[1].running_var.detach().clone()
    head_before = model.classifier.weight.detach().clone()

    optimizer = torch.optim.SGD(transfer.trainable_parameters(model), lr=0.1, momentum=0.9)
    # Every fixture image is labelled class 0, so the fresh head's class-0 row
    # receives the strictly negative gradient mean((p0 - 1) * h) with h > 0.
    stats = transfer.train_epoch(
        model,
        _loader(_one_hot_images(), [0, 0, 0]),
        torch.nn.CrossEntropyLoss(),
        optimizer,
        "cpu",
        frozen_prefixes=("features",),
    )

    assert stats["samples"] == 3
    assert not torch.equal(model.classifier.weight.detach(), head_before)
    assert torch.equal(model.features[0].weight.detach(), conv_before)
    assert torch.equal(model.features[1].running_mean.detach(), running_mean_before)
    assert torch.equal(model.features[1].running_var.detach(), running_var_before)
    assert model.features[1].training is False  # held in eval mode after model.train()


def test_scratch_training_updates_backbone_parameters_and_bn_running_stats() -> None:
    model = _tiny_model()
    _score_classes_zero_and_one(model)
    assert all(parameter.requires_grad for parameter in model.parameters())

    conv_before = model.features[0].weight.detach().clone()
    running_mean_before = model.features[1].running_mean.detach().clone()

    optimizer = torch.optim.SGD(transfer.trainable_parameters(model), lr=0.1, momentum=0.9)
    transfer.train_epoch(
        model,
        _loader(_one_hot_images(), [0, 1, 0]),
        torch.nn.CrossEntropyLoss(),
        optimizer,
        "cpu",
    )

    # With the class-0/class-1 head the BatchNorm input gradients for the three
    # one-hot images are (dx0, dx1, dx2) ~= (-0.68, 1.01, -0.34), so each
    # input-channel weight of the 1x1 convolution moves by lr * dx.
    assert not torch.equal(model.features[0].weight.detach(), conv_before)
    # Running statistics only change while BatchNorm is in training mode.
    assert not torch.equal(model.features[1].running_mean.detach(), running_mean_before)
    assert model.features[1].training is True


def test_train_epoch_weights_losses_by_actual_sample_counts() -> None:
    model = TrainableLogitBias(num_classes=2)
    # Stepping with lr 0 keeps the zero-initialised bias at zero, so every batch
    # sees exactly the hand-written logits below and CrossEntropyLoss reports the
    # independently known cross-entropy of each row.
    optimizer = torch.optim.SGD(transfer.trainable_parameters(model), lr=0.0, momentum=0.9)
    log3 = math.log(3.0)
    logits = torch.tensor([[log3, 0.0], [log3, 0.0], [log3, 0.0], [0.0, log3]])

    # Batches of 3 and 1 samples, all labelled class 0: [log 3, 0] puts
    # p(class 0) = 3/4, so each of the three samples contributes log(4/3), and
    # [0, log 3] puts p(class 0) = 1/4, so the last sample contributes log(4).
    # The sample-count-weighted mean is (3 * log(4/3) + log(4)) / 4, whereas
    # averaging the two batch means would give (log(4/3) + log(4)) / 2.
    stats = transfer.train_epoch(
        model,
        _loader(logits, [0, 0, 0, 0]),
        torch.nn.CrossEntropyLoss(),
        optimizer,
        "cpu",
    )

    weighted = (3 * math.log(4 / 3) + math.log(4)) / 4
    batch_mean = (math.log(4 / 3) + math.log(4)) / 2
    assert stats["loss"] == pytest.approx(weighted, rel=1e-6)
    assert stats["loss"] != pytest.approx(batch_mean, rel=1e-6)
    assert stats["samples"] == 4
    assert stats["batches"] == 2


def test_evaluate_reports_sample_weighted_loss_and_macro_f1_over_all_100_classes() -> None:
    table = torch.zeros(4, 100)
    # Argmax predictions per row: 0, 0, 1, 3; the labels are 0, 0, 1, 2.
    table[0, 0] = 1.0
    table[1, 0] = 2.0
    table[2, 1] = 1.0
    table[3, 3] = 1.0
    model = FixedLogits(table)

    # Batches of 3 and 1 samples. Hand-computed per-sample cross-entropies with
    # a softmax over 99 zero logits plus one hot logit (e = 2.718282):
    #   ln(e^1 + 99) - 1 = 3.622207, ln(e^2 + 99) - 2 = 2.667103,
    #   ln(e^1 + 99) - 1 = 3.622207, ln(e^1 + 99)     = 4.622207.
    metrics = transfer.evaluate(
        model,
        _loader(torch.arange(4), [0, 0, 1, 2], batch_size=3),
        torch.nn.CrossEntropyLoss(),
        "cpu",
    )

    assert metrics["loss"] == pytest.approx(3.633431, rel=1e-4)
    assert metrics["accuracy"] == pytest.approx(0.75)
    # Classes 0 and 1 each score F1 = 1, the missed class 2 and the spurious
    # class 3 score 0, and the macro average still divides by all 100 classes.
    assert metrics["macro_f1"] == pytest.approx(2 / 100)
    assert metrics["samples"] == 4
