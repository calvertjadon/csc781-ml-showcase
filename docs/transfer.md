# DenseNet-161 on CIFAR-100: scratch vs frozen ImageNet-1K transfer

## Question

Does a DenseNet-161 initialised from ImageNet-1K with a frozen backbone and a fresh 100-class head beat a randomly initialised DenseNet-161 under the same one-epoch SGD budget, stratified CIFAR-100 split, preprocessing and seed? This rebuild keeps the original question and fixes the methodology. All values come from [`results/transfer.json`](../results/transfer.json); no historical score is reused.

## Contribution vs libraries

torchvision supplies the DenseNet-161 architecture, the `DenseNet161_Weights.IMAGENET1K_V1` checkpoint and its transform recipe; scikit-learn supplies macro F1; `matplotlib`/`json` render the evidence. The contribution in [`mlshowcase/transfer.py`](../mlshowcase/transfer.py) is the protocol harness: model construction, genuine freezing, seeded shuffling, development-macro-F1 selection, held-out evaluation and the evidence writers.

## Corrections to the original implementation

- **Weights API.** `densenet161(pretrained=True/False)` → `weights=None` (scratch) / `DenseNet161_Weights.IMAGENET1K_V1` (transfer) plus the weights' own transforms.
- **Normalisation.** The original resized to 224 px without the ImageNet normalisation the weights expect; the rebuild inherits the recipe's mean/std and bilinear antialiasing.
- **Real freezing.** `requires_grad = False` does not stop `model.train()` from updating BatchNorm running buffers; the rebuild re-applies `eval()` to `features` after every `train()`, and the optimiser owns only trainable parameters.
- **Matched budget, no test peeking.** The original ran 20 epochs for scratch and 5 for transfer while printing test accuracy; the rebuild gives both the same fixed budget and scores the official test split once after development-macro-F1 selection.
- **Determinism.** Python, NumPy and torch are seeded; both variants seed their DataLoader shuffle generator with 42.

[`tests/test_transfer.py`](../tests/test_transfer.py) pins frozen BatchNorm buffers, the trainable head, sample-count-weighted losses and 100-class macro F1 (`zero_division=0`).

## Protocol

- **Data.** Official CIFAR-100 (50,000 train / 10,000 test, 100 classes) via `torchvision.datasets.CIFAR100`; a stratified 90/10 carve of the official training split (split seed 42, fixed across seeds) gives 45,000 train / 5,000 dev (450/50 per class); the official test split stays untouched until after selection.
- **Preprocessing.** Both variants share one transform from the weights recipe, `DenseNet161_Weights.IMAGENET1K_V1.transforms(crop_size=64, resize_size=74)`: ImageNet mean `[0.485, 0.456, 0.406]` / std `[0.229, 0.224, 0.225]` and bilinear antialiasing inherited; 64/74 overrides the recipe's 224/256 as a disclosed CPU-budget deviation; no augmentation.
- **Models.** Scratch (`weights=None`): all **26,692,900** parameters trainable. Transfer: ImageNet weights with backbone parameters and BatchNorm running buffers frozen (`features` held in `eval()`), fresh 100-class head — **220,900** trainable of 26,692,900 (26,472,000 frozen).
- **Settings.** SGD lr 0.01, momentum 0.9, batch 32, one epoch, `CrossEntropyLoss`, 4 torch threads, CPU, seed 42; both variants share the split and shuffle order.
- **Selection.** Best development macro F1 over all 100 classes, tie-break earliest epoch; the selected checkpoint is reloaded and the official test split scored once. With one epoch there is exactly one candidate, so selection could not discriminate; the rule keeps the test split untouched.

## Results

| Variant | Trainable / total params | Dev macro F1 | Test loss | Test accuracy | Test macro F1 | Train (s) | Total (s) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| scratch | 26,692,900 / 26,692,900 | 0.2070 | 3.1360 | 0.2328 | 0.2103 | 959.25 | 1054.55 |
| pretrained | 220,900 / 26,692,900 | 0.4904 | 6.9069 | 0.4884 | 0.4854 | 291.61 | 389.26 |

The transfer variant wins on accuracy and macro F1 but **its cross-entropy is worse** — 6.9069 against 3.1360, the same direction as development (6.6121 vs 3.1647). Both are reported as measured; no calibration statistics were collected and no cause or calibration claim is made.

## Interpretation and limitations

- **≈24 min CPU wall time** (1054.55 s + 389.26 s; AMD Ryzen 7 7840U, 4 torch threads, torch 2.6.0+cpu / torchvision 0.21.0+cpu).
- **One seed, so no variance is estimated**: a JSON `std` of 0.0 over one value is not evidence of zero variability.
- **One epoch is a budget, not convergence**, and scratch's 0.2328 is one pass over 45,000 images, not a converged baseline.
- **64 px, not the checkpoint's 224 px** evaluation resolution.
- **Unequal trainable counts and compute** (220,900 vs 26,692,900 updated parameters) measure the recipes as deployed, not initialisation alone.
- **One dataset, one architecture**: the evidence covers this CIFAR-100 / DenseNet-161 protocol only.

## Figures and evidence

[`transfer_learning_curves.png`](../results/transfer_learning_curves.png) shows development macro F1 and cross-entropy per epoch (single markers; one epoch has no trajectory); [`transfer_comparison.png`](../results/transfer_comparison.png) shows final test accuracy and macro F1 (single seed, no error bars) and the log-scale parameter panel.

Implementation [`mlshowcase/transfer.py`](../mlshowcase/transfer.py) · tests [`tests/test_transfer.py`](../tests/test_transfer.py) · evidence [`results/transfer.json`](../results/transfer.json) · notebook [`notebooks/transfer.ipynb`](../notebooks/transfer.ipynb).

## Reproduce

```sh
uv run --locked --extra deep-learning python -m mlshowcase.transfer --device cpu --epochs 1 --seeds 42 --data-dir data
```

The first run downloads CIFAR-100 into git-ignored `data/` and the IMAGENET1K_V1 checkpoint from `download.pytorch.org`; neither is redistributed. The command rewrites `results/transfer.json`, both PNGs and git-ignored checkpoints under `results/checkpoints/`.

## Provenance and citations

- **CIFAR-100** — Krizhevsky (2009), *Learning Multiple Layers of Features from Tiny Images*, <https://www.cs.toronto.edu/~kriz/cifar.html>; downloaded at run time, not redistributed.
- **DenseNet-161** — Huang et al. (2017), *Densely Connected Convolutional Networks*, <https://arxiv.org/abs/1608.06993>; weights `DenseNet161_Weights.IMAGENET1K_V1` from `download.pytorch.org`, not bundled; macro F1 via scikit-learn `f1_score`.
- Full dataset, weights and software attribution: [`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md); the MIT licence covers this repository's code and documentation only.
