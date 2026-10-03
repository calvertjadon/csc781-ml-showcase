# Changelog

## 2026-10-03 — Public showcase

### Added

- Separate clean-history showcase with three reusable experiment modules,
  explanatory notebooks, measured JSON/PNG evidence, and short technical studies.
- Locked Python 3.12 setup, optional CPU PyTorch extra, one command per experiment,
  environment/runtime recording, and behavioral regression tests.
- MIT license for the author's code/documentation and explicit third-party
  dataset/weight/software attribution and exclusions.

### Corrected from the coursework

- **k-NN:** omitted final feature in distance calculations; mixed float labels;
  commented-out tuning sweep; unseeded splits. All-feature arithmetic, separate
  integer labels, seeded stratified splits, and development selection replace them.
  Vote-count ties choose the smallest label; equal-distance ranks keep training
  order, which can affect membership at the neighborhood boundary.
- **Logistic:** overflow-prone sigmoid, independently normalized splits, and
  evaluating the last-fitted rather than selected parameters. Stable logit-based
  loss, train-only scaling, joint development selection, explicit optimizer
  termination, and malignant-positive labels replace those choices.
- **Transfer:** legacy weights API, missing ImageNet normalization, BatchNorm
  buffers updating in a supposedly frozen backbone, unequal epoch budgets, and
  test-set monitoring. Modern weights-derived transforms, genuine freezing,
  matched epochs, development checkpoint selection, and final-only testing
  replace them. The 64-pixel resolution and one-epoch budget remain explicit.

### Measured and verified

- Ran k-NN and logistic experiments for seeds 42/43/44 with library baselines.
  k-NN test predictions agreed on every recorded split. Full scores and timings:
  [`results/knn.json`](results/knn.json), [`results/logistic.json`](results/logistic.json).
- Trained scratch and frozen-pretrained DenseNet-161 for one full-data epoch each:
  45,000 train / 5,000 dev / 10,000 official test images. This was actual training,
  separate from the two-sample real-model training/evaluation smoke check.
  Evidence: [`results/transfer.json`](results/transfer.json).
- Ruff checks passed; **59 behavioral tests passed** with the deep-learning extra.
  All three notebooks executed in a torch-free base environment; wrong-root
  notebook execution was rejected before imports/result reads.
- Generated and visually inspected result figures. Launched the documented
  JupyterLab workflow and checked notebook presentation.

### Scope and limits

No original prompts, submission reports, saved outputs, missing-data experiment,
private archive links, downloaded datasets or model checkpoints are published.
No historical score is reproduced or claimed. Classical results cover only three
overlapping splits; capped logistic fits and one-epoch DenseNets are not converged
comparisons. Transfer has one seed, unequal trainable parameters and unequal
compute; improved F1 accompanied worse cross-entropy. No clinical, calibration,
benchmark or state-of-the-art claim is made.
