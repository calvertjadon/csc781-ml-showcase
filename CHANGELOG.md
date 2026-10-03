# Changelog

## 2026-10-03 prose edits

- Rewrote the README, studies, attribution, and notebooks in plain language.
- Shortened comments and docstrings, expanded compressed explanations, and
  removed rhetorical framing and repeated disclaimers.
- Updated command help, result descriptions, and figure labels without changing
  calculations, settings, recorded measurements, or license terms.
- Adjusted nearest-neighbor figure spacing so the new captions do not overlap
  the axis labels.

## 2026-10-03 public release

### Added

- Created a separate repository with three Python experiment modules, notebooks,
  JSON results, figures, and technical studies.
- Added a locked Python 3.12 setup, optional CPU PyTorch dependencies, reproduction
  commands, environment and timing records, and behavioral tests.
- Added an MIT license for the author's code and documentation. Recorded the
  upstream sources and separate terms for datasets, weights, and software.

### Corrections to the coursework

- The k-NN code now includes the final feature in each distance calculation,
  keeps integer labels separate, and runs the development sweep. Seeded,
  stratified splits replace unseeded splits. Tied votes choose the smallest
  label. Tied distances retain training row order, which can change membership
  at the neighborhood boundary.
- The logistic code now uses a stable sigmoid and loss, fits scaling statistics
  on training data, and evaluates the selected model rather than the last fit.
  Development data selects the learning rate and threshold together. The
  optimizer has a step limit and gradient tolerance. Malignant is positive class 1.
- The transfer code now uses the weights API and ImageNet normalization. It
  freezes both backbone parameters and BatchNorm buffers. The two models receive
  the same epoch budget. Development data selects checkpoints, and the official
  test split evaluates only the selected checkpoint. The study records the
  one-epoch budget and 64-pixel resolution.

### Recorded runs and checks

- Ran the k-NN and logistic experiments for seeds 42, 43, and 44 with library
  baselines. The k-NN implementations agreed on every test prediction. The
  [k-NN results](results/knn.json) and [logistic results](results/logistic.json)
  contain the full scores and timings.
- Trained scratch and frozen-pretrained DenseNet-161 for one epoch each on 45,000
  images. Both used 5,000 development images and 10,000 official test images.
  These runs were separate from the two-image training and evaluation smoke
  check. The [transfer results](results/transfer.json) contain the measurements.
- Ruff checks passed, and the full suite passed 59 tests with the deep-learning
  extra. All notebooks ran without torch. They rejected the wrong checkout root
  before importing experiment code or reading results.
- Generated and inspected the figures. Launched JupyterLab and checked a notebook
  and its link to the study.

### Limits

The repository excludes original prompts, submitted reports, saved outputs,
private archive links, downloaded datasets, model checkpoints, and the experiment
with missing CSV data. It does not reuse historical scores.

The classical results cover three overlapping splits. The capped logistic fits
and one-epoch DenseNets did not establish convergence. The transfer comparison
has one seed and unequal parameter counts and compute. Transfer improved F1 but
increased cross-entropy. These experiments do not establish clinical suitability,
calibration, or state-of-the-art performance.
