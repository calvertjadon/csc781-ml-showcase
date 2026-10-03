# CSC 781 machine-learning experiments

I reworked three experiments from my graduate machine-learning course. This
repository contains the Python implementations, tests, notebooks, and results
from new runs. I corrected problems in the original distance calculations,
normalization, model selection, and transfer-learning setup.

I am Jadon Calvert. The original submissions and instructor materials remain in
 a separate private archive. This repository has its own Git history.

## What I wrote

- The NumPy k-NN code computes distances over every feature, ranks neighbors,
  votes with deterministic tie rules, and selects hyperparameters on development
  data. It reuses each distance ranking across neighborhood sizes.
- The NumPy logistic-regression code computes a stable sigmoid, cross-entropy
  from logits, and an analytic gradient. It fits models with full-batch gradient
  descent and fits scaling statistics on training data only.
- The transfer-learning code trains and evaluates DenseNet-161. It freezes
  backbone parameters and BatchNorm buffers, selects checkpoints on development
  data, and evaluates the selected checkpoint on the official test split.
- The experiment code records splits, settings, package versions, hardware,
  timings, and metrics. It writes JSON results and plots those results.

NumPy provides array operations. Scikit-learn provides the datasets, stratified
splitting, metrics, and reference estimators. Torchvision provides DenseNet-161,
the ImageNet weights, and their preprocessing transforms.

## Measured results

The classical experiments use seeds 42, 43, and 44. Their results show the mean
and population standard deviation across those seeds, not a confidence interval.
The DenseNet comparison uses one seed, one epoch, 64-pixel inputs, and the full
CIFAR-100 dataset. The metrics in different rows are not interchangeable.

| Experiment | NumPy model or scratch training | Library model or pretrained backbone | Held-out metric |
| --- | --- | --- | --- |
| Digits k-NN | 0.9805 ± 0.0045 | 0.9805 ± 0.0045 | 10-class macro F1 |
| Breast-cancer logistic regression | 0.9719 ± 0.0057 | 0.9369 ± 0.0128 | Malignant-class F1 |
| CIFAR-100 DenseNet-161 | 0.2103 | 0.4854 | 100-class macro F1 |

The [JSON files](results/) contain the full metrics, selected settings, package
versions, hardware, and timings. These numbers come from new runs, not the
original submissions.

## Setup

Install [uv](https://docs.astral.sh/uv/), clone this repository, and run commands
from its root. The project uses Python 3.12. `uv.lock` fixes the package versions.

```sh
uv sync --locked
```

The base environment runs both classical experiments and all notebooks without
PyTorch. The transfer command adds the `deep-learning` extra with CPU-only torch
and torchvision wheels. The recorded runs used an AMD Ryzen 7 7840U. Transfer
training used four torch threads.

For CUDA, create a separate environment with the
[official PyTorch installer](https://pytorch.org/get-started/locally/). This
repository contains no GPU results.

## k-NN from scratch

Does handwritten distance calculation, ranking, and voting reproduce scikit-learn
on the same data and hyperparameters? Both implementations produced identical
test predictions on all three recorded splits. The development sweep chooses `k`
and the distance metric before either model evaluates the test set. Each seed
took about one second on the recorded machine.

![Development scores across distance metrics and neighborhood sizes](results/knn_dev_sweep.png)

```sh
uv run --locked python -m mlshowcase.knn --output-dir results --seeds 42 43 44
```

[Study](docs/knn.md) · [Code](mlshowcase/knn.py) ·
[Notebook](notebooks/knn.ipynb) · [Tests](tests/test_knn.py) · [Results](results/knn.json)

## Logistic regression from scratch

Can the handwritten model avoid numerical overflow and evaluation leakage?
The scaler fits training data only. Development data selects the learning rate
and threshold together. None of the gradient-descent fits met the convergence
criterion before the 3,000-step limit. The LBFGS baseline selects its own
threshold and uses a different stopping rule. The score difference does not show
that one solver is generally better. Each seed took about 0.7 to 0.8 seconds.
Do not use this experiment for clinical decisions.

![Development threshold selection and held-out metrics for both models](results/logistic_comparison.png)

```sh
uv run --locked python -m mlshowcase.logistic --output-dir results --seeds 42 43 44
```

[Study](docs/logistic.md) · [Code](mlshowcase/logistic.py) ·
[Notebook](notebooks/logistic.ipynb) · [Tests](tests/test_logistic.py) · [Results](results/logistic.json)

## DenseNet transfer learning

Does a frozen ImageNet representation help within one epoch of training? Both
variants use 45,000 training images, 5,000 development images, and 10,000 official
test images. Transfer improved accuracy and macro F1, but its test cross-entropy
was higher, 6.9069 versus 3.1360. One seed cannot estimate variability. Neither
model trained to convergence, and matching epochs does not match compute.

![Test accuracy and macro F1 beside each model's trainable parameter count](results/transfer_comparison.png)

```sh
uv run --locked --extra deep-learning python -m mlshowcase.transfer --device cpu --epochs 1 --seeds 42 --data-dir data
```

The two runs took about 24 minutes on CPU. The first run downloads CIFAR-100 and
the ImageNet checkpoint. Git ignores downloaded data, weights, and selected model
checkpoints. The repository does not distribute them.

[Study](docs/transfer.md) · [Code](mlshowcase/transfer.py) ·
[Notebook](notebooks/transfer.ipynb) · [Tests](tests/test_transfer.py) · [Results](results/transfer.json)

## Notebooks and checks

```sh
uv run --locked jupyter lab
uv run --locked --extra deep-learning ruff check mlshowcase tests
uv run --locked --extra deep-learning pytest -q
```

Open a notebook under `notebooks/`. The notebooks demonstrate the Python APIs and
read the committed JSON and figures. Run All does not rerun the experiments or
download data or weights. The committed notebooks have no saved outputs. If your
kernel uses a different working directory, set `MLSHOWCASE_ROOT` to the checkout
root before starting Jupyter.

The full suite passed 59 tests. A separate smoke check trained and evaluated
DenseNet on a real image batch. All notebooks ran in an environment without
PyTorch. [CHANGELOG.md](CHANGELOG.md) records these checks and the corrections to
the coursework. The classical results describe three overlapping splits of small
datasets. They do not establish performance on other datasets.

## License and attribution

[MIT](LICENSE) covers the author's code and documentation. Datasets, pretrained
weights, and dependency software keep their upstream terms.
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) contains the UCI citations and
CC BY 4.0 attribution, CIFAR-100 provenance, and DenseNet and ImageNet references.
The public repository excludes instructor prompts, submitted reports, original
notebook outputs, and the experiment whose original CSV dataset is missing.
