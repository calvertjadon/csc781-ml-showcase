# Third-party notices and licensing scope

This repository is a public showcase of machine-learning experiments. This file
records which parts of it the author licenses, which parts carry other people's
terms, and where those terms come from. It is a scope and attribution notice, not
a replacement for any upstream license text.

## 1. What the MIT license in `LICENSE` covers

The MIT license in [`LICENSE`](LICENSE) applies to the original material created
for this repository:

- the Python package (`mlshowcase/*.py`),
- the notebooks (`notebooks/*.ipynb`),
- the test suite (`tests/*.py`),
- the documentation (`README.md`, `docs/*.md`, `CHANGELOG.md`, this file), and
- the evidence these experiments produced: `results/*.json` metrics and
  `results/*.png` figures.

The same MIT license does **not** cover, and no MIT grant is made here for:

- the third-party software packages this project depends on,
- the datasets the experiments read (scikit-learn's bundled copies of the UCI
  handwritten-digits and breast-cancer datasets, and CIFAR-100),
- the ImageNet-pretrained DenseNet-161 weights, or
- model checkpoints written by this code, which are derived from those datasets
  and weights.

Each of those remains under its own upstream terms, described below. Where this
file links to an upstream source, that link is the authoritative statement of the
relevant terms; nothing in this repository re-licenses third-party material.

## 2. Datasets

### 2.1 Handwritten digits (`sklearn.datasets.load_digits`)

- Used by the k-nearest-neighbors study (`mlshowcase/knn.py`,
  `notebooks/knn.ipynb`, `results/knn.json`).
- Bundled copy: 1,797 observations, 64 features, 10 classes of 8x8 grayscale
  images, read from scikit-learn with no network access.
- Relationship to the UCI source: 1,797 observations is the test partition of the
  UCI "Optical Recognition of Handwritten Digits" dataset. The UCI record lists
  5,620 observations in total across the files it publishes (3,823 training plus
  1,797 test), so this showcase uses the 1,797-observation test split, not the
  full 5,620-observation dataset.
- UCI dataset: "Optical Recognition of Handwritten Digits", donated 30 June 1998.
  Creators: E. Alpaydin and C. Kaynak. Citation: Alpaydin, E. & Kaynak, C. (1998).
  *Optical Recognition of Handwritten Digits* [Dataset]. UCI Machine Learning
  Repository. <https://doi.org/10.24432/C50P49>
- UCI license: the UCI dataset page states the dataset is licensed under a
  Creative Commons Attribution 4.0 International (CC BY 4.0) license,
  <https://creativecommons.org/licenses/by/4.0/>
- Source page:
  <https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits>
- Redistribution: the dataset bytes are not copied into this repository. The
  figures and metrics under `results/` are this project's own outputs.

### 2.2 Breast Cancer Wisconsin (Diagnostic) (`sklearn.datasets.load_breast_cancer`)

- Used by the logistic-regression study (`mlshowcase/logistic.py`,
  `notebooks/logistic.ipynb`, `results/logistic.json`).
- Bundled copy: 569 observations and 30 features, read from scikit-learn with no
  network access; the bundled class counts are 357 benign and 212 malignant.
- Label remap: scikit-learn's bundled copy encodes malignant as 0 and benign as 1.
  This showcase deliberately remaps malignant to the positive class 1 and benign
  to 0 before splitting, and records that remap in `results/logistic.json`
  (`protocol.provenance.label_remap`). Malignant is therefore the positive class
  in every precision, recall, and F1 number reported here.
- Creators: William Wolberg, Olvi Mangasarian, Nick Street, and W. Street.
  Citation: Wolberg, W., Mangasarian, O., Street, N., & Street, W. (1993).
  *Breast Cancer Wisconsin (Diagnostic)* [Dataset]. UCI Machine Learning
  Repository. <https://doi.org/10.24432/C5DW2B>
- UCI license: the UCI dataset page states the dataset is licensed under CC BY 4.0,
  <https://creativecommons.org/licenses/by/4.0/>
- Source page:
  <https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic>
- The dataset is not distributed here, and the experiment is an educational
  methods exercise with no clinical use (see `CHANGELOG.md`).

### 2.3 CIFAR-100 (`torchvision.datasets.CIFAR100`)

- Used by the transfer-learning study (`mlshowcase/transfer.py`,
  `notebooks/transfer.ipynb`, `results/transfer.json`).
- Created by Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton. The dataset has 100
  classes with 600 images per class: 500 training and 100 test images per class,
  i.e. 50,000 official training images and 10,000 official test images.
- Requested citation: Krizhevsky, A. (2009). *Learning Multiple Layers of Features
  from Tiny Images* (technical report).
  <https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf>
- Primary page: <https://www.cs.toronto.edu/~kriz/cifar.html>
- Terms: the primary page asks that users of the dataset cite the technical report;
  it does not state a blanket license grant for the dataset, and no such grant is
  asserted here. This repository does not redistribute CIFAR-100: torchvision
  downloads it on the user's machine at run time, under the terms published with
  the dataset.

## 3. Pretrained weights: DenseNet-161 ImageNet-1K V1

- Used by the `pretrained` variant of the transfer study, loaded through
  `torchvision.models.densenet161(weights=DenseNet161_Weights.IMAGENET1K_V1)`.
- Architecture reference: *Densely Connected Convolutional Networks* (Huang, Liu,
  van der Maaten, Weinberger), <https://arxiv.org/abs/1608.06993>
- torchvision 0.21 model and weights reference:
  <https://docs.pytorch.org/vision/0.21/models/generated/torchvision.models.densenet161.html>
  the weights enum used is `DenseNet161_Weights.IMAGENET1K_V1`, which is also
  reachable as `DenseNet161_Weights.DEFAULT`.
- Provenance: these weights were trained on ImageNet-1K (1,000 classes) and the
  torchvision page documents them as ported from LuaTorch.
- Upstream metadata on that page: 77.138% top-1 and 93.56% top-5 accuracy on
  ImageNet-1K. Those figures are upstream metadata about the checkpoint, not
  results measured by this showcase; the only ImageNet-derived numbers this
  repository reports are the CIFAR-100 scores recorded in `results/transfer.json`.
- Standard inference recipe of those weights: resize 256, central crop 224,
  bilinear interpolation, ImageNet mean/std normalization. The transfer experiment
  deliberately overrides the resolution to resize 74 / crop 64 and records that
  override in `results/transfer.json`; normalization and interpolation are
  inherited from the weights recipe.
- Terms: no license for the weights is asserted here. This repository does not
  bundle them: torchvision downloads them from
  <https://download.pytorch.org/models/densenet161-8d451a50.pth> when a user runs
  the pretrained variant, and use of the weights is subject to the terms published
  by their provider. torchvision itself is a separate third-party package with its
  own license (see section 4).

## 4. Third-party software

Declared in `pyproject.toml` and installed by `uv sync --locked` from upstream
distributions. Each package keeps its own license, and the MIT license in
`LICENSE` does not cover any of them; no upstream license is restated here - follow
the project links for the current terms.

| Package | Role | Upstream project |
| --- | --- | --- |
| jupyterlab | notebook environment | <https://jupyter.org> |
| matplotlib | figures | <https://matplotlib.org> |
| nbclient | notebook execution tests | <https://jupyter.org> |
| nbformat | notebook reading | <https://jupyter.org> |
| numpy | array maths | <https://numpy.org> |
| scikit-learn | bundled datasets, baseline estimators, metrics | <https://scikit-learn.org> |
| scipy | numerical support | <https://scipy.org> |
| torch (optional `deep-learning` extra) | DenseNet-161 training | <https://pytorch.org> |
| torchvision (optional `deep-learning` extra) | DenseNet-161 model and weights, CIFAR-100 loader | <https://pytorch.org/vision/> |
| pytest (`dev` group) | test runner | <https://docs.pytest.org> |
| ruff (`dev` group) | linter | <https://docs.astral.sh/ruff> |

The CPU builds of torch and torchvision are resolved from the PyTorch package index
at <https://download.pytorch.org/whl/cpu>, as recorded in `pyproject.toml` and
`uv.lock`. No dependency source code is vendored or copied into this repository.

## 5. Material that is not part of this repository

The original course materials behind these studies - assignment briefs, submitted
reports and their scores, and historical notebook outputs - are not included in
this repository, are not reproduced anywhere in it, and are not linked from it.
This repository publishes only the author's rebuilt protocols, freshly measured
evidence, and the documentation written for it. Where a rebuild corrects a
methodological problem in the original work, the correction is described in
`CHANGELOG.md` and in the corresponding notebook and study document.
