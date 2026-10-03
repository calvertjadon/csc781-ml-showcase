# Third-party notices and licensing scope

## MIT license

[LICENSE](LICENSE) grants MIT terms for the author's Python code, tests,
notebooks, documentation, and generated metrics and figures in `results/`.

That grant does not cover:

- Third-party dependency software.
- The UCI handwritten-digits and breast-cancer datasets or CIFAR-100.
- The ImageNet-pretrained DenseNet-161 weights.
- Model checkpoints that this code writes using those datasets or weights.

Each keeps its upstream terms. The sources below describe those terms. This
repository does not grant new rights to third-party material.

## Handwritten digits

The k-NN experiment loads `sklearn.datasets.load_digits()` without a download.
It uses 1,797 observations, 64 features, and 10 classes of 8-by-8 grayscale images.
These observations form the test partition of UCI's *Optical Recognition of
Handwritten Digits* dataset. UCI publishes 3,823 training observations and 1,797
test observations, 5,620 in total. This experiment uses only the test partition.

E. Alpaydin and C. Kaynak published the dataset in 1998,
[DOI 10.24432/C50P49](https://doi.org/10.24432/C50P49).
The [UCI source page](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits)
states that it uses
[Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/).

The repository distributes its calculated metrics and figures, not dataset files.

## Breast-cancer data

The logistic experiment loads `sklearn.datasets.load_breast_cancer()` without a
download. The bundled dataset has 569 observations and 30 features, with 357
benign cases and 212 malignant cases. Scikit-learn encodes malignant as 0 and
benign as 1. This experiment remaps malignant to positive class 1 before splitting.
The JSON records that remap under `protocol.provenance.label_remap`. Precision,
recall, and F1 refer to malignant cases.

William Wolberg, Olvi Mangasarian, Nick Street, and W. Street published *Breast
Cancer Wisconsin (Diagnostic)* in 1993,
[DOI 10.24432/C5DW2B](https://doi.org/10.24432/C5DW2B).
The [UCI source page](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)
states that it uses CC BY 4.0.

The repository does not distribute dataset files. This experiment does not
validate the model for clinical use.

## CIFAR-100

The transfer experiment loads `torchvision.datasets.CIFAR100()`. Alex Krizhevsky,
Vinod Nair, and Geoffrey Hinton created CIFAR-100. It has 100 classes, each with
500 official training images and 100 test images. The dataset totals 50,000
training images and 10,000 test images.

The [dataset page](https://www.cs.toronto.edu/~kriz/cifar.html) requests a citation
to Krizhevsky's 2009 technical report,
[*Learning Multiple Layers of Features from Tiny Images*](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf).
That page does not state a blanket license grant. This repository asserts none.
Torchvision downloads the dataset on the user's machine when the experiment runs.
The repository does not distribute it.

## DenseNet-161 and ImageNet weights

The pretrained model loads `DenseNet161_Weights.IMAGENET1K_V1` through
`torchvision.models.densenet161`. Huang, Liu, van der Maaten, and Weinberger
 describe the architecture in
[*Densely Connected Convolutional Networks*](https://arxiv.org/abs/1608.06993).
The [torchvision 0.21 model documentation](https://docs.pytorch.org/vision/0.21/models/generated/torchvision.models.densenet161.html)
identifies the weights as ImageNet-1K weights ported from LuaTorch. The `DEFAULT`
weights enum refers to the same checkpoint.

The documentation reports ImageNet-1K top-1 accuracy of 77.138% and top-5 accuracy
of 93.56%. Those are upstream checkpoint measurements, not this experiment's
results. The CIFAR-100 measurements are in `results/transfer.json`.

The standard weights transform resizes to 256 pixels and crops to 224. It uses
bilinear interpolation and ImageNet means and standard deviations. This
experiment resizes to 74 pixels and crops to 64, with the same normalization and
interpolation. The JSON records the resolution change.

Torchvision downloads the
[checkpoint](https://download.pytorch.org/models/densenet161-8d451a50.pth) when the
pretrained experiment runs. The repository does not distribute the weights or
grant a license for them. Use remains subject to their provider's terms.
Torchvision's software license is separate from the weights' terms.

## Dependency software

`pyproject.toml` lists the dependencies. `uv sync --locked` installs them from
upstream distributions. Each package keeps its own license. Follow the project
links for those terms.

| Package | Use | Upstream project |
| --- | --- | --- |
| jupyterlab | Notebook editor | <https://jupyter.org> |
| matplotlib | Figures | <https://matplotlib.org> |
| nbclient | Notebook execution | <https://jupyter.org> |
| nbformat | Notebook files | <https://jupyter.org> |
| numpy | Array operations | <https://numpy.org> |
| scikit-learn | Datasets, reference estimators, metrics, and splitting | <https://scikit-learn.org> |
| scipy | Numerical calculations | <https://scipy.org> |
| torch | DenseNet training, through the `deep-learning` extra | <https://pytorch.org> |
| torchvision | DenseNet, weights, and CIFAR-100 loading, through the `deep-learning` extra | <https://pytorch.org/vision/> |
| pytest | Tests, through the `dev` group | <https://docs.pytest.org> |
| ruff | Lint checks, through the `dev` group | <https://docs.astral.sh/ruff> |

The lockfile selects CPU builds of torch and torchvision from
<https://download.pytorch.org/whl/cpu>. This repository includes no dependency
source code.

## Excluded course materials

The repository contains no original assignment briefs, submitted reports or
scores, saved notebook outputs, or links to the private archive. The
[studies](docs/) and [changelog](CHANGELOG.md) describe the methodological
corrections without publishing those materials.
