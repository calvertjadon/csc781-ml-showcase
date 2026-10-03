# DenseNet-161 transfer learning on CIFAR-100

## Question and implementation

Does an ImageNet-pretrained DenseNet-161 with a frozen backbone outperform one
trained from random initialization within the same one-epoch budget? Both models
use the same CIFAR-100 split, preprocessing, and seed. The
[recorded results](../results/transfer.json) come from new runs, not original
submission scores.

Torchvision supplies DenseNet-161, the `DenseNet161_Weights.IMAGENET1K_V1`
checkpoint, and its preprocessing transforms. Scikit-learn supplies macro F1.
I wrote the model setup, backbone freezing, seeded data loading, training loop,
checkpoint selection, evaluation, and result files in
[mlshowcase/transfer.py](../mlshowcase/transfer.py).

[Notebook](../notebooks/transfer.ipynb) · [Tests](../tests/test_transfer.py) ·
[Results](../results/transfer.json)

## Corrections to the coursework

- The code replaces the deprecated `pretrained=True/False` arguments with
  `weights=None` for scratch training and `DenseNet161_Weights.IMAGENET1K_V1`
  for transfer. It uses the transforms supplied with those weights.
- The original resized images to 224 pixels but omitted ImageNet normalization.
  Both models now use the weights' normalization, bilinear interpolation, and
  antialiasing.
- Setting `requires_grad = False` does not stop `model.train()` from updating
  BatchNorm buffers. After each training-mode call, this code returns `features`
  to `eval()` and gives the optimizer only trainable parameters.
- The original trained scratch for 20 epochs and transfer for 5 while printing
  test accuracy. This experiment gives each model one epoch, selects a checkpoint
  on development data, and evaluates the official test split once.
- The code seeds Python, NumPy, and torch. Both models seed their DataLoader
  shuffle generator with 42, giving them the same shuffle order.

The tests check frozen BatchNorm buffers, classifier updates, sample-weighted
losses, and macro F1 over all 100 classes with `zero_division=0`.

## Protocol

`torchvision.datasets.CIFAR100` supplies 50,000 official training images and
10,000 test images across 100 classes. A stratified split with seed 42 reserves
5,000 training images for development. This leaves 45,000 for training, with
450 training images and 50 development images per class. The split stays fixed
across model seeds. The official test split remains separate until evaluation.

Both models use
`DenseNet161_Weights.IMAGENET1K_V1.transforms(crop_size=64, resize_size=74)`.
The transform uses ImageNet means `[0.485, 0.456, 0.406]` and standard deviations
`[0.229, 0.224, 0.225]`. Bilinear interpolation and antialiasing come from the
weights' transforms. The 64-pixel crop and 74-pixel resize replace the standard
224-pixel crop and 256-pixel resize to reduce CPU work. Neither model uses data
augmentation.

Scratch training updates all 26,692,900 parameters. Transfer freezes 26,472,000
backbone parameters and their BatchNorm buffers. Its new 100-class classifier
has 220,900 trainable parameters.

Both models use SGD with learning rate 0.01, momentum 0.9, batch size 32, and
cross-entropy loss. Each trains for one epoch on CPU with four torch threads and
model seed 42.

Checkpoint selection maximizes development macro F1, with ties choosing the
earliest epoch. The code reloads that checkpoint before evaluating the official
test split. One epoch provides only one candidate checkpoint, so the selection
rule cannot distinguish checkpoints in these runs.

## Results

| Model | Trainable parameters | Total parameters | Development macro F1 | Test loss | Test accuracy | Test macro F1 | Training seconds | Total seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Scratch | 26,692,900 | 26,692,900 | 0.2070 | 3.1360 | 0.2328 | 0.2103 | 959.25 | 1054.55 |
| Pretrained | 220,900 | 26,692,900 | 0.4904 | 6.9069 | 0.4884 | 0.4854 | 291.61 | 389.26 |

Transfer improved accuracy and macro F1, but its test cross-entropy was higher,
6.9069 versus 3.1360. Development loss showed the same ordering, 6.6121 versus
3.1647. The experiment did not measure calibration or test a cause for this
difference.

The two runs took about 24 minutes on an AMD Ryzen 7 7840U with four torch
threads. The environment used torch 2.6.0+cpu and torchvision 0.21.0+cpu.

## Figures

[transfer_learning_curves.png](../results/transfer_learning_curves.png) plots
development macro F1 and cross-entropy at the recorded epoch. Each model has one
point, so the figure shows no training trajectory.
[transfer_comparison.png](../results/transfer_comparison.png) plots test accuracy
and macro F1 without error bars, beside the trainable parameter counts on a log
scale.

## Limitations

One seed does not estimate variability. The JSON's standard deviation of 0.0
reflects one observation, not evidence that repeated runs would be identical.
One epoch limits compute but does not establish convergence for either model.
The 64-pixel inputs also differ from the pretrained weights' standard evaluation
resolution.

The models update different numbers of parameters and take different amounts of
time. The comparison tests initialization and freezing together, not
initialization alone under equal parameter counts or compute. Its results apply
to this CIFAR-100 and DenseNet-161 protocol only.

## Reproduce

From the repository root:

```sh
uv run --locked --extra deep-learning python -m mlshowcase.transfer --device cpu --epochs 1 --seeds 42 --data-dir data
```

The first run downloads CIFAR-100 into `data/` and the ImageNet checkpoint from
`download.pytorch.org`. The command rewrites `results/transfer.json` and both
figures. It also writes checkpoints under `results/checkpoints/`. Git ignores
the data and checkpoints. The repository does not distribute them.

## Attribution

CIFAR-100's source is Krizhevsky's 2009 report, *Learning Multiple Layers of
Features from Tiny Images*. The [dataset page](https://www.cs.toronto.edu/~kriz/cifar.html)
links to that report. Huang and colleagues describe DenseNet in their 2017 paper,
[*Densely Connected Convolutional Networks*](https://arxiv.org/abs/1608.06993).

[THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md) contains the dataset, weights,
and software sources and terms. MIT covers this repository's code and
documentation, not third-party datasets or weights.
