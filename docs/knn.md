# Handwritten k-nearest neighbors on digits

## Question and implementation

Does a NumPy k-NN classifier reproduce `sklearn.neighbors.KNeighborsClassifier`
on the same data and hyperparameters? Which distance and neighborhood size does
selection on development data choose for the bundled 8-by-8 digit images?

I wrote the distance calculations, neighbor rankings, majority vote,
`KNNClassifier`, and experiment code in [mlshowcase/knn.py](../mlshowcase/knn.py).
The code computes one ranking per distance metric and reuses it for each `k`.
Scikit-learn supplies the dataset, split function, metrics, and reference
classifier. Matplotlib draws the figures.

[Notebook](../notebooks/knn.ipynb) · [Tests](../tests/test_knn.py) ·
[Results](../results/knn.json) · [Split helper](../mlshowcase/common.py)

## Corrections to the coursework

The original Euclidean and Manhattan loops omitted the final feature. The new
calculations use all 64 features. The tests check examples whose distances change
from 13 to 5 and from 19 to 7 if the final feature is dropped.

The original combined labels with pixel values, converting integer labels to
floats. This implementation keeps them in separate arrays. It also runs the
hyperparameter sweep that the original notebook commented out and replaces the
fixed Manhattan `k = 5` choice with development-set selection. Seeds and
stratification replace the original unseeded splits.

## Protocol

- `load_digits()` supplies 1,797 samples with 64 features and 10 classes without
  a download. This is the UCI dataset's test partition. The code uses raw pixel
  intensities without scaling or feature selection.
- Seeds 42, 43, and 44 each produce a stratified split with 1,078 training rows,
  359 development rows, and 360 test rows. Development data selects the
  configuration. Each model evaluates the test split after selection.
- The sweep tests seven values of `k`, 1, 3, 5, 7, 9, 11, and 15. Each uses
  Euclidean, Manhattan, or Minkowski distance with `p = 1.5`. That gives 21
  candidates per seed.
- Selection maximizes unrounded development macro F1. Ties choose the smallest
  `k`, then prefer Euclidean, Manhattan, and Minkowski in that order.
- A tied vote chooses the smallest class label. Equal distances retain training
  row order. That order can affect which rows enter the first `k` neighbors.
- The reference `KNeighborsClassifier(algorithm="brute")` uses the same training
  rows and selected configuration. This checks the implementation rather than
  comparing independently tuned classifiers.

## Results

| Seed | Selected configuration | Development macro F1 | Test accuracy | Test macro F1 |
| --- | --- | --- | --- | --- |
| 42 | Euclidean, k=1 | 0.9888 | 0.9750 | 0.9750 |
| 43 | Euclidean, k=1 | 0.9915 | 0.9806 | 0.9805 |
| 44 | Minkowski, p=1.5, k=3 | 0.9776 | 0.9861 | 0.9860 |

The mean test macro F1 is 0.9805 with a population standard deviation of 0.0045.
Mean accuracy is 0.9806 with the same standard deviation. Scikit-learn produced
identical predictions on all 360 test rows for each seed. The JSON records
`predictions_match = true` for every run.

Development scores generally fell as `k` increased. On seed 43, Euclidean and
Minkowski with `k = 1` tied at 0.9915 development macro F1. The metric preference
selected Euclidean.

The runs used CPython 3.12.13 on Linux x86_64 and an AMD Ryzen 7 7840U. Each seed
took 0.69, 0.65, or 1.11 seconds for selection, fitting, and evaluation of both
implementations. The JSON contains package versions. Torch appears in that
record because the environment had it installed, but k-NN does not use it.

[knn_dev_sweep.png](../results/knn_dev_sweep.png) plots development scores against
`k`, with individual seeds, means, selected points, and zoomed axes.
[knn_baseline.png](../results/knn_baseline.png) plots held-out scores on axes from
0 to 1.05 and the two matching confusion matrices for seed 42.

## Interpretation and limitations

Agreement on these splits does not prove agreement for every input or show that
one implementation is better. Both models use the configuration selected by the
NumPy implementation. The recorded results do not say whether a tied distance at
the neighborhood boundary affected any prediction.

The selected configurations differ across seeds, and test macro F1 ranges from
0.9750 to 0.9860. Three overlapping splits of one small dataset do not establish
a general preference for a distance metric or neighborhood size. Their standard
deviation describes those runs only. It is not a confidence interval.

The implementation computes distances between every query and training row for
each metric. It uses neither an index nor approximate search. No experiment
varies preprocessing, and the results apply only to this dataset and protocol.

## Reproduce

From the repository root:

```sh
uv run --locked python -m mlshowcase.knn --output-dir results --seeds 42 43 44
uv run --locked pytest tests/test_knn.py
```

The experiment command rewrites the JSON and both figures. The notebook reads
those files without rerunning the experiment. Use `notebooks/` as its working
directory, or set `MLSHOWCASE_ROOT` to the checkout root.

## Attribution

E. Alpaydin and C. Kaynak published *Optical Recognition of Handwritten Digits*
in 1998. UCI distributes it under CC BY 4.0,
[DOI 10.24432/C50P49](https://doi.org/10.24432/C50P49). Scikit-learn bundles the
1,797-observation test partition used here. This repository does not distribute
the dataset or original coursework materials. See
[THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md) for the sources and terms.
