# Handwritten k-nearest neighbours on digits: seeded selection and agreement with scikit-learn

**Question.** Can a from-scratch NumPy k-nearest-neighbours classifier — distances, exhaustive
ranking, majority vote — reproduce `sklearn.neighbors.KNeighborsClassifier` under one identical
protocol, and which (distance, k) configuration does development-split selection choose on the
bundled 8×8 digit images?

This rebuild of the CSC781 Module 2 Assignment 2 k-NN exercise keeps the assignment's goal and
corrects its methodology: all 64 features are used, labels stay integer arrays separate from the
pixels, the sweep executes instead of a commented-out loop fixed at Manhattan `k = 5`, and splits
are seeded and stratified. It is not a new architecture.

**Contribution.** Handwritten: three distances over all features, one exhaustive ranking per
metric reused across every `k`, majority vote with both tie rules, `KNNClassifier`, and the
protocol/JSON/figure harness. Library: `load_digits`, `train_test_split` (inside
`stratified_split`), accuracy / macro F1 / confusion matrix, matplotlib, and the
`KNeighborsClassifier(algorithm="brute")` reference.

**Evidence and code:**
[`mlshowcase/knn.py`](../mlshowcase/knn.py), [`mlshowcase/common.py`](../mlshowcase/common.py),
[`results/knn.json`](../results/knn.json), [`results/knn_dev_sweep.png`](../results/knn_dev_sweep.png),
[`results/knn_baseline.png`](../results/knn_baseline.png),
[`notebooks/knn.ipynb`](../notebooks/knn.ipynb), [`tests/test_knn.py`](../tests/test_knn.py).
Tests pin the all-feature reductions (worked examples 13→5 and 19→7 when the last pixel is
dropped), label dtype, `p >= 1` / `k` validation, ranking reuse, and both tie rules.

## Protocol

- **Data and splits.** `load_digits()`: 1797 samples × 64 features, 10 classes, no network; raw
  pixel intensities with no scaling or feature selection. The bundled copy is the
  1,797-observation UCI *test* partition, not redistributed. Stratified 60/20/20 splits are
  redrawn per seed 42/43/44, each giving 1078 / 359 / 360 rows; selection uses dev only and test is
  scored once afterwards.
- **Sweep.** `k ∈ {1, 3, 5, 7, 9, 11, 15}` × {Euclidean, Manhattan, Minkowski `p = 1.5`} = 21
  candidates per seed.
- **Selection.** Maximum unrounded dev macro F1; ties go to the smallest `k`, then to metric order
  Euclidean → Manhattan → Minkowski.
- **Tie rules, stated separately.** Vote-count ties pick the smallest class label. Equal distances
  keep training-row order, so which equally distant rows fall inside the first `k` is
  row-order-determined; that boundary membership, not the vote, can change the prediction.
- **Baseline.** `KNeighborsClassifier(algorithm="brute")` on the same rows with the identical
  selected configuration; an implementation check, not a tuned competitor.
- **Runtime.** CPython 3.12.13, Linux x86_64, Ryzen 7 7840U; versions in the JSON (torch listed,
  unused). Recorded stages — selection with the ranking passes, fit, evaluation, baseline — total
  0.69 / 0.65 / 1.11 s per seed: about one second per seed.

## Results

| Seed | Selected on dev | Dev macro F1 | Test accuracy | Test macro F1 |
| --- | --- | --- | --- | --- |
| 42 | euclidean, k=1 | 0.9888 | 0.9750 | 0.9750 |
| 43 | euclidean, k=1 | 0.9915 | 0.9806 | 0.9805 |
| 44 | minkowski (p=1.5), k=3 | 0.9776 | 0.9861 | 0.9860 |

Held-out macro F1 is 0.9805 ± 0.0045, accuracy 0.9806 ± 0.0045 (population SD, three seeds);
sklearn's rows are identical field for field: `predictions_match` is true for every recorded seed
(3 × 360 held-out rows). Dev macro F1 declines with `k` on all three distances. Seed 43 exercised
the tie-break: Euclidean `k=1` and Minkowski `p=1.5` `k=1` tie at 0.9915 dev macro F1, and metric
order chose Euclidean.

**Figures.** [`knn_dev_sweep.png`](../results/knn_dev_sweep.png) plots dev macro F1 and accuracy
against `k` per distance (per-seed lines, metric means, starred selections, zoomed axes);
[`knn_baseline.png`](../results/knn_baseline.png) shows per-seed held-out bars for both models on
the full 0–1.05 scale plus seed 42's confusion matrices, equal because predictions match.

## Interpretation and failures

- **Agreement is implementation evidence, not superiority.** The baseline receives the
  handwritten-selected configuration, so this is not an independent comparison and shows neither
  implementation better.
- **Three seeds do not generalize every boundary tie.** Agreement is empirical for these splits:
  equal-distance neighbours are ordered by training-row order, so a `k`-boundary tie can change
  which labels vote, and the recorded runs may not contain such a case.
- **Small-`k` preference is not a general finding.** Selections differ across seeds and held-out
  macro F1 spans 0.9750–0.9860. No preprocessing is swept, and the
  exhaustive ranking costs one query–train pass per metric — reproducible, not indexed or
  approximate; timings are provenance.

## Limitations

- Three overlapping splits of one 1797-row dataset; mean ± SD is descriptive only, no confidence
  interval, and results apply to this dataset and protocol alone.
- Agreement covers the recorded splits and pinned test behaviours, not every input; whether
  equal-distance ordering ever changed a recorded prediction is not recorded.

## Reproduction

```console
uv run --locked python -m mlshowcase.knn --output-dir results --seeds 42 43 44
```

Rewrites `results/knn.json` and both PNGs. Focused check:
`uv run --locked pytest tests/test_knn.py`. The notebook loads the stored JSON and figures and
never retrains (run with `notebooks/` as cwd, or set `MLSHOWCASE_ROOT`).

## Data, rights, and citation

- **Dataset.** *Optical Recognition of Handwritten Digits*, E. Alpaydin and C. Kaynak (1998), UCI
  Machine Learning Repository, CC BY 4.0, DOI [10.24432/C50P49](https://doi.org/10.24432/C50P49);
  loaded via `sklearn.datasets.load_digits()` (bundled 1,797-observation UCI test partition),
  bytes not redistributed — full attribution in
  [`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md).
- scikit-learn 1.9.1 supplies the reference estimator, split primitive and metrics; original
  coursework materials are not included, reproduced or linked.
