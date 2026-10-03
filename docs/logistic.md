# Logistic regression with stable numerics

## Question and implementation

Can a handwritten logistic model avoid numerical overflow and evaluation
leakage? This experiment compares NumPy full-batch gradient descent with
scikit-learn's unregularized LBFGS estimator on the same data.

I wrote the sigmoid, mean cross-entropy from logits, analytic gradient,
optimizer, training-only scaler, and experiment code. NumPy supplies array and
math operations. Scikit-learn supplies the dataset, stratified splitting,
metrics, and LBFGS estimator.

[Code](../mlshowcase/logistic.py) · [Notebook](../notebooks/logistic.ipynb) ·
[Tests](../tests/test_logistic.py) · [Results](../results/logistic.json)

## Corrections to the coursework

- The sigmoid uses separate branches for positive and negative scores. Both
  exponentiate only non-positive values. Tests cover scores of ±1000 without
  suppressing overflow warnings.
- The objective computes cross-entropy from logits with `np.logaddexp`.
  Tests compare the analytic gradient with finite differences. Each optimizer
  state uses one matrix-vector product for both loss and gradient.
- The scaler fits population means and standard deviations on training data
  once, then applies them to development and test data. A constant training
  column gets scale 1 to avoid division by zero. The original normalized each
  split independently, producing inconsistent transformations.
- The experiment retains the parameters for the selected learning rate instead
  of evaluating the last model fitted by a tuning loop.
- The optimizer returns the parameter state for the last recorded history entry.
  It stops at the gradient tolerance or step limit. A non-finite state marks the
  fit as diverged and excludes it from selection.

The results come from new runs. They do not reuse original notebook outputs or
submission scores.

## Protocol

- The bundled Wisconsin Diagnostic Breast Cancer dataset has 569 rows and 30
  features. Scikit-learn encodes malignant as 0. This experiment remaps malignant
  to positive class 1 and benign to 0, so precision, recall, and F1 describe
  malignancy.
- Seeds 42, 43, and 44 each produce a stratified split with 341 training rows,
  114 development rows, and 114 test rows through the shared
  [split helper](../mlshowcase/common.py).
- Full-batch gradient descent starts all parameters, including the intercept,
  at zero. It tests learning rates 0.01, 0.05, 0.1, and 0.5. Each fit runs for at
  most 3,000 steps and stops earlier if `max(abs(gradient)) < 1e-6`.
- Selection maximizes unrounded malignant-class F1 on development data across
  the four rates and 17 thresholds from 0.10 to 0.90 in steps of 0.05. Ties choose
  the smaller threshold, then the smaller rate. Only the selected model
  evaluates the test split.
- The baseline uses `LogisticRegression(C=np.inf, solver="lbfgs", max_iter=10000,
  tol=1e-10)` with an intercept and the same scaled training data. It selects its
  own threshold from the same development grid. Neither model uses test data
  to choose its settings.

## Results

Each entry shows the mean and population standard deviation over the three seeds.
The JSON contains the per-seed scores, confusion matrices, histories, and timings.

| Held-out metric | NumPy gradient descent | Scikit-learn LBFGS |
| --- | --- | --- |
| Accuracy | 0.9795 ± 0.0041 | 0.9532 ± 0.0109 |
| Malignant precision | 0.9919 ± 0.0115 | 0.9464 ± 0.0388 |
| Malignant recall | 0.9528 ± 0.0005 | 0.9291 ± 0.0195 |
| Malignant F1 | 0.9719 ± 0.0057 | 0.9369 ± 0.0128 |
| ROC AUC | 0.9981 ± 0.0004 | 0.9796 ± 0.0121 |

Seed 42 selected rate 0.01 and threshold 0.40. Seed 43 selected 0.01 and 0.50.
Seed 44 selected 0.05 and 0.35. The baseline selected threshold 0.10 for every
seed because its development F1 was flat across the grid. The first maximum won.

All twelve gradient-descent fits reached 3,000 steps without meeting the gradient
tolerance. None diverged. Higher rates lowered training loss further, but
selection chose the lower rates. LBFGS recorded 50, 47, and 39 iterations under
its own stopping rule.

Each seed took about 0.72 to 0.76 seconds on an AMD Ryzen 7 7840U. Its four
gradient-descent fits took about 0.61 to 0.63 seconds. These timings use different
stopping rules and do not provide a matched solver benchmark.

![Training loss by completed gradient-descent steps](../results/logistic_loss.png)

![Development threshold selection and held-out metrics](../results/logistic_comparison.png)

## Interpretation and limitations

The selected gradient-descent models scored better here, but the solvers use
different stopping rules and do not reach the same optimization endpoint.
Stopping gradient descent after a fixed number of steps may act as implicit
regularization. This experiment does not test that explanation or establish that
gradient descent is generally better than LBFGS.

Each test split has only 114 samples. Every gradient-descent confusion matrix has
two false negatives and zero or one false positive. A few cases change F1.
The three splits overlap, so they are not independent replications. Their
standard deviation is not a confidence interval. This experiment does not
measure calibration or validate the models on external data. Do not use it for
diagnosis or treatment.

## Reproduce

From the repository root:

```sh
uv run --locked python -m mlshowcase.logistic --output-dir results --seeds 42 43 44
```

The command regenerates the JSON and both figures. The notebook reads those
files and demonstrates functions on small synthetic inputs. It does not rerun
the recorded experiment. The JSON records the environment versions.

## Attribution

Wolberg, Mangasarian, Street, and Street published *Breast Cancer Wisconsin
(Diagnostic)* in 1993. UCI distributes it under CC BY 4.0,
[DOI 10.24432/C5DW2B](https://doi.org/10.24432/C5DW2B). Scikit-learn supplies the
bundled copy. This repository does not distribute the dataset. See
[THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md) for the sources and terms.
