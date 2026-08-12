# FPCA30 Component-Count Decision v0.1

Status: research-draft model-selection record.

Decision: retain the first 30 fold-local FPCA components for the `0.3.1-beta`
LR1 profile encoder. This is a controlled dimensionality-reduction choice, not
evidence that 30 biological tissue modes exist.

## Question

The preceding model used 100 radial-profile bins directly in LR1. Increasing
azimuthal integration to 256 bins increases q sampling density but gives LR1 256
correlated predictors for a small cohort. The experiment therefore asked:

```text
Can 256-bin profiles be compressed while preserving the patient-safe
sensitivity/specificity operating point of the raw100 model?
```

The component count was not selected to maximize train-on-all performance or
explained variance. Both criteria can favor a model that reconstructs profiles
or fits the training cohort without transferring to held-out patients.

The practical selection criterion applied to this retrospective comparison was
proximity to the raw100 sensitivity/specificity pair under the unchanged
high-sensitivity threshold policy. ROC AUC, explained variance, and train-all
fit were retained as supporting diagnostics rather than substituted for this
criterion.

## Encoder Definition

Each normalized profile sampled on the common q grid is approximated as

```text
x(q) = mean(q) + sum[z_j * phi_j(q)],  j = 1..K
```

where `phi_j(q)` is PCA loading j and `z_j` is its score. PCA on the uniformly
sampled q grid is used as a discrete approximation to functional PCA. LR1 uses
the standardized scores:

```text
logit(p_cancer) = beta_0 + sum[beta_j * standardized(z_j)],  j = 1..K
```

PCA is unsupervised: components maximize profile variance, not separation of
BENIGN and CANCER. A high cumulative explained-variance ratio therefore does
not establish adequate classification or calibration.

## Leakage Control

Evaluation used repeated patient-safe stratified 5-fold cross-validation x20,
giving 100 outer test folds. Within every fold, PCA mean, PCA loadings,
StandardScaler, LR1, LR2, and decision threshold were fitted using outer-train
patients only. All measurements and target cases from one patient remained in
one fold.

Fixed conditions:

```text
source H5 SHA256: d2d61e83850b282c3d2479ea436deed821c4488b96983252d294f3d56ee3f1f9
PyFAI: 2026.5.0
integration method: bbox / csr / cython
LR1 C: 0.1
LR2 C: 0.3
threshold policy: target sensitivity 0.95 on each outer-train partition
```

Only profile resolution and retained component count changed.

## Controlled Common Cohort

The direct comparison used identical measurement identities for raw100 and
FPCA256: 803 measurements, 161 patients, and 164 target-breast cases.

| Encoder | Explained variance | ROC AUC | Sensitivity | Specificity | Train-all specificity |
|---|---:|---:|---:|---:|---:|
| raw100 | n/a | 0.679 | 0.826 | 0.400 | 0.564 |
| FPCA10 | 0.956 | 0.670 | 0.891 | 0.277 | 0.340 |
| FPCA15 | 0.960 | 0.665 | 0.887 | 0.306 | 0.351 |
| FPCA20 | 0.963 | 0.654 | 0.862 | 0.343 | 0.415 |
| FPCA25 | 0.966 | 0.646 | 0.838 | 0.370 | 0.479 |
| FPCA30 | 0.968 | 0.656 | 0.832 | 0.404 | 0.606 |

Values are means across the 100 patient-safe outer folds, except the explicitly
in-sample train-all column.

## Why 10 Components Were Not Selected

FPCA10 reconstructs `95.6%` of total profile variance and gives a slightly
higher mean ROC AUC than FPCA30 (`0.670` versus `0.656`). It nevertheless moves
the fixed-threshold operating point: sensitivity increases from `0.826` to
`0.891`, whereas specificity decreases from `0.400` to `0.277`. Thus many more
BENIGN target cases are called CANCER.

This result separates ranking from operating behavior. ROC AUC measures ranking
over all possible thresholds. The product uses one threshold selected for high
sensitivity. FPCA10 can preserve ranking while changing score scale and LR2
calibration enough to reduce specificity at that threshold.

The small explained-variance difference between FPCA10 and FPCA30 (`95.6%` to
`96.8%`) is also informative. Profile reconstruction has nearly converged by 10
components, but specificity continues to change through 30 components. The
additional directions carry little total variance yet materially affect the
decision boundary after LR1 aggregation and LR2 refinement.

## Why 30 Components Were Selected

Thirty was the first tested count in the 10, 15, 20, 25, 30 grid that recovered
the raw100 thresholded operating point on the controlled common cohort:

```text
raw100: sensitivity 0.826, specificity 0.400
FPCA30: sensitivity 0.832, specificity 0.404
```

FPCA30 therefore reduced the LR1 input from 256 to 30 coordinates, an `88.3%`
reduction, while reproducing the raw100 sensitivity/specificity pair. It was
selected for this operating-point match, not because it maximized ROC AUC.
FPCA10 had higher ROC AUC but failed the operating-point criterion.

## Full-Cohort Confirmation

The current 256-bin product preprocessing produced 876 measurements, 163
patients, and 171 target-breast cases. The same component sweep gave:

| Encoder | ROC AUC | Sensitivity | Specificity | Train-all specificity |
|---|---:|---:|---:|---:|
| FPCA10 | 0.660 | 0.873 | 0.283 | 0.340 |
| FPCA15 | 0.647 | 0.867 | 0.273 | 0.320 |
| FPCA20 | 0.644 | 0.861 | 0.277 | 0.351 |
| FPCA25 | 0.645 | 0.863 | 0.285 | 0.340 |
| FPCA30 | 0.648 | 0.860 | 0.312 | 0.351 |

FPCA30 again produced the highest held-out specificity in this tested grid,
although the gain over FPCA10 was only `0.029` and specificity remained below
the raw100 historical result. The full-cohort run therefore supports 30 over 10
for the intended operating point, but does not establish a performance gain over
the preceding product model.

## Component-Stability Audit

The early PCA loadings are mathematically nested across FPCA4, FPCA7, FPCA15,
and FPCA30 when fitted on identical rows. PC1, PC2, PC4, PC6, and PC7 were the
most stable fold-local directions. However, a complete model using only these
five components gave ROC AUC `0.682`, sensitivity `0.905`, and specificity
`0.245` on the common cohort. Stable early components preserved ranking but did
not preserve the final operating point.

Late-component blocks were weak when isolated in LR1:

| LR1 representation | Patient-safe ROC AUC | Train-all ROC AUC | Fit gap |
|---|---:|---:|---:|
| PC1/2/4/6/7 | 0.610 | 0.657 | 0.047 |
| PC8-15 only | 0.505 | 0.598 | 0.092 |
| PC16-30 only | 0.530 | 0.671 | 0.141 |

The increasing fit gap indicates that late components contain substantial
cohort-specific or noise-sensitive structure. Nevertheless, removing them
changes LR1 score scale, LR2 calibration, and specificity. FPCA30 is therefore a
pragmatic operating-point compromise, not proof that every retained component
is transferable.

## Decision And Limitations

FPCA30 was selected because it was the tested compressed representation that
most closely reproduced the raw100 thresholded behavior on the matched cohort.
The decision is bounded by five limitations:

1. Component counts were tested on one small retrospective cohort.
2. Repeated folds overlap; fold-to-fold ranges are descriptive, not independent
   confidence intervals.
3. FPCA30 retains a train-all to held-out gap and late-component instability.
4. The full product cohort did not reproduce the matched-cohort specificity.
5. Component count was selected and evaluated using the same historical archive;
   the reported patient-safe folds do not constitute independent confirmation
   after model selection.

Independent data are required to determine whether 30 components improve
generalization. Until then, `0.3.1-beta` remains research decision support and
not an autonomous diagnosis.
