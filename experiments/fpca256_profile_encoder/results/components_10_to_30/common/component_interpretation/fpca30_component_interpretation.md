# FPCA30 component interpretation

Research-only analysis of the matched common cohort: 161 patients, 164 target cases, and 449 biopsy-side measurement profiles.

## What is measured

- `explained_variance_ratio`: share of total profile variation represented by the component. It is not a cancer-specific measure.
- `fold_mean_abs_lr1_coefficient`: mean absolute LR1 coefficient for a one-SD component-score change over the 100 patient-safe outer-fold fits.
- `fold_basis_abs_cosine_mean`: similarity of a fold-local PCA basis vector to the train-on-all vector after allowing the arbitrary PCA sign to reverse.
- PCA component signs are arbitrary. Positive/negative profile directions are therefore meaningful only after alignment to the train-on-all basis.

## Most LR1-active components

| Rank | PC | Variance | Mean |LR1 coefficient| | Basis similarity | Univariate AUC |
|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 58.914% | 0.302 | 0.999 | 0.587 |
| 2 | 4 | 2.018% | 0.236 | 0.996 | 0.576 |
| 3 | 6 | 0.182% | 0.194 | 0.977 | 0.544 |
| 4 | 7 | 0.120% | 0.164 | 0.953 | 0.542 |
| 5 | 2 | 29.320% | 0.153 | 0.998 | 0.540 |
| 6 | 29 | 0.053% | 0.133 | 0.210 | 0.567 |
| 7 | 28 | 0.055% | 0.131 | 0.192 | 0.503 |
| 8 | 10 | 0.084% | 0.129 | 0.442 | 0.566 |
| 9 | 24 | 0.058% | 0.119 | 0.271 | 0.514 |
| 10 | 25 | 0.058% | 0.117 | 0.247 | 0.508 |

## Interpretation

- PC1 and PC2 represent the dominant broad profile variation. Together they explain most of the profile variance, but their individual class separation is modest.
- PC1, PC2, PC4, PC6, and PC7 are the stable model-active group: each has mean fold-basis similarity above 0.95. PC1 describes a broad transfer between the q approximately 13.4 peak region and q above approximately 17; PC2 is a broader q approximately 14 contrast. PC4 and PC6 include low-q versus mid-q contrast. These are profile-shape descriptions, not molecular assignments.
- Several low-variance components receive substantial train-on-all LR1 coefficients. PC10, PC26, and PC29 have low fold-basis similarity and must be treated as unstable candidate patterns, not established biological features.
- `fpca30_active_components.png` displays the profile perturbation for a one-SD change in each selected component. The chart is descriptive; it does not identify a molecular origin for a component.
