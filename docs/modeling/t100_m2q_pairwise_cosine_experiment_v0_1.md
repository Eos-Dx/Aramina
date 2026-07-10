# T100 M2Q Pairwise Cosine Experiment v0.1

Status: research draft. This experiment tests the original Aramis pairwise
cosine symmetry formulation. It does not change the current M2Q product
schema.

## Question

The current M2Q feature table already computes four raw cosine fields for
audit, but they are not part of the model schema. The earlier formulation is
more specific than a cosine distance between two averaged profiles:

```text
between_mean:
  mean cosine distance across every target measurement x every contralateral measurement

within_target_mean:
  mean cosine distance across every pair of target-breast replicates

within_contralateral_mean:
  mean cosine distance across every pair of contralateral-breast replicates

within_mean:
  mean(within_target_mean, within_contralateral_mean)

asymmetry_score:
  between_mean - within_mean
```

Thus the score asks whether the difference between breasts exceeds normal
measurement-to-measurement variation within each breast. This is the intended
paired-measurement cosine representation. It differs from
`sk_cosine_distance_full_q2`, which is an SK field calculated on aggregated
profiles.

## Availability

The pairwise value is never silently treated as biological zero. The experiment
keeps explicit flags:

```text
pairwise cosine asymmetry available: 147 / 164 patients
target within-breast replicate value available: 156 / 164 patients
contralateral within-breast replicate value available: 145 / 164 patients
```

For unavailable values, the value-times-availability field is zero and its
availability flag remains zero. The model can therefore distinguish an absent
pairwise estimate from a measured small distance.

## Evaluation

```text
cohort: T100 biopsy cohort, 164 patients
model core: M2Q screened_core_4
LR1 C: 0.3
LR2 C: 0.1
validation: repeated patient-safe stratified 5-fold, 20 repeats, 100 folds
threshold: learned on each train fold for target sensitivity 0.95
```

Each fold trains LR1 and LR2 only on its train patients. Pairwise cosine fields
are then built separately for the corresponding train and held-out patients.

## Results

| candidate | pairwise cosine fields added | ROC AUC | sensitivity | specificity |
|---|---:|---:|---:|---:|
| `screened_core_4` | 0 | 0.679 +/- 0.072 | 0.789 +/- 0.113 | 0.459 +/- 0.124 |
| `+ pairwise_asymmetry` | 3 | 0.681 +/- 0.070 | 0.797 +/- 0.114 | 0.457 +/- 0.119 |
| `+ pairwise_between_within` | 6 | 0.675 +/- 0.072 | 0.791 +/- 0.117 | 0.462 +/- 0.114 |
| `+ full_pairwise_block` | 9 | 0.671 +/- 0.073 | 0.790 +/- 0.123 | 0.468 +/- 0.118 |
| no SK + full pairwise block | 9 | 0.655 +/- 0.075 | 0.793 +/- 0.122 | 0.446 +/- 0.112 |

The asymmetry-only addition is numerically positive but small:

```text
ROC AUC: +0.0020
sensitivity: +0.0087
specificity: -0.0017
```

This is far smaller than the fold-to-fold variation. The full pairwise block
reduces held-out ROC AUC by `0.0087`. It cannot replace the four selected SK
fields.

Train-all values are descriptive only:

| candidate | ROC AUC | sensitivity | specificity |
|---|---:|---:|---:|
| `screened_core_4` | 0.904 | 0.960 | 0.573 |
| `+ pairwise_asymmetry` | 0.903 | 0.960 | 0.539 |
| `+ full_pairwise_block` | 0.902 | 0.960 | 0.551 |

## Relation To Existing Audit Fields

The old and current calculations overlap but are not identical:

```text
pairwise within-target vs audit target-within: r = 1.000
pairwise within-contralateral vs audit contralateral-within: r = 1.000
pairwise centroid vs audit between-breasts mean profile: r = 1.000
pairwise asymmetry score vs audit symmetry_cosine_score: r = 0.908
```

The difference is meaningful: `between_mean` uses all cross-breast measurement
pairs, while the audit `between_breasts_cosine_distance_mean` uses the two
breast mean profiles. The pairwise asymmetry score is therefore retained in the
internal audit table and remains an experimental candidate.

## Decision

Do not add pairwise cosine fields to M2Q v0.1 now. Keep the current
`screened_core_4` candidate. Reconsider the pairwise asymmetry score when a
larger independent cohort is available; it is the only pairwise field family
member with a small positive held-out signal.

Results:

```text
docs/modeling/results/t100_m2q_pairwise_cosine_comparison_v0_1.csv
docs/modeling/results/t100_m2q_pairwise_cosine_correlations_v0_1.csv
```

