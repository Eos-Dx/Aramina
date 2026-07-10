# T100 M0Q Patient-Label Permutation Sanity Check v0.1

Status: research draft. This is a leakage and overfitting sanity check, not
clinical validation.

## Question

Could the T100 M0Q result arise solely because the two-stage route can retain
or memorise patient labels?

## Method

The T100 biopsy cohort, target-side assignment, profiles, target measurement
counts, LR1 regularisation (`C=0.3`), LR2 regularisation (`C=0.1`), and a
fixed patient-safe stratified five-fold split were held constant.

For each null run, the `BENIGN`/`CANCER` patient labels were randomly
permuted. Target-side assignment was deliberately left unchanged: it is a
clinical-input rule, not a learning label. LR1 and LR2 were then fitted and
evaluated using only the permuted labels.

This is stricter than merely shuffling prediction scores. It tests the full
LR1 profile -> target-breast aggregation -> M0Q route under a no-label-signal
null.

## Result

```text
observed M0Q ROC AUC: 0.628 +/- 0.113 across the fixed five folds
20 patient-label permutation means: 0.490 +/- 0.066
permutations with ROC AUC >= observed: 0 / 20
empirical one-sided p: (0 + 1) / (20 + 1) = 0.048
```

The null distribution is centred near random classification. This is evidence
against a simple patient-label leakage route in M0Q. It does not prove strong
generalisation: the observed result remains uncertain because the cohort is
small, the observed five-fold standard deviation is large, and 20 permutations
give only coarse p-value resolution.

## Interpretation

The appropriate conclusion is narrow:

```text
M0Q contains signal above this permutation null in the current T100 cohort.
The current evidence is not sufficient to claim that it will generalise to a
new acquisition cohort.
```

The next stronger checks are:

1. Increase to at least 100-1,000 patient-label permutations for a stable null
   estimate.
2. Run a temporally separated or independently acquired patient hold-out.
3. Repeat learning curves over increasing patient counts. A genuine signal
   should stabilise or improve as the cohort increases.

Result file:

```text
docs/modeling/results/t100_m0q_patient_label_permutation_v0_1.csv
```
