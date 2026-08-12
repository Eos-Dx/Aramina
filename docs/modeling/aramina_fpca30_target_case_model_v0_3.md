# Aramina FPCA30 Target-Case Model v0.3

Status: research-draft source model definition.

Model version: `0.3.1-beta`.

No `0.3.1-beta` joblib is tracked in Git. The executable artifact is generated
from the pinned preprocessing and training YAMLs. Preserved `0.2.x` joblibs
remain unchanged under `models/`.

## Purpose

The change tests whether the first model stage can retain useful profile
information while reducing 256 integrated q bins to 30 orthogonal components.
It is dimensionality reduction, not a new clinical endpoint. The cohort, labels,
LR2 architecture, Core4 symmetry gate, regularization, and threshold policy are
unchanged.

The component-count decision, including the 10-30 sweep and late-component
stability audit, is documented in
[`fpca30_component_selection_v0_1.md`](fpca30_component_selection_v0_1.md).

## Architecture

```text
2D XRD frame
-> azimuthal integration: 256 bins, q=2..23 nm^-1
-> normalization: median at q=6.7..7.1 nm^-1
-> FPCA/PCA: 256 bins to 30 components
-> StandardScaler
-> LR1 LogisticRegression, C=0.1
-> measurement p_cancer values
-> target-breast logit-average
-> LR2 with age and optional gated Core4 symmetry, C=0.3
-> final p_cancer and fixed threshold
```

FPCA30 is fitted only on LR1 training rows. During patient-safe evaluation it
is refitted inside every train fold; held-out patients never contribute to the
component basis. Final train-on-all fits the basis on all accepted LR1 rows and
stores it inside `model.joblib`. Prediction therefore applies exactly the fitted
training transform.

The fitted train-on-all basis explains `96.816%` of profile variance. This value
describes profile reconstruction, not BENIGN/CANCER separation. Thirty
components were selected because they best preserved the preceding raw100
sensitivity/specificity operating point in the controlled component sweep, not
because `96.816%` is a predefined variance threshold.

## Data And Evaluation

```text
source H5 SHA256: d2d61e83850b282c3d2479ea436deed821c4488b96983252d294f3d56ee3f1f9
measurements: 876
patients after preprocessing: 163
LR1 patients: 161
target-breast cases: 171
  CANCER: 74
  BENIGN: 97
evaluation: repeated patient-safe stratified 5-fold x20, seed 42
```

All target cases and measurements belonging to one patient remain in one fold.
The final train-on-all threshold targets sensitivity at least `0.95` on accepted
training target cases.

## Results

### Component-count evidence

On the matched 161-patient cohort, FPCA10 gave ROC AUC `0.670`, sensitivity
`0.891`, and specificity `0.277`. FPCA30 gave ROC AUC `0.656`, sensitivity
`0.832`, and specificity `0.404`, compared with `0.679`, `0.826`, and `0.400`
for raw100. Ten components therefore retained ranking but shifted too many
BENIGN cases above the high-sensitivity threshold. Thirty components recovered
the raw100 thresholded operating point while reducing 256 profile coordinates
to 30. The complete rationale and limitations are recorded in the component
selection document.

### Release verification

Patient-safe held-out means across 100 folds:

| metric | 0.2.13-beta raw100 | 0.3.1-beta FPCA30 |
|---|---:|---:|
| ROC AUC | 0.645 | 0.648 |
| sensitivity | 0.818 | 0.860 |
| specificity | 0.376 | 0.312 |

This release-lineage table is descriptive rather than paired: the preserved
`0.2.13-beta` record contains 164 patients and 175 target cases, whereas the
`0.3.1-beta` run contains 163 patients and 171 target cases. The controlled
component-selection comparison used the exact matched 161-patient, 164-case
cohort and is reported separately above.

Final train-on-all description:

| metric | 0.2.13-beta raw100 | 0.3.1-beta FPCA30 |
|---|---:|---:|
| threshold | 0.24666 | 0.25992 |
| ROC AUC | 0.86497 | 0.80649 |
| sensitivity | 0.96053 | 0.95946 |
| specificity | 0.49495 | 0.35052 |
| TP / TN / FN / FP | 73 / 49 / 3 / 50 | 71 / 34 / 3 / 63 |

Train-on-all values describe fitted training cases and are not independent
validation. Patient-safe evaluation remains the relevant evidence for transfer
to new patients.

## Interpretation

FPCA30 reduced the LR1 input from 256 bins to 30 components and retained nearly
the same held-out ROC AUC as raw100. Held-out sensitivity increased by about
4.2 percentage points, while specificity decreased by about 6.4 points. The
train-on-all ROC AUC and specificity also decreased. Thus dimensionality was
reduced successfully, but this run did not establish a better product operating
point.

This result is consistent with the selection experiment but weaker than its
matched-cohort operating-point comparison. It confirms that 30 is a documented
compression choice, not evidence of improved clinical performance.

All three one-patient H5 fixtures completed preprocessing and report generation
with the generated `0.3.1-beta` joblib. The benign fixture was above the frozen
threshold, consistent with limited specificity. This model remains research
decision support and is not an autonomous diagnosis.

## Reproduction

```bash
python -m aramina preprocess-train \
  --config config/preprocessing_and_training/config_preprocess_and_train_target_breast_risk_v0_1.yaml \
  --verbose
```

The source H5 is a controlled external input at `data/combined_archive.h5` and
is not stored in Git.
