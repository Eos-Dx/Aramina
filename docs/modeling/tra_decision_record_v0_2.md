# TRA Decision Record v0.2

Status: research draft.

Aramis TRA changed from a percentile-only rank to a threshold-centred internal
score tier. The prior rank did not identify the decision boundary: a score near
the threshold could be assigned any percentile-defined tier. The current policy
keeps the fixed `p_cancer` decision threshold as the only source of the target
action and places the `TRA 2/3` boundary exactly at that threshold.

The borderline width is recalibrated from repeated patient-safe OOF prediction
stability whenever the model is retrained. The model artifact records the OOF
source, number of target cases, unstable cases, margin boundaries, derived
probability boundaries, and the final threshold. This makes the tiering policy
reproducible without changing `p_cancer`, model coefficients, or the classifier
threshold-selection method.

The target report uses `target_class_risk_level` and `biopsy_required` from the frozen
probability threshold. TRA is internal explanatory evidence only. Contralateral
scoring retains TRA but has no independent action because it is not the
clinician-selected suspicious breast and its symmetry terms are neutralized.

Full contract: `docs/contracts/tissue_risk_assessment_v0_2.md`.
