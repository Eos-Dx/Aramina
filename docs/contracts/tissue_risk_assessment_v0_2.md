# Tissue Risk Assessment Contract v0.2

Status: research draft. TRA is an internal ordinal score tier. It is not an
individual cancer probability, diagnosis, clinical calibration, or independent
decision rule.

## Decision Rule

The target-breast action is determined only by the frozen final-model
probability threshold:

```text
p_cancer < decision_threshold  -> suggested_class: BENIGN, biopsy_required: false
p_cancer >= decision_threshold -> suggested_class: CANCER, biopsy_required: true
```

TRA does not replace or modify this rule. Contralateral evidence has a TRA
level but no suggested class or biopsy action.

## Automatic Calibration at Training

Every final fit freezes a new TRA policy in its `model.joblib` after repeated
patient-safe stratified evaluation. For each historical target case, the
trainer aggregates its out-of-fold predictions and measures whether the
threshold-derived class changes across repetitions.

```text
case margin = logit(median OOF p_cancer) - logit(final decision_threshold)
```

Cases whose decision changes across OOF folds define the borderline zone. The
median absolute margin of these cases is rounded to `0.1` logit units and
stored as `borderline_margin_logit`. `high_margin_logit` is three times this
value. If fewer than five unstable OOF target cases are available, a documented
`0.5`-logit fallback is stored instead. The final probability threshold still
changes whenever the model is retrained.

```text
TRA 1: margin < -borderline_margin_logit
TRA 2: -borderline_margin_logit <= margin < 0
TRA 3: 0 <= margin < borderline_margin_logit
TRA 4: borderline_margin_logit <= margin < high_margin_logit
TRA 5: margin >= high_margin_logit
```

Thus the `TRA 2/3` boundary is exactly the final probability decision
threshold. `TRA 1-2` agree with the BENIGN action; `TRA 3-5` agree with the
CANCER action. TRA 3 is a borderline score tier above the decision threshold;
TRA 4 and TRA 5 are high and very-high score tiers, respectively. These terms
describe position relative to the frozen model threshold, not calibrated cancer
probability or expected individual error.

## Frozen Artifact

```yaml
tissue_risk_assessment:
  contract: aramis_tra_v0_2
  reference_score: final_prediction.p_cancer
  reference_population: patient_safe_oof_target-breast_cases
  decision_threshold: 0.24666
  calibration:
    method: patient_safe_oof_decision_stability
    target_cases: 175
    unstable_target_cases: 76
    decision_stability_cutoff: 1.0
    borderline_margin_logit: 0.5
  logit_margin_boundaries:
    tra_1_to_2: -0.5
    tra_2_to_3: 0.0
    tra_3_to_4: 0.5
    tra_4_to_5: 1.5
  probability_boundaries:
    tra_1_to_2: 0.16569
    tra_2_to_3: 0.24666
    tra_3_to_4: 0.35058
    tra_4_to_5: 0.59471
```

The numeric example is illustrative. Each retraining run recalculates and
freezes its own policy, including threshold-dependent probability boundaries.
