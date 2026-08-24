# Aramina Acquisition Protocol Contract v0.1

Status: research draft. This contract defines a reproducible, radiologist-led
acquisition record for Aramina clinical decision-support research. It does not
define autonomous diagnosis, treatment, or a release claim.

Canonical files:

- [YAML schema](../../config/acquisition/schema/aramina_acquisition_protocol_v0_1.yaml)
- [Validated YAML example](../../examples/acquisition/aramina_acquisition_protocol_v0_1.yaml)
- [Validator](../../src/aramina/acquisition_contract.py)

## Clinical Context

The baseline record is collected before biopsy for women with BI-RADS 3 or
BI-RADS 4 findings. Intended user is a radiologist or qualified breast-imaging
clinician. Measurements support research decision support and require clinical
review; they are not an autonomous cancer diagnosis.

## Profile And Geometry

The breast profile is captured and stored for both target and contralateral
breasts. Each profile stores a stable profile identifier, laterality, measured
breast width, and measured thickness.

The acquisition system records a machine-guided central point in instrument
coordinates for each breast. The middle plane is defined by the stored breast
profile and the instrument coordinate system. Each breast has exactly three
standardized points at normalized width positions `0.5 - alpha`, `0.5`, and
`0.5 + alpha`, where `alpha` is recorded in the protocol record. In physical
coordinates, offsets are `center +/- alpha * measured breast width`. The
validator requires `0 < alpha < 0.5` and equal symmetric offsets. Physical
offsets are therefore proportional to measured breast width; an operator-selected fixed
millimetre distance is forbidden. The final alpha remains pending hardware
validation and is not fixed by this research draft.

A point-count variant is selected only after profile and signal-quality review.
Counts are total patient points, not points per breast:

| Variant | Application |
|---|---|
| 6 total points | Three standardized middle-plane points per breast. |
| 9 total points | Six standardized points plus three target-lesion-local points. |
| 12 total points | Nine points plus three mirrored contralateral lesion-local points. |

The selected total and every normalized position are stored. Target and
mirrored contralateral lesion-local blocks are arrays of three points, with
paired points sharing a `mirror_pair_id` and retaining their own measured
coordinates. The 6-point variant stores both local blocks as empty arrays, the
9-point variant stores only the target block, and the 12-point variant stores
both blocks.

## Measurements And Thickness

Every standardized and lesion-local point has both:

- XRD acquisition status, measurement identifier, and raw-data reference.
- Attenuation coefficient in `cm^-1`, uncertainty, method, and source.

Preferred sample/breast thickness is `<=50.0 mm`. The record stores an explicit
measured value. A value above the preferred limit is not silently discarded: it
requires qualified review, a deviation record, and a reason before the record
can be accepted for research analysis.

## Dose And Traceability

Each point stores planned dose, delivered dose, maximum allowed dose, and a
limit result. The session also stores planned, delivered, and maximum cumulative
dose. Per-point and cumulative limits must be enforced, with stop-on-exceedance
control enabled. The example values are illustrative measurements, not a
clinical dose recommendation.

Operator identity and role, hardware identity/model/serial/firmware/calibration,
and session identity/timestamps/software/protocol version are mandatory.

## Timing And Follow-up

The baseline visit is `baseline_pre_biopsy`. Acquisition and planned biopsy
timestamps plus measured hours before biopsy are recorded. Any timing deviation
requires a reason.

Future longitudinal visits are optional. They may be appended as separately
identified visits after this contract is extended with visit-specific metadata;
they do not change the baseline pre-biopsy endpoint definition.
