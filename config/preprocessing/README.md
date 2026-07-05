# Aramis Preprocessing YAML Layout

Root `aramis_*_v0_1.yaml` files are runnable configs.

Current v0.1-beta model-input configs:

```text
aramis_all_patients_model_input_v0_1.yaml
aramis_biopsy_patients_model_input_v0_1.yaml
```

Use these two configs for the current M0/M1/M2 comparison unless a review task
explicitly asks for an experimental cohort.

Shared fragments:

```text
shared/aramis_policy_v0_1.yaml      common GFRM-only product policy
shared/aramis_pipeline_v0_1.yaml    ordered transformer steps
exclusions/agbh_quality_exclusions_v0_1.yaml  long AgBH exclusion lists
outputs/max_output_v0_1.yaml        MAX output schema, keep transform columns
outputs/min_output_v0_1.yaml        MIN output schema, model-essential columns
branches/*.yaml                     one-to-one/one-to-many cohort rules
```

`aramis_main_max_v0_1.yaml` and `aramis_main_min_v0_1.yaml` are shared bases.
They are not directly runnable because they do not define a branch.

Runnable configs extend one MAIN base plus one branch fragment, then define only:

```text
aramis_preprocessing.name/version
io.output_joblib_path
branch_settings.output_columns
```

`io.output_joblib_path` receives a preprocessing artifact joblib: it contains
the final DataFrame plus the resolved YAML config, original YAML text, config
SHA256, H5 SHA256, Aramis version/git SHA, and branch.

Runnable one-to-many cohort configs:

```text
aramis_all_patients_model_input_v0_1.yaml
    model-input cohort: all labelled patients, NORMAL -> BENIGN,
    selected patient/specimen metadata plus normalized radial_profile_data

aramis_biopsy_patients_model_input_v0_1.yaml
    model-input cohort: keep patients with any biopsy=True row,
    include contralateral/non-biopsy rows, NORMAL -> BENIGN,
    selected patient/specimen metadata plus normalized radial_profile_data

aramis_one_to_many_biopsy_min_v0_1.yaml
    strict row-level biopsy=True cohort

aramis_one_to_many_biopsy_patient_expanded_min_v0_1.yaml
    patient-level biopsy cohort: keep any patient with at least one biopsy=True row,
    keep contralateral rows, map NORMAL specimens to BENIGN product group,
    use AgBH monochromaticity exclusions

aramis_one_to_many_fda_like_date_only_min_v0_1.yaml
    FDA-like date-only cohort: keep biopsy=True rows from 2026-01-01 onward,
    do not apply AgBH monochromaticity exclusions

aramis_monochromatic_metadata_pool_max_v0_1.yaml
    source-pool cohort: use AgBH monochromaticity exclusions,
    do not filter by diagnosis or biopsy,
    keep MAX metadata for later DataFrame slicing by biopsy/date/status
```

Experimental / historical grid configs are kept for audit and method
comparison:

```text
aramis_grid_t70_*_v0_1.yaml
aramis_grid_t100_*_v0_1.yaml
aramis_grid_t130_*_v0_1.yaml
aramis_wide_t70_max_v0_1.yaml
aramis_wide_t100_max_v0_1.yaml
aramis_wide_t130_max_v0_1.yaml
aramis_one_to_many_fda_like_date_only_min_v0_1.yaml
```

Do not treat these as the default product-development route without updating
the implementation plan and rerunning the model grid.

Biopsy branch meaning:

```text
one-to-many biopsy: row-level biopsy filter; keep only biopsy=True specimens
one-to-many biopsy-patient-expanded: patient-level biopsy filter; keep patients
                                      with any biopsy=True row and include
                                      contralateral/non-biopsy rows
one-to-one biopsy: patient-level biopsy filter; keep patients with any biopsy=True row,
                   then keep both breasts for paired symmetry analysis
```
