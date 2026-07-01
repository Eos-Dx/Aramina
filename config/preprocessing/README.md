# Aramis Preprocessing YAML Layout

Root `aramis_one_to_*_v0_1.yaml` files are runnable configs.

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
path, SHA256, and run metadata.

Biopsy branch meaning:

```text
one-to-many biopsy: row-level biopsy filter; keep only biopsy=True specimens
one-to-one biopsy: patient-level biopsy filter; keep patients with any biopsy=True row,
                   then keep both breasts for paired symmetry analysis
```
