# Aramis Reproducible Training Bundle

Status: research draft decision-support prototype.

Run `install_and_train.bat` on Windows. It installs Miniforge when required,
installs Git when required through `winget`, checks out the exact Aramis and
XRD-preprocessing commits in `bundle_manifest.json`, creates the conda
environment, runs the full H5 preprocessing and M2Q `preprocess-train` workflow,
then compares the generated model with the reference model loaded from the
checked-out Aramis repository.

The comparison requires equal H5 SHA256, recipe, YAML checksums, evaluation
summary, thresholds, and executable LR1/LR2 parameters. `created_at` and the
joblib file SHA are intentionally not compared because they change for each run.

The H5 is copied into the workspace layout expected by the immutable historical
preprocessing YAML. Do not edit the H5 or any YAML before the comparison.
