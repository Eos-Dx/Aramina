# Aramis Reproducible Training Bundle

Status: research draft decision-support prototype.

Run `install_and_train.bat` on Windows or `./install_and_train.sh` on macOS/Linux.
The launchers install Miniforge and Git when required, then check out the exact
Aramis and XRD-preprocessing commits in `bundle_manifest.json`. Existing bundle
workspaces and conda environments are reused: the repositories are fetched and
reset to the selected commits, while editable package installation refreshes the
two Python packages without recreating the environment.

Each run verifies the copied H5 checksum, deletes only prior generated workflow
outputs, runs the full H5 preprocessing and M2Q `preprocess-train` workflow,
and compares the generated model with the reference model from the selected
Aramis checkout. The console shows each stage and writes the same output to
`workspace/logs/install_and_train_<timestamp>.log`.

The full input archive is stored at `workspace/data/combined_archive.h5`.

The comparison requires equal H5 SHA256, recipe, YAML checksums, evaluation
summary, thresholds, and executable LR1/LR2 parameters. `created_at` and the
joblib file SHA are intentionally not compared because they change for each run.

The H5 is copied into the workspace layout expected by the immutable historical
preprocessing YAML. Do not edit the H5 or any YAML before the comparison.
