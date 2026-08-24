# Data Versioning

Status: research draft product-data contract for Aramina `0.2.x`.

## Ownership

| Layer | Canonical responsibility |
|---|---|
| Git | Code, YAML contracts, and `data/combined_archive.h5.dvc`. |
| DVC | Exact content revision of the internal source H5. |
| MLflow | One preprocessing, evaluation, final-fit, and model run against that DVC revision. |

The source H5 remains internal. Git stores only its DVC content hash, size, and
relative output path. The current workstation uses a local internal DVC remote;
the remote URL is kept in ignored `.dvc/config.local`.

## Fail-Closed Training Check

The product training YAML declares:

```yaml
data_version:
  contract: aramina_dvc_input_v0_1
  system: dvc
  dataset_id: aramina_combined_archive_h5
  dvc_version: 3.67.1
  pointer_path: data/combined_archive.h5.dvc
```

Before preprocessing, Aramina requires one DVC output and verifies its path,
byte size, and MD5 content hash. SHA256 is calculated during the same file read
and remains the independent source checksum used by the model artifact and
MLflow.

Each MLflow run stores `data_version.json`, a copy of the `.dvc` pointer, and the tags
`data_version_system`, `dvc_dataset_id`, `dvc_data_hash`, and
`dvc_pointer_path`. The training joblib stores the same record under
`reproducibility.source_h5.data_version`.

## Local Internal Storage

```bash
python -m pip install -e '.[data]'
dvc config --local cache.dir /path/to/controlled/aramina-dvc/cache
dvc config --local cache.type reflink,hardlink,copy
dvc remote add --local --default internal-h5 /path/to/controlled/aramina-dvc/remote
dvc pull data/combined_archive.h5.dvc
```

`dvc add` updates the pointer after an intentional source-data revision.
`dvc push` copies that revision to the configured internal remote. Data changes
must be reviewed together with preprocessing, model evaluation, and MLflow
lineage; changing only the H5 pointer does not promote a product model.
