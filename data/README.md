# Product H5 Data

`combined_archive.h5.dvc` is the frozen `0.2.14-beta` training archive pointer.
`20260827_combined_archive.h5.dvc` is the `0.2.15-beta` development archive
pointer. H5 payloads are not stored in Git.

Configure this workstation once:

```bash
python -m pip install -e '.[data]'
dvc config --local cache.dir /path/to/controlled/aramina-dvc/cache
dvc config --local cache.type reflink,hardlink,copy
dvc remote add --local --default internal-h5 /path/to/controlled/aramina-dvc/remote
```

Materialize and verify the tracked revision:

```bash
dvc pull data/combined_archive.h5.dvc
dvc status --cloud
```

`.dvc/config.local`, the cache, and the internal remote are machine-local and
must not be committed. A future external DVC remote can replace the local URL
without changing the data pointer or model code.
