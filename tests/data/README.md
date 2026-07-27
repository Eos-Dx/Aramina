# Aramina Test Data

`aramina_real_h5_subset_20260128_5_patients.h5` is a small real Eos-Dx H5 v0.3
archive slice copied from:

```text
/Users/sad/dev/eos_play/jupyter_notebooks/Clinical_trials/data/product-aramina-data/combined_archive.h5
```

Copied archive group:

```text
calib_20260128_132622
```

Copied child sessions:

```text
calibration
sample_01_20260128_Nova_376_Right
sample_02_20260128_Nova_376_Left
sample_05_20260128_Nova_378_Right
sample_06_20260128_Nova_378_Left
sample_07_20260128_Nova_379_Right
sample_08_20260128_Nova_379_Left
sample_15_20260128_Nova_383_Right
sample_16_20260128_Nova_383_Left
sample_17_20260128_Nova_384_Right
sample_18_20260128_Nova_384_Left
```

The fixture has 5 patients and 10 sample sessions. Each sample session contains 3 sample
measurement sets, so `H5ToDataFrameTransformer(..., set_category="SAMPLE")`
returns 30 measurement rows.

Patient-level diagnosis pairs:

```text
Nova_376: BENIGN / BENIGN
Nova_378: NORMAL / BENIGN
Nova_379: NORMAL / BENIGN
Nova_383: CANCER / NORMAL
Nova_384: BENIGN / CANCER
```
