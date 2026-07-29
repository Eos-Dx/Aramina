# pyFAI radial-resolution and pixel-splitting sweep

Research-only comparison of the frozen Aramina model architecture under eight
azimuthal-integration settings:

| Variant | `npt` | pyFAI method |
|---|---:|---|
| `npt100_bbox` | 100 | `bbox / csr / cython` |
| `npt150_bbox` | 150 | `bbox / csr / cython` |
| `npt200_bbox` | 200 | `bbox / csr / cython` |
| `npt250_bbox` | 250 | `bbox / csr / cython` |
| `npt256_bbox` | 256 | `bbox / csr / cython` |
| `npt512_bbox` | 512 | `bbox / csr / cython` |
| `npt512_no` | 512 | `no / csr / cython` |
| `npt512_full` | 512 | `full / csr / cython` |

Everything else remains fixed: source H5, PONI geometry, q range, error model,
QC, normalization, labels, model architecture, regularization, threshold
policy, 5-fold x20 patient-safe evaluation, and random seed.

The runner preprocesses every variant first and trains all models on the common
measurement intersection. This prevents cohort changes from confounding the
integration comparison. Per-variant preprocessing retention is reported
separately.

```bash
PYTHONPATH=/Users/sad/dev/Aramina-npt-experiment/src:/Users/sad/dev/XRD-preprocessing-npt-experiment/src \
conda run -n eosproduct python \
  experiments/npt_method_sweep/run_npt_method_sweep.py
```

Generated large artifacts remain under `outputs/npt_method_sweep/`. Compact
tables, plots, and the final interpretation are written beside this README.

