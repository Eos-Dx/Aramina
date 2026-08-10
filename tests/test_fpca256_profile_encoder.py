from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aramina.patient_features import lr1_training_rows
from aramina.training_evaluation import _evaluate_m2q_model
from experiments.fpca256_profile_encoder.config import validate_experiment_config
from experiments.fpca256_profile_encoder.component_interpretation import (
    align_coefficient_to_reference,
    component_landmarks,
)
from experiments.fpca256_profile_encoder.model import (
    FoldLocalProfileEncoder,
    ProfileSpec,
    validate_profile_grid,
)
from experiments.fpca256_profile_encoder.lineage import validate_pyfai_runtime
from experiments.fpca256_profile_encoder.runner import (
    _validate_artifact_lineage,
    create_research_npt256_artifact,
    run_cohort_experiment,
    validate_common_cohort,
)


SOURCE_H5_SHA256 = "d2d61e83850b282c3d2479ea436deed821c4488b96983252d294f3d56ee3f1f9"
INTEGRATION_METHOD = ["bbox", "csr", "cython"]


def _config() -> dict:
    return {
        "contract": "aramina_fpca256_profile_encoder_experiment_v0_1",
        "clinical_stage": "research_only",
        "lineage": {
            "aramina_base_main_git_sha": (
                "394a34640441d33ebb994cd93107ac4447707461"
            ),
            "source_h5_sha256": SOURCE_H5_SHA256,
            "base_preprocessing_config_sha256": (
                "2913d1dfada1596bc12afec69ddb69c217af577f4cae10243931760900d01d3b"
            ),
            "integration_method": INTEGRATION_METHOD,
            "pyfai_version": "2026.5.0",
            "integration_method_source": "pyfai_integrate1d_default",
        },
        "cohorts": {
            "common": {
                "enabled": True,
                "npt100_artifact": _artifact_pin(
                    "common100.joblib",
                    npt=100,
                    variant="npt100_bbox",
                ),
                "npt256_artifact": _artifact_pin(
                    "common256.joblib",
                    npt=256,
                    variant="npt256_bbox",
                ),
                "expected_rows": 80,
                "expected_patients": 20,
                "expected_target_cases": 20,
            },
            "full_npt256": {
                "enabled": True,
                "npt256_artifact": _artifact_pin(
                    "full256.joblib",
                    npt=256,
                    variant="npt256_bbox",
                ),
                "expected_rows": 80,
                "expected_patients": 20,
                "expected_target_cases": 20,
            },
        },
        "preprocessing": {
            "base_config_path": "preprocessing.yaml",
            "generated_npt256_artifact_path": "generated.joblib",
            "integration_npt": 256,
        },
        "model": {
            "profile_column": "radial_profile_data",
            "label_column": "product_status_group",
            "group_column": "patientId",
            "specimen_column": "specimenId",
            "side_column": "side",
            "q_column": "q_range",
            "age_column": "age",
            "biopsy_column": "biopsy",
            "lr1_row_policy": "biopsy_only",
            "lr1_logreg_c": 0.1,
            "lr2_logreg_c": 0.3,
            "class_weight": "balanced",
            "profile_aggregation": "logit_average",
            "lr2_architecture": "age_plus_gated_sk_core4",
            "raw_baselines": [100, 256],
            "fpca_components": [10, 15, 20, 25, 30],
        },
        "evaluation": {
            "method": "repeated_stratified_kfold",
            "folds": 2,
            "repeats": 1,
            "random_seed": 17,
            "target_sensitivity": 0.95,
        },
        "output": {"folder": "outputs"},
    }


def _small_run_config() -> dict:
    """Use a rank-compatible FPCA sweep for the 20-case synthetic fixture."""
    config = _config()
    config["model"]["fpca_components"] = [4, 5, 6, 7]
    return config


def _artifact_pin(path: str, *, npt: int, variant: str) -> dict:
    if npt == 100:
        artifact_sha = (
            "1e33f4a5447993c40d4496dedc955a884cf8738447bb39d2b11eae0163bd4eff"
        )
        pipeline_fingerprint = (
            "fc0a58dd54a85348985b8e16201a65b9152baed22e3a6be65ecaeaffaf47d95d"
        )
    elif "full" in path:
        artifact_sha = (
            "2eb7b46c6f42d284d996bd1d9b6d0a3d8df3ed372fb7fe0b9346bcbd87479504"
        )
        pipeline_fingerprint = (
            "9436335af89fc7c9457ca614e5e4f17797de23c7cf14d771f4b20e253728968e"
        )
    else:
        artifact_sha = (
            "195d3150dedc3847ca967abbbd37d49b89a4dd68a0aab13b3e9c2f20f5e739ae"
        )
        pipeline_fingerprint = (
            "9436335af89fc7c9457ca614e5e4f17797de23c7cf14d771f4b20e253728968e"
        )
    return {
        "path": path,
        "sha256": artifact_sha,
        "pipeline_fingerprint": pipeline_fingerprint,
        "input_h5_sha256": SOURCE_H5_SHA256,
        "integration_variant": variant,
        "integration_npt": npt,
        "integration_method": INTEGRATION_METHOD,
    }


def _frame(*, npt: int = 256, patients: int = 20) -> pd.DataFrame:
    q = np.linspace(2.0, 23.0, npt)
    rows = []
    for patient_index in range(patients):
        cancer = patient_index % 2 == 0
        for side_index, side in enumerate(("Left", "Right")):
            label = "CANCER" if cancer and side == "Left" else "BENIGN"
            for measurement_index in range(2):
                signal = (
                    3.0
                    + 0.12 * np.sin(q * (0.35 + patient_index * 0.002))
                    + 0.06 * np.cos(q * 0.8 + side_index * 0.2)
                    + measurement_index * 0.004
                )
                if label == "CANCER":
                    signal = signal + 0.08 * np.exp(-((q - 14.0) / 1.2) ** 2)
                rows.append(
                    {
                        "patientId": f"P{patient_index:03d}",
                        "specimenId": f"P{patient_index:03d}_{side}",
                        "side": side,
                        "position": f"P{measurement_index + 1}",
                        "started_at": f"2026-01-{patient_index + 1:02d}T00:00:00",
                        "product_status_group": label,
                        "age": 35 + patient_index,
                        "biopsy": side == "Left",
                        "q_range": q.copy(),
                        "radial_profile_data": signal,
                    }
                )
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def frozen_common_frames() -> dict[int, pd.DataFrame]:
    """Return deterministic matched inputs for product-parity regression."""
    return {100: _frame(npt=100), 256: _frame(npt=256)}


def test_config_locks_product_model_controls() -> None:
    config = _config()
    validate_experiment_config(config)

    changed = deepcopy(config)
    changed["model"]["lr1_logreg_c"] = 1.0
    with pytest.raises(ValueError, match="Controlled model field"):
        validate_experiment_config(changed)


def test_component_landmarks_use_one_standard_deviation_change() -> None:
    landmarks = component_landmarks(
        np.array([6.0, 7.0, 8.0]),
        np.array([-1.0, 0.0, 1.0]),
        explained_variance=4.0,
    )
    assert landmarks["one_sd_min_q_nm_inv"] == 6.0
    assert landmarks["one_sd_max_q_nm_inv"] == 8.0
    assert landmarks["one_sd_peak_to_peak"] == 4.0


def test_component_coefficient_alignment_preserves_prediction_direction() -> None:
    aligned, similarity = align_coefficient_to_reference(
        np.array([1.0, 0.0]), np.array([-1.0, 0.0]), coefficient=-0.4
    )
    assert aligned == pytest.approx(0.4)
    assert similarity == pytest.approx(1.0)


def test_input_artifact_hash_mismatch_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "artifact.joblib"
    artifact = {
        "dataframe": _frame(),
        "pipeline_fingerprint": "2" * 64,
        "metadata": {
            "input_h5_sha256": SOURCE_H5_SHA256,
            "integration_variant": "npt256_bbox",
            "integration_npt": 256,
            "integration_method": INTEGRATION_METHOD,
        },
    }
    joblib.dump(artifact, path)
    with pytest.raises(ValueError, match="Input artifact SHA-256 mismatch"):
        _validate_artifact_lineage(
            path,
            artifact,
            expected=_artifact_pin(
                str(path),
                npt=256,
                variant="npt256_bbox",
            ),
            experiment_lineage=_config()["lineage"],
            generated_mode=False,
        )


def test_pyfai_runtime_version_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError, match="runtime version mismatch"):
        validate_pyfai_runtime(
            _config()["lineage"],
            installed_version="2026.4.0",
            integrate1d_default=("bbox", "csr", "cython"),
        )


def test_pyfai_integrate1d_default_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError, match="method default mismatch"):
        validate_pyfai_runtime(
            _config()["lineage"],
            installed_version="2026.5.0",
            integrate1d_default=("full", "csr", "cython"),
        )


def test_raw_h5_source_hash_mismatch_is_rejected(tmp_path: Path) -> None:
    h5_path = tmp_path / "input.h5"
    config_path = tmp_path / "preprocessing.yaml"
    h5_path.write_bytes(b"not-the-pinned-source")
    config_path.write_text("contract: unused\n", encoding="utf-8")
    with pytest.raises(ValueError, match="raw H5 input SHA-256 mismatch"):
        create_research_npt256_artifact(
            h5_path,
            base_config_path=config_path,
            output_artifact_path=tmp_path / "generated.joblib",
            lineage=_config()["lineage"],
        )


def test_validate_profile_grid_requires_256_uniform_shared_bins() -> None:
    frame = _frame().iloc[:4].copy()
    grid = validate_profile_grid(
        frame,
        profile_column="radial_profile_data",
        q_column="q_range",
        expected_npt=256,
    )
    assert grid.size == 256

    wrong_bins = frame.copy()
    wrong_bins["radial_profile_data"] = wrong_bins["radial_profile_data"].map(
        lambda values: values[:-1]
    )
    with pytest.raises(ValueError, match="Expected 256-bin profiles"):
        validate_profile_grid(
            wrong_bins,
            profile_column="radial_profile_data",
            q_column="q_range",
            expected_npt=256,
        )

    nonuniform = frame.copy()
    changed_grid = nonuniform.iloc[0]["q_range"].copy()
    changed_grid[10] += 0.01
    nonuniform.at[nonuniform.index[0], "q_range"] = changed_grid
    with pytest.raises(ValueError, match="shared q grid"):
        validate_profile_grid(
            nonuniform,
            profile_column="radial_profile_data",
            q_column="q_range",
            expected_npt=256,
        )


def test_fpca_fits_only_supplied_training_patients() -> None:
    frame = _frame(patients=12)
    train = frame[frame["patientId"].isin([f"P{i:03d}" for i in range(8)])]
    test = frame[~frame["patientId"].isin(train["patientId"].unique())]
    rows = lr1_training_rows(
        train,
        label_column="product_status_group",
        biopsy_column="biopsy",
        lr1_row_policy="biopsy_only",
    )
    encoder = FoldLocalProfileEncoder(
        spec=ProfileSpec("fpca256_4", 256, "fpca", 4),
        profile_column="radial_profile_data",
        label_column="product_status_group",
        group_column="patientId",
        q_column="q_range",
        logreg_c=0.1,
        random_state=42,
    ).fit(rows)

    assert encoder.pca_ is not None
    assert encoder.pca_.components_.shape == (4, 256)
    assert encoder.training_patient_ids_ == frozenset(train["patientId"].unique())
    assert not encoder.training_patient_ids_.intersection(test["patientId"].unique())


def test_common_cohort_rejects_measurement_mismatch() -> None:
    config = _small_run_config()
    frame100 = _frame(npt=100)
    frame256 = _frame(npt=256)
    frame256.loc[0, "position"] = "DIFFERENT"
    contexts = {
        100: _dataset_context(frame100, config),
        256: _dataset_context(frame256, config),
    }
    with pytest.raises(ValueError, match="measurement identities differ"):
        validate_common_cohort(frame100, frame256, contexts)


def test_raw_paths_match_current_product_evaluator_exactly(
    frozen_common_frames: dict[int, pd.DataFrame],
) -> None:
    config = _small_run_config()
    result = run_cohort_experiment(
        frozen_common_frames,
        config=config,
        cohort_name="common",
    )
    for npt, encoder in ((100, "raw100"), (256, "raw256")):
        product_metrics, product_predictions = _product_evaluation(
            frozen_common_frames[npt],
            config,
        )
        experiment_metrics = (
            result["fold_metrics"]
            .loc[lambda frame: frame["profile_encoder"] == encoder]
            .sort_values("split_id")
            .reset_index(drop=True)
        )
        experiment_predictions = (
            result["fold_predictions"]
            .loc[lambda frame: frame["profile_encoder"] == encoder]
            .sort_values(["split_id", "target_case_id"])
            .reset_index(drop=True)
        )
        product_predictions = product_predictions.sort_values(
            ["split_id", "target_case_id"]
        ).reset_index(drop=True)
        np.testing.assert_allclose(
            experiment_predictions["p_cancer"],
            product_predictions["p_cancer"],
            rtol=0.0,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            experiment_predictions["threshold_target"],
            product_predictions["threshold_target"],
            rtol=0.0,
            atol=1e-12,
        )
        metric_mapping = {
            "roc_auc": "roc_auc",
            "sensitivity": "sensitivity_target",
            "specificity": "specificity_target",
            "balanced_accuracy": "balanced_accuracy_target",
            "ppv": "ppv_target",
            "npv": "npv_target",
            "true_positives": "tp_target",
            "true_negatives": "tn_target",
            "false_positives": "fp_target",
            "false_negatives": "fn_target",
            "threshold_target": "threshold_target",
        }
        product_metrics = product_metrics.sort_values("split_id").reset_index(
            drop=True
        )
        for experiment_name, product_name in metric_mapping.items():
            np.testing.assert_allclose(
                experiment_metrics[experiment_name],
                product_metrics[product_name],
                rtol=0.0,
                atol=1e-12,
            )


def test_fold_manifest_is_complete_and_bilateral_patient_safe() -> None:
    config = _small_run_config()
    frame = _frame()
    bilateral = frame["patientId"].eq("P000") & frame["side"].eq("Right")
    frame.loc[bilateral, "biopsy"] = True
    result = run_cohort_experiment(
        {256: frame},
        config=config,
        cohort_name="full_npt256",
    )
    manifest = result["fold_manifest"]
    target_cases = set(
        result["fold_predictions"]["target_case_id"].astype(str).unique()
    )
    assert len(target_cases) == 21
    for _, split in manifest.groupby("split_id"):
        assert set(split["target_case_id"]) == target_cases
        assert not split["target_case_id"].duplicated().any()
        assert split.groupby("patientId")["set"].nunique().max() == 1
        bilateral_rows = split[split["patientId"] == "P000"]
        assert len(bilateral_rows) == 2
        assert bilateral_rows["set"].nunique() == 1


def test_small_end_to_end_full_npt256_run(tmp_path: Path) -> None:
    result = run_cohort_experiment(
        {256: _frame()},
        config=_small_run_config(),
        cohort_name="full_npt256",
        output_folder=tmp_path,
    )

    assert set(result["aggregate_summary"]["profile_encoder"]) == {
        "raw256",
        "fpca256_4",
        "fpca256_5",
        "fpca256_6",
        "fpca256_7",
    }
    assert len(result["fold_metrics"]) == 10
    assert result["dataset"][256]["target_cases"] == 20
    assert set(result["train_all_models"]) == set(
        result["aggregate_summary"]["profile_encoder"]
    )
    assert "repeat_averaged_cross_fitted_predictions" in result
    assert not result["paired_fold_deltas"].empty
    assert set(result["paired_fold_deltas"]["reference_encoder"]) == {"raw256"}
    for filename in (
        "fold_metrics.csv",
        "fold_predictions.csv",
        "aggregate_summary.yaml",
        "aggregate_summary.csv",
        "repeat_averaged_cross_fitted_predictions.csv",
        "fold_manifest.csv",
        "paired_fold_deltas.csv",
        "paired_delta_summary.csv",
        "train_all_artifact.joblib",
        "pca_explained_variance.csv",
        "pca_basis_components.csv",
        "pca_fold_basis.joblib",
        "roc_comparison.png",
        "fpca_component_convergence.png",
    ):
        assert (tmp_path / filename).is_file()


def _dataset_context(frame: pd.DataFrame, config: dict) -> pd.DataFrame:
    from experiments.fpca256_profile_encoder.model import build_dataset_context

    return build_dataset_context(frame, config["model"])


def _product_evaluation(
    frame: pd.DataFrame,
    config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    model = config["model"]
    evaluation = config["evaluation"]
    return _evaluate_m2q_model(
        frame,
        config={
            "evaluation": {
                "mode": "stratified_kfold",
                "n_splits": evaluation["folds"],
                "n_repeats": evaluation["repeats"],
            }
        },
        profile_column=model["profile_column"],
        label_column=model["label_column"],
        group_column=model["group_column"],
        specimen_column=model["specimen_column"],
        side_column=model["side_column"],
        q_column=model["q_column"],
        age_column=model["age_column"],
        biopsy_column=model["biopsy_column"],
        lr1_row_policy=model["lr1_row_policy"],
        lr1_logreg_c=model["lr1_logreg_c"],
        lr2_logreg_c=model["lr2_logreg_c"],
        random_state=evaluation["random_seed"],
        target_sensitivity=evaluation["target_sensitivity"],
    )
