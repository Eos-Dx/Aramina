from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
import yaml
from xrd_preprocessing import save_preprocessing_artifact

from aramina.__main__ import main
from aramina.training import (
    PatientModelInputBuilder,
    _logit_average_probability,
    _patient_split_pairs,
    _project_owned_path,
    _lr1_training_rows,
    run_training_from_config,
)
from aramina.training_config import load_training_config
from aramina.training_config import PRODUCT_MODEL_NAME
from aramina.target_breast_model import build_profile_logistic
from aramina.symmetry_features import (
    SK_SYMMETRY_COLUMNS,
    target_contralateral_symmetry_features,
)


def _patient_training_frame() -> pd.DataFrame:
    rows = []
    q = np.linspace(2.0, 23.0, 256)
    for patient_idx in range(30):
        cancer = patient_idx % 3 == 0
        patient_label = "CANCER" if cancer else "BENIGN"
        for side in ("Left", "Right"):
            specimen_id = f"P{patient_idx:02d}_{side}"
            specimen_label = patient_label if side == "Left" else "BENIGN"
            for measurement_idx in range(2):
                shift = 0.8 if specimen_label == "CANCER" else -0.4
                rows.append(
                    {
                        "patientId": f"P{patient_idx:02d}",
                        "specimenId": specimen_id,
                        "measurementId": f"{specimen_id}_M{measurement_idx}",
                        "side": side,
                        "product_status_group": specimen_label,
                        "radial_profile_data": shift
                        + np.sin(q / 3.0)
                        + measurement_idx * 0.01,
                        "q_range": q,
                        "age": 45 + patient_idx,
                        "biopsy": side == "Left",
                    }
                )
    return pd.DataFrame(rows)


def test_fpca30_profile_model_requires_256_bins():
    model = build_profile_logistic(logreg_c=0.1, random_state=42)
    labels = np.array([0, 1] * 20)

    with pytest.raises(ValueError, match="requires 256-bin profiles"):
        model.fit(np.zeros((40, 100)), labels)


def _training_config(
    input_path: Path,
    output_folder: Path,
    *,
    mode: str,
) -> dict:
    return {
        "contract": "aramina_training_config_v0_3",
        "model": {
            "name": PRODUCT_MODEL_NAME,
            "version": "0.1-beta",
            "model_author": "test",
            "clinical_stage": "research draft",
            "intended_use": "Synthetic decision-support test.",
        },
        "run": {
            "evaluation": mode in {"evaluation", "final_fit"},
            "train_on_all": mode == "final_fit",
        },
        "input": {"dataframe_joblib_path": str(input_path)},
        "output": {"folder": str(output_folder)},
        "evaluation": {
            "method": "repeated_stratified_kfold",
            "folds": 5,
            "repeats": 20,
            "random_seed": 42,
        },
    }


def _write_training_input(path: Path) -> None:
    save_preprocessing_artifact(
        _patient_training_frame(),
        path,
        preprocessing_config_text=(
            "pipeline:\n"
            "  steps:\n"
            "  - name: test\n"
            "    transformer: H5ToDataFrameTransformer\n"
        ),
        metadata={"input_h5_sha256": "abc"},
    )


def test_model_owned_preprocessing_path_uses_training_config_project_root(tmp_path: Path):
    config_path = tmp_path / "Aramina" / "config" / "training" / "train.yaml"
    config_path.parent.mkdir(parents=True)
    expected = (
        tmp_path
        / "Aramina"
        / "config"
        / "preprocessing"
        / "config_preprocessing_prediction_patient_v0_1.yaml"
    )
    expected.parent.mkdir()
    expected.touch()

    resolved = _project_owned_path(
        "config/preprocessing/config_preprocessing_prediction_patient_v0_1.yaml",
        config_path,
    )

    assert resolved == expected


def test_training_contract_rejects_unknown_fields(tmp_path: Path):
    config_path = tmp_path / "train.yaml"
    config = _training_config(tmp_path / "input.joblib", tmp_path, mode="evaluation")
    config["evaluation"]["shuffle"] = True
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(ValueError, match="Unknown evaluation fields"):
        load_training_config(config_path)


def test_training_contract_allows_custom_repeated_stratified_kfold(tmp_path: Path):
    config_path = tmp_path / "train.yaml"
    config = _training_config(tmp_path / "input.joblib", tmp_path, mode="evaluation")
    config["evaluation"] = {
        "method": "repeated_stratified_kfold",
        "folds": 3,
        "repeats": 2,
        "random_seed": 7,
    }
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    loaded, _text = load_training_config(config_path)

    assert loaded["evaluation"] == config["evaluation"]


def test_training_contract_requires_intended_use(tmp_path: Path):
    config_path = tmp_path / "train.yaml"
    config = _training_config(tmp_path / "input.joblib", tmp_path, mode="evaluation")
    del config["model"]["intended_use"]
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(ValueError, match=r"Missing model fields: \['intended_use'\]"):
        load_training_config(config_path)


@pytest.mark.parametrize(
    ("section", "field", "value", "error"),
    [
        ("model", "model_author", 42, "model.model_author must be a string"),
        ("model", "intended_use", "  ", "model.intended_use must not be empty"),
        ("input", "dataframe_joblib_path", 42, "input.dataframe_joblib_path must be a string"),
        ("output", "folder", "", "output.folder must not be empty"),
    ],
)
def test_training_contract_rejects_invalid_string_values(
    tmp_path: Path,
    section: str,
    field: str,
    value: object,
    error: str,
):
    config_path = tmp_path / "train.yaml"
    config = _training_config(tmp_path / "input.joblib", tmp_path, mode="evaluation")
    config[section][field] = value
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises((TypeError, ValueError), match=error):
        load_training_config(config_path)


def test_training_contract_requires_at_least_one_requested_operation(tmp_path: Path):
    config_path = tmp_path / "train.yaml"
    config = _training_config(tmp_path / "input.joblib", tmp_path, mode="evaluation")
    config["run"] = {"evaluation": False, "train_on_all": False}
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(ValueError, match="At least one"):
        load_training_config(config_path)


def test_training_contract_allows_only_product_evaluation_method(tmp_path: Path):
    config_path = tmp_path / "train.yaml"
    config = _training_config(tmp_path / "input.joblib", tmp_path, mode="evaluation")
    config["evaluation"]["method"] = "leave_one_out"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(ValueError, match="evaluation.method"):
        load_training_config(config_path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("folds", 1, "evaluation.folds must be >= 2"),
        ("repeats", 0, "evaluation.repeats must be >= 1"),
        ("random_seed", -1, "evaluation.random_seed must be >= 0"),
    ],
)
def test_training_contract_rejects_invalid_evaluation_values(
    tmp_path: Path,
    field: str,
    value: int,
    message: str,
):
    config_path = tmp_path / "train.yaml"
    config = _training_config(tmp_path / "input.joblib", tmp_path, mode="evaluation")
    config["evaluation"][field] = value
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_training_config(config_path)


def test_evaluation_mode_writes_patient_safe_footprint_only(tmp_path: Path):
    input_path = tmp_path / "input.joblib"
    config_path = tmp_path / "train.yaml"
    _write_training_input(input_path)
    config_path.write_text(
        yaml.safe_dump(_training_config(input_path, tmp_path / "runs", mode="evaluation")),
        encoding="utf-8",
    )

    artifact = run_training_from_config(config_path)
    run_folder = Path(artifact["run_folder"])

    assert artifact["output_type"] == "aramina_evaluation_artifact"
    assert len(artifact["split_metrics"]) == 100
    assert set(artifact["split_metrics"]["evaluation_mode"]) == {"stratified_kfold"}
    summary = artifact["metric_summary"].iloc[0]
    assert np.isfinite(summary["roc_auc_ci_low"])
    assert np.isfinite(summary["specificity_ci_high"])
    assert (run_folder / "evaluation_metrics.csv").exists()
    assert (run_folder / "evaluation_predictions.csv").exists()
    assert not (run_folder / "evaluation.joblib").exists()
    assert not (run_folder / "evaluation.json").exists()
    assert (run_folder / "evaluation.yaml").exists()
    assert not (run_folder / "model.joblib").exists()


def test_final_fit_writes_clean_model_and_description(tmp_path: Path):
    input_path = tmp_path / "input.joblib"
    config_path = tmp_path / "train.yaml"
    _write_training_input(input_path)
    config_path.write_text(
        yaml.safe_dump(_training_config(input_path, tmp_path / "runs", mode="final_fit")),
        encoding="utf-8",
    )

    result = run_training_from_config(config_path)
    model_path = Path(result["model_path"])
    artifact = joblib.load(model_path)
    description = yaml.safe_load(
        (model_path.parent / "model_description.yaml").read_text(encoding="utf-8")
    )
    evaluation = yaml.safe_load(
        (model_path.parent / "evaluation.yaml").read_text(encoding="utf-8")
    )

    assert artifact["kind"] == "aramina_training_artifact"
    assert "created_at" not in artifact
    assert set(artifact["models"]) == {PRODUCT_MODEL_NAME}
    assert artifact["model_identity"]["name"] == PRODUCT_MODEL_NAME
    assert artifact["model_definition_yaml"]
    assert artifact["training_config_yaml"]
    assert artifact["historical_preprocessing_yaml"]
    assert artifact["prediction_preprocessing_yaml"]
    assert artifact["prediction_contract_yaml"]
    assert artifact["reproducibility"]["source_h5"]["sha256"] == "abc"
    tra_policy = artifact["models"][PRODUCT_MODEL_NAME]["tissue_risk_assessment"]
    assert tra_policy["contract"] == "aramina_tra_v0_2"
    assert tra_policy["decision_threshold"] == pytest.approx(
        artifact["models"][PRODUCT_MODEL_NAME]["thresholds"]["threshold_target"]
    )
    assert tra_policy["calibration"]["method"] in {
        "patient_safe_oof_decision_stability",
        "fixed_logit_margin_without_oof_calibration",
    }
    assert tra_policy["calibration"]["target_cases"] > 0
    assert tra_policy["logit_margin_boundaries"]["tra_2_to_3"] == 0.0
    assert artifact["evaluation"]["protocol"] == {
        "method": "repeated_stratified_kfold",
        "folds": 5,
        "repeats": 20,
        "random_seed": 42,
    }
    assert artifact["evaluation"]["summary"][0]["splits"] == 100
    performance = artifact["model_performance"]
    assert performance["evaluation_method"] == "repeated_stratified_kfold"
    assert performance["folds"] == 5
    assert performance["repeats"] == 20
    assert set(performance["held_out_metrics"]) == {
        "roc_auc",
        "sensitivity",
        "specificity",
    }
    assert artifact["final_fit_training_metrics"]["evaluation_status"] == (
        "in_sample_not_independent"
    )
    assert set(artifact["final_fit_training_metrics"]) >= {
        "roc_auc",
        "sensitivity",
        "specificity",
        "true_positives",
        "true_negatives",
        "false_positives",
        "false_negatives",
    }
    assert not (model_path.parent / "model_performance.yaml").exists()
    for filename in (
        "preprocessing_config.yaml",
        "prediction_preprocessing_config.yaml",
        "training_config.yaml",
    ):
        assert (model_path.parent / filename).is_file()
    assert not (model_path.parent / "model_performance.json").exists()
    reproducibility = artifact["reproducibility"]
    assert reproducibility["contract"] == "aramina_reproducibility_v0_1"
    assert reproducibility["reproduction_mode"] == "preprocessed_artifact_train"
    assert reproducibility["source_h5"]["sha256"] == "abc"
    assert reproducibility["source_h5"]["filename"] == "unknown"
    assert reproducibility["configs"]["training_yaml"] == artifact["training_config_yaml"]
    assert reproducibility["checksums"]["training_yaml_sha256"]
    assert "training_config" not in artifact
    assert "training_config_sha256" not in artifact
    assert "split_predictions" not in artifact
    assert description["model"]["id"] == result["model_id"]
    assert description["model"]["artifact_sha256"]
    assert description["model_performance"]["evaluation_method"] == (
        performance["evaluation_method"]
    )
    assert description["final_fit_training_metrics"]["evaluation_status"] == (
        "in_sample_not_independent"
    )
    assert description["final_fit_training_metrics"]["target_cases"] == 30
    assert set(description["final_fit_training_metrics"]) >= {
        "roc_auc",
        "sensitivity",
        "specificity",
    }
    assert "model_performance_files" not in description
    assert "kind" not in description
    assert "selected_model" not in description
    assert set(description["feature_schema"]) == {"final_model"}
    assert description["model_summary"]["architecture"] == {
        "stage_1": "target_xrd_profile_fpca30_logistic_regression",
        "stage_2": "age_and_optional_symmetry_refinement",
        "symmetry_behavior": "neutralized_unless_2_valid_measurements_per_breast_and_finite_core4_features",
    }
    assert description["model_summary"]["profile_encoder"] == {
        "type": "discrete_fpca",
        "input_q_bins": 256,
        "components": 30,
        "fit_scope": "fold_local_during_evaluation_train_all_for_final_fit",
    }
    assert (
        description["model_summary"]["lr1_profile_model"]["steps"]["fpca"][
            "n_components"
        ]
        == 30
    )
    assert (
        description["model_summary"]["lr1_profile_model"]["steps"][
            "profile_shape"
        ]["expected_features"]
        == 256
    )
    assert (
        description["model_summary"]["lr1_profile_model"]["steps"]["logreg"][
            "classes"
        ]
        == ["BENIGN", "CANCER"]
    )
    assert set(description["evaluation_artifacts"]) == {
        "summary",
        "metrics",
        "predictions",
    }
    assert description["evaluation_artifacts"] == {
        "summary": "evaluation.yaml",
        "metrics": "evaluation_metrics.csv",
        "predictions": "evaluation_predictions.csv",
    }
    assert evaluation["output_type"] == "aramina_evaluation_artifact"
    assert "kind" not in evaluation
    assert evaluation["model"] == description["model"]
    assert evaluation["threshold_selection"] == "train_fold_target_sensitivity"
    assert evaluation["training_config_sha256"]
    assert evaluation["decision_threshold"]["id"] == "target_sensitivity_0_95"
    assert evaluation["decision_threshold"]["value"] == pytest.approx(
        round(
            artifact["models"][PRODUCT_MODEL_NAME]["thresholds"]["threshold_target"],
            5,
        )
    )
    assert "evaluation_view" not in evaluation["metric_summary"][0]
    assert "rows" not in evaluation["dataset_summary"][0]
    assert evaluation["dataset_summary"][0]["measurements"] == 120
    assert evaluation["dataset_summary"][0]["lr1_measurements"] == 60
    assert evaluation["files"] == {
        "metrics": "evaluation_metrics.csv",
        "predictions": "evaluation_predictions.csv",
    }
    assert not _has_unrounded_float(evaluation)
    assert not _has_unrounded_float(description)
    assert description["model_summary"]["final_model"]["type"] == (
        "GatedSymmetryLogistic"
    )
    assert set(description) == {
        "output_type",
        "version",
        "model",
        "model_summary",
        "model_joblib",
        "model_performance",
        "final_fit_training_metrics",
        "decision_thresholds",
        "feature_schema",
        "dataset_summary",
        "evaluation_artifacts",
        "clinical_stage",
    }
    assert set(evaluation) == {
        "output_type",
        "version",
        "created_at",
        "model",
        "threshold_selection",
        "target_sensitivity",
        "training_config_sha256",
        "decision_threshold",
        "dataset_summary",
        "metric_summary",
        "files",
    }


def test_train_on_all_can_skip_evaluation_artifacts(tmp_path: Path):
    input_path = tmp_path / "input.joblib"
    config_path = tmp_path / "train.yaml"
    _write_training_input(input_path)
    config = _training_config(input_path, tmp_path / "runs", mode="final_fit")
    config["run"] = {"evaluation": False, "train_on_all": True}
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    result = run_training_from_config(config_path)
    artifact = joblib.load(result["model_path"])

    assert artifact["evaluation"]["requested"] is False
    assert artifact["evaluation"]["summary"] == []
    assert artifact["evaluation"]["artifacts"] == {}
    assert artifact["model_performance"]["evaluation_available"] is False
    assert artifact["model_performance"]["held_out_metrics"] == {}
    assert not (Path(result["run_folder"]) / "model_performance.yaml").exists()
    assert not (Path(result["run_folder"]) / "evaluation.joblib").exists()


def _has_unrounded_float(value: object) -> bool:
    if isinstance(value, dict):
        return any(_has_unrounded_float(item) for item in value.values())
    if isinstance(value, list):
        return any(_has_unrounded_float(item) for item in value)
    return isinstance(value, float) and value != round(value, 5)


def test_train_cli_lists_and_describes_models(capsys):
    assert main(["train", "--list-models"]) == 0
    assert PRODUCT_MODEL_NAME in capsys.readouterr().out

    assert main(["train", "--describe-model", PRODUCT_MODEL_NAME]) == 0
    assert "lr2_logreg_c: 0.3" in capsys.readouterr().out


def test_bilateral_biopsy_creates_two_target_cases_in_one_patient_safe_split():
    frame = _patient_training_frame()
    frame.loc[(frame["patientId"] == "P00") & (frame["side"] == "Right"), "biopsy"] = True
    builder = PatientModelInputBuilder(lr1_row_policy="all_rows", random_state=7)
    feature_table = builder.fit_transform(frame)

    bilateral = feature_table.query("patientId == 'P00'")
    assert set(bilateral["target_side"]) == {"Left", "Right"}
    assert bilateral["target_case_id"].nunique() == 2

    for train_index, test_index in _patient_split_pairs(
        mode="stratified_kfold",
        base_features=feature_table,
        y_patients=feature_table["label"].to_numpy(dtype=int),
        n_splits=5,
        n_repeats=20,
        random_state=42,
    ):
        train_patients = set(feature_table.iloc[train_index]["patientId"])
        test_patients = set(feature_table.iloc[test_index]["patientId"])
        assert ("P00" in train_patients) != ("P00" in test_patients)


def test_logit_average_probability_preserves_consistent_evidence():
    scores = np.array([0.95, 0.95, 0.50])

    assert float(np.mean(scores)) == pytest.approx(0.80)
    assert _logit_average_probability(scores) == pytest.approx(0.877, abs=0.001)


def test_lr1_single_class_error_reports_the_retained_cohort():
    frame = _patient_training_frame()
    frame["product_status_group"] = "BENIGN"

    with pytest.raises(ValueError, match=r"rows=120.*label_counts=\{'BENIGN': 120\}"):
        _lr1_training_rows(
            frame,
            label_column="product_status_group",
            biopsy_column="biopsy",
            lr1_row_policy="all_rows",
        )


def test_symmetry_refinement_requires_two_measurements_per_breast():
    frame = _patient_training_frame().query("patientId == 'P00'").copy()
    q = np.linspace(2.0, 23.0, 100)
    for index, row in frame.iterrows():
        measurement_index = int(str(row["measurementId"])[-1])
        shift = 0.8 if row["side"] == "Left" else -0.4
        frame.at[index, "q_range"] = q
        frame.at[index, "radial_profile_data"] = (
            shift + np.sin(q / 3.0) + measurement_index * 0.01
        )
    paired = target_contralateral_symmetry_features(
        frame,
        profile_column="radial_profile_data",
        q_column="q_range",
        side_column="side",
        target_side_norm="LEFT",
        contralateral_side_norm="RIGHT",
    )
    assert paired["symmetry_available"] == 1

    unpaired = frame.drop(frame.query("side == 'Right'").index[1])
    neutral = target_contralateral_symmetry_features(
        unpaired,
        profile_column="radial_profile_data",
        q_column="q_range",
        side_column="side",
        target_side_norm="LEFT",
        contralateral_side_norm="RIGHT",
    )
    assert neutral["symmetry_available"] == 0
    assert neutral["symmetry_reason"] == "fewer_than_2_valid_measurements_per_breast"
    assert {neutral[column] for column in SK_SYMMETRY_COLUMNS} == {0.0}


def test_noncomputable_core4_neutralizes_symmetry_without_using_zero_as_data():
    frame = _patient_training_frame().query("patientId == 'P00'").copy()
    q = np.linspace(2.0, 6.0, 100)
    frame["q_range"] = [q for _ in range(len(frame))]
    frame["radial_profile_data"] = [np.sin(q / 3.0) for _ in range(len(frame))]

    features = target_contralateral_symmetry_features(
        frame,
        profile_column="radial_profile_data",
        q_column="q_range",
        side_column="side",
        target_side_norm="LEFT",
        contralateral_side_norm="RIGHT",
    )

    assert features["symmetry_available"] == 0
    assert features["symmetry_reason"] == "sk_core4_not_computable"
    assert {features[column] for column in SK_SYMMETRY_COLUMNS} == {0.0}


def test_final_fit_rejects_plain_dataframe_without_preprocessing_lineage(tmp_path: Path):
    input_path = tmp_path / "plain.joblib"
    joblib.dump(_patient_training_frame(), input_path)
    config_path = tmp_path / "train.yaml"
    config_path.write_text(
        yaml.safe_dump(_training_config(input_path, tmp_path / "runs", mode="final_fit")),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="preprocessing artifact joblib"):
        run_training_from_config(config_path)


def test_training_output_contract_examples_are_complete():
    example_root = Path(__file__).parents[1] / "contracts" / "training" / "examples"
    description = yaml.safe_load(
        (example_root / "model_description.yaml").read_text(encoding="utf-8")
    )
    evaluation = yaml.safe_load(
        (example_root / "evaluation.yaml").read_text(encoding="utf-8")
    )

    assert {
        "output_type",
        "model",
        "model_summary",
        "model_performance",
        "final_fit_training_metrics",
        "decision_thresholds",
        "dataset_summary",
        "evaluation_artifacts",
    }.issubset(description)
    assert (
        description["model_summary"]["symmetry_feature_contract"]
        == "aramina_sk_symmetry_v0_2"
    )
    assert {
        "roc_auc",
        "sensitivity",
        "specificity",
        "calibration_intercept",
        "calibration_slope",
        "true_positives",
        "true_negatives",
        "false_negatives",
        "false_positives",
    }.issubset(description["final_fit_training_metrics"])
    assert {
        "output_type",
        "model",
        "training_config_sha256",
        "dataset_summary",
        "metric_summary",
        "files",
    }.issubset(evaluation)
    assert evaluation["files"] == {
        "metrics": "evaluation_metrics.csv",
        "predictions": "evaluation_predictions.csv",
    }
