"""Compare radial bin counts and pyFAI pixel-splitting methods."""

from __future__ import annotations

import argparse
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
import subprocess
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xrd_preprocessing
import yaml
from xrd_preprocessing import (
    build_pipeline_from_config,
    load_preprocessing_artifact,
    load_preprocessing_config,
    pipeline_spec_sha256,
    resolve_pipeline_spec,
    save_preprocessing_artifact,
)

from aramina.training import (
    _effective_training_config,
    load_training_config,
    train_m2q_model_artifact,
)
from aramina.training_config import resolve_model_definition


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs" / "npt_method_sweep"
PREPROCESSING_CONFIG = (
    ROOT / "config/preprocessing/config_preprocessing_biopsy_patients_v0_2.yaml"
)
TRAINING_CONFIG = (
    ROOT / "config/training/config_training_target_breast_risk_v0_1.yaml"
)
DEFAULT_H5 = (
    ROOT.parent
    / "eos_play/jupyter_notebooks/Clinical_trials/data/"
    "product-aramis-data/combined_archive.h5"
)
IDENTITY_COLUMNS = ["patientId", "specimenId", "side", "position", "started_at"]
VARIANTS = (
    ("npt100_bbox", 100, ("bbox", "csr", "cython")),
    ("npt100_no", 100, ("no", "csr", "cython")),
    ("npt100_full", 100, ("full", "csr", "cython")),
    ("npt150_bbox", 150, ("bbox", "csr", "cython")),
    ("npt200_bbox", 200, ("bbox", "csr", "cython")),
    ("npt250_bbox", 250, ("bbox", "csr", "cython")),
    ("npt256_bbox", 256, ("bbox", "csr", "cython")),
    ("npt512_bbox", 512, ("bbox", "csr", "cython")),
    ("npt512_no", 512, ("no", "csr", "cython")),
    ("npt512_full", 512, ("full", "csr", "cython")),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-h5", type=Path, default=DEFAULT_H5)
    parser.add_argument(
        "--force-preprocessing",
        action="store_true",
        help="Rebuild preprocessing artifacts that already exist.",
    )
    return parser.parse_args()


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_sha(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        text=True,
    ).strip()


def _research_lineage(
    *,
    name: str,
    npt: int,
    method: tuple[str, str, str],
) -> dict[str, Any]:
    xrd_root = Path(xrd_preprocessing.__file__).resolve().parents[2]
    return {
        "contract": "aramina_research_preprocessing_lineage_v0_1",
        "status": "research_only_not_product_compatible",
        "product": {
            "name": "aramina_biopsy_patients_model_input",
            "route": "training",
        },
        "integration_experiment": {
            "variant": name,
            "npt": int(npt),
            "method": list(method),
        },
        "xrd_preprocessing": {
            "version": version("xrd-preprocessing"),
            "git_sha": _git_sha(xrd_root),
            "branch": "experiment/npt-method-sweep",
        },
    }


def _identity(frame: pd.DataFrame) -> pd.Series:
    return frame.loc[:, IDENTITY_COLUMNS].astype(str).agg("\x1f".join, axis=1)


def _integration_step(config: dict[str, Any]) -> dict[str, Any]:
    for step in config["pipeline"]["steps"]:
        if step.get("name") == "azimuthal_integration":
            return step
    raise ValueError("Missing azimuthal_integration pipeline step.")


def _variant_config(
    *,
    input_h5: Path,
    name: str,
    npt: int,
    method: tuple[str, str, str],
    output_path: Path,
) -> dict[str, Any]:
    config = load_preprocessing_config(PREPROCESSING_CONFIG)
    config["integration"]["npt"] = int(npt)
    config["io"]["input_h5_path"] = str(input_h5)
    config["io"]["output_joblib_path"] = str(output_path)
    config["aramina_preprocessing"]["name"] = (
        f"aramina_biopsy_patients_{name}_experiment"
    )
    config["aramina_preprocessing"]["route"] = "training"
    config["provenance"]["status"] = "research-only pyFAI integration sweep"
    config["provenance"]["canonical_location"] = str(Path(__file__).relative_to(ROOT))
    config["experiment"]["integration_variant"] = name
    config["experiment"]["integration_npt"] = int(npt)
    config["experiment"]["integration_method"] = list(method)
    _integration_step(config)["params"]["method"] = list(method)
    return config


def _preprocess_variant(
    *,
    input_h5: Path,
    input_h5_sha256: str,
    name: str,
    npt: int,
    method: tuple[str, str, str],
    force: bool,
) -> tuple[pd.DataFrame, dict[str, Any], Path]:
    folder = OUTPUT_DIR / name
    folder.mkdir(parents=True, exist_ok=True)
    artifact_path = folder / "preprocessing_full.joblib"
    if artifact_path.exists() and not force:
        artifact = load_preprocessing_artifact(artifact_path)
        return artifact["dataframe"], artifact, artifact_path

    config = _variant_config(
        input_h5=input_h5,
        name=name,
        npt=npt,
        method=method,
        output_path=artifact_path,
    )
    config_text = yaml.safe_dump(config, sort_keys=False)
    (folder / "preprocessing_resolved.yaml").write_text(
        config_text,
        encoding="utf-8",
    )
    print(f"[{name}] preprocessing")
    frame = build_pipeline_from_config(config, verbose=True).fit_transform(
        str(input_h5)
    )
    spec = resolve_pipeline_spec(config)
    artifact = save_preprocessing_artifact(
        frame,
        artifact_path,
        preprocessing_config_text=config_text,
        preprocessing_config=config,
        resolved_pipeline_spec=spec,
        pipeline_fingerprint=pipeline_spec_sha256(spec),
        metadata={
            "input_h5_sha256": input_h5_sha256,
            "research_experiment": "pyfai_npt_method_sweep",
            "integration_variant": name,
            "integration_npt": int(npt),
            "integration_method": list(method),
            "aramina_preprocessing_lineage": _research_lineage(
                name=name,
                npt=npt,
                method=method,
            ),
        },
    )
    return frame, artifact, artifact_path


def _filter_common_measurements(
    frame: pd.DataFrame,
    common_ids: set[str],
) -> pd.DataFrame:
    keys = _identity(frame)
    return frame.loc[keys.isin(common_ids)].copy().reset_index(drop=True)


def _preprocessing_summary(
    frame: pd.DataFrame,
    *,
    name: str,
    npt: int,
    method: tuple[str, str, str],
    common_rows: int,
) -> dict[str, Any]:
    q_steps = [
        float(np.median(np.diff(np.asarray(values, dtype=float))))
        for values in frame["q_range"]
    ]
    return {
        "variant": name,
        "npt": int(npt),
        "splitting": method[0],
        "algorithm": method[1],
        "implementation": method[2],
        "q_step_nm_inv": float(np.median(q_steps)),
        "retained_measurements": int(len(frame)),
        "common_measurements": int(common_rows),
        "retained_patients": int(frame["patientId"].astype(str).nunique()),
        "median_snr_db": float(frame["snr_db"].median()),
        "profile_gate_pass_rate": float(frame["radial_profile_value_pass"].mean()),
    }


def _train_variant(
    *,
    name: str,
    cohort: str,
    frame: pd.DataFrame,
    source_artifact: dict[str, Any],
    source_config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    variant_folder = OUTPUT_DIR / name
    folder = variant_folder if cohort == "common" else variant_folder / f"training_{cohort}"
    folder.mkdir(parents=True, exist_ok=True)
    result_path = folder / "result.yaml"
    artifact_path = folder / "training_artifact.joblib"
    if result_path.exists() and artifact_path.exists():
        return yaml.safe_load(result_path.read_text(encoding="utf-8")), joblib.load(
            folder / f"preprocessing_{cohort}.joblib"
        )

    model_input_path = folder / f"preprocessing_{cohort}.joblib"
    config_text = yaml.safe_dump(source_config, sort_keys=False)
    spec = resolve_pipeline_spec(source_config)
    method = tuple(source_config["experiment"]["integration_method"])
    lineage = source_artifact.get("metadata", {}).get(
        "aramina_preprocessing_lineage"
    ) or _research_lineage(
        name=name,
        npt=int(source_config["integration"]["npt"]),
        method=method,
    )
    common_artifact = save_preprocessing_artifact(
        frame,
        model_input_path,
        preprocessing_config_text=config_text,
        preprocessing_config=source_config,
        resolved_pipeline_spec=spec,
        pipeline_fingerprint=pipeline_spec_sha256(spec),
        metadata={
            **dict(source_artifact.get("metadata", {})),
            "modeling_cohort": cohort,
            "modeling_measurements": int(len(frame)),
            "aramina_preprocessing_lineage": lineage,
        },
    )

    public_config, _ = load_training_config(TRAINING_CONFIG)
    public_config["model"]["version"] = f"0.2.12-beta-{name}-experiment"
    model_definition = resolve_model_definition(public_config["model"]["name"])
    effective_config = _effective_training_config(public_config, model_definition)
    training_text = yaml.safe_dump(public_config, sort_keys=False)
    print(f"[{name}] 5-fold x20 evaluation and train-on-all")
    artifact = train_m2q_model_artifact(
        frame,
        config=effective_config,
        config_text=training_text,
        input_dataframe_joblib_path=model_input_path,
        preprocessing_artifact=common_artifact,
        prediction_preprocessing=None,
    )
    joblib.dump(artifact, artifact_path)
    artifact["split_metrics"].to_csv(folder / "evaluation_metrics.csv", index=False)
    artifact["split_predictions"].to_csv(
        folder / "evaluation_predictions.csv",
        index=False,
    )

    held_out = artifact["metric_summary"].iloc[0].to_dict()
    model_name = next(iter(artifact["models"]))
    train_all = dict(
        artifact["models"][model_name]["final_fit_training_metrics"]
    )
    result = {
        "variant": name,
        "held_out": held_out,
        "train_all": train_all,
    }
    result_path.write_text(
        yaml.safe_dump(result, sort_keys=False),
        encoding="utf-8",
    )
    return result, common_artifact


def _flat_result(
    result: dict[str, Any],
    preprocessing: dict[str, Any],
) -> dict[str, Any]:
    held = result["held_out"]
    final = result["train_all"]
    return {
        **preprocessing,
        "target_cases": int(final["target_cases"]),
        "kfold_roc_auc_mean": float(held["roc_auc_mean"]),
        "kfold_roc_auc_std": float(held["roc_auc_std"]),
        "kfold_pr_auc_mean": float(held["pr_auc_mean"]),
        "kfold_pr_auc_std": float(held["pr_auc_std"]),
        "kfold_sensitivity_mean": float(held["sensitivity_target_mean"]),
        "kfold_sensitivity_std": float(held["sensitivity_target_std"]),
        "kfold_specificity_mean": float(held["specificity_target_mean"]),
        "kfold_specificity_std": float(held["specificity_target_std"]),
        "kfold_brier_mean": float(held["brier_score_mean"]),
        "kfold_log_loss_mean": float(held["log_loss_mean"]),
        "train_all_threshold": float(final["decision_threshold"]),
        "train_all_roc_auc": float(final["roc_auc"]),
        "train_all_pr_auc": float(final["pr_auc"]),
        "train_all_sensitivity": float(final["sensitivity"]),
        "train_all_specificity": float(final["specificity"]),
        "train_all_balanced_accuracy": float(final["balanced_accuracy"]),
        "train_all_tp": int(final["true_positives"]),
        "train_all_tn": int(final["true_negatives"]),
        "train_all_fp": int(final["false_positives"]),
        "train_all_fn": int(final["false_negatives"]),
    }


def _plot_results(results: pd.DataFrame, *, cohort: str) -> None:
    labels = results["variant"].str.replace("npt", "", regex=False)
    x = np.arange(len(results))
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), constrained_layout=True)

    for metric, color in (
        ("roc_auc", "#2f6fbb"),
        ("sensitivity", "#c43b3b"),
        ("specificity", "#2f8f5b"),
    ):
        axes[0].errorbar(
            x,
            results[f"kfold_{metric}_mean"],
            yerr=results[f"kfold_{metric}_std"],
            marker="o",
            capsize=3,
            linewidth=1.8,
            color=color,
            label=metric.replace("_", " ").title(),
        )
    axes[0].axhline(0.5, color="#888888", linestyle="--", linewidth=1)
    axes[0].set_title("Patient-safe repeated 5-fold x20")
    axes[0].set_ylabel("Metric")
    axes[0].set_xticks(x, labels, rotation=35, ha="right")
    axes[0].set_ylim(0.2, 1.0)
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=0.25)

    for metric, color in (
        ("roc_auc", "#2f6fbb"),
        ("sensitivity", "#c43b3b"),
        ("specificity", "#2f8f5b"),
    ):
        axes[1].plot(
            x,
            results[f"train_all_{metric}"],
            marker="o",
            linewidth=1.8,
            color=color,
            label=metric.replace("_", " ").title(),
        )
    axes[1].axhline(0.5, color="#888888", linestyle="--", linewidth=1)
    axes[1].set_title("Train-on-all, in-sample")
    axes[1].set_xticks(x, labels, rotation=35, ha="right")
    axes[1].set_ylim(0.2, 1.0)
    axes[1].legend(frameon=False)
    axes[1].grid(axis="y", alpha=0.25)

    fig.suptitle(
        f"Aramina pyFAI radial-resolution and splitting sweep ({cohort} cohort)"
    )
    fig.savefig(
        EXPERIMENT_DIR / f"npt_method_sweep_metrics_{cohort}.png",
        dpi=180,
    )
    plt.close(fig)


def _paired_fold_deltas() -> pd.DataFrame:
    baseline = pd.read_csv(
        OUTPUT_DIR / "npt100_bbox" / "evaluation_metrics.csv"
    ).set_index("split_id")
    rows: list[dict[str, Any]] = []
    for name, npt, method in VARIANTS:
        current = pd.read_csv(
            OUTPUT_DIR / name / "evaluation_metrics.csv"
        ).set_index("split_id")
        rows.append(
            {
                "variant": name,
                "npt": npt,
                "splitting": method[0],
                "delta_roc_auc_mean": float(
                    (current["roc_auc"] - baseline["roc_auc"]).mean()
                ),
                "delta_roc_auc_std": float(
                    (current["roc_auc"] - baseline["roc_auc"]).std()
                ),
                "fraction_folds_with_higher_roc_auc": float(
                    (current["roc_auc"] > baseline["roc_auc"]).mean()
                ),
                "delta_sensitivity_mean": float(
                    (
                        current["sensitivity_target"]
                        - baseline["sensitivity_target"]
                    ).mean()
                ),
                "delta_specificity_mean": float(
                    (
                        current["specificity_target"]
                        - baseline["specificity_target"]
                    ).mean()
                ),
                "delta_brier_mean": float(
                    (current["brier_score"] - baseline["brier_score"]).mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = _parse_args()
    input_h5 = args.input_h5.expanduser().resolve()
    if not input_h5.exists():
        raise FileNotFoundError(f"Missing source H5: {input_h5}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    input_h5_sha256 = _file_sha256(input_h5)

    frames: dict[str, pd.DataFrame] = {}
    artifacts: dict[str, dict[str, Any]] = {}
    configs: dict[str, dict[str, Any]] = {}
    for name, npt, method in VARIANTS:
        frame, artifact, _ = _preprocess_variant(
            input_h5=input_h5,
            input_h5_sha256=input_h5_sha256,
            name=name,
            npt=npt,
            method=method,
            force=args.force_preprocessing,
        )
        frames[name] = frame
        artifacts[name] = artifact
        configs[name] = _variant_config(
            input_h5=input_h5,
            name=name,
            npt=npt,
            method=method,
            output_path=OUTPUT_DIR / name / "preprocessing_full.joblib",
        )

    common_ids = set.intersection(
        *[set(_identity(frame)) for frame in frames.values()]
    )
    if not common_ids:
        raise ValueError("Integration variants have no common measurements.")

    common_results: list[dict[str, Any]] = []
    full_results: list[dict[str, Any]] = []
    preprocessing_rows: list[dict[str, Any]] = []
    for name, npt, method in VARIANTS:
        common_frame = _filter_common_measurements(frames[name], common_ids)
        preprocessing = _preprocessing_summary(
            frames[name],
            name=name,
            npt=npt,
            method=method,
            common_rows=len(common_frame),
        )
        preprocessing_rows.append(preprocessing)
        common_result, _ = _train_variant(
            name=name,
            cohort="common",
            frame=common_frame,
            source_artifact=artifacts[name],
            source_config=configs[name],
        )
        full_result, _ = _train_variant(
            name=name,
            cohort="full",
            frame=frames[name],
            source_artifact=artifacts[name],
            source_config=configs[name],
        )
        common_results.append(_flat_result(common_result, preprocessing))
        full_results.append(_flat_result(full_result, preprocessing))

    common_df = pd.DataFrame(common_results)
    full_df = pd.DataFrame(full_results)
    preprocessing_df = pd.DataFrame(preprocessing_rows)
    common_df.to_csv(
        EXPERIMENT_DIR / "npt_method_sweep_results_common.csv",
        index=False,
    )
    full_df.to_csv(
        EXPERIMENT_DIR / "npt_method_sweep_results_full.csv",
        index=False,
    )
    preprocessing_df.to_csv(
        EXPERIMENT_DIR / "npt_method_sweep_preprocessing.csv",
        index=False,
    )
    paired_deltas = _paired_fold_deltas()
    paired_deltas.to_csv(
        EXPERIMENT_DIR / "npt_method_sweep_paired_fold_deltas.csv",
        index=False,
    )
    summary = {
        "experiment": "Aramina pyFAI npt and pixel-splitting sweep",
        "status": "research only",
        "source_h5": {
            "path": str(input_h5),
            "sha256": input_h5_sha256,
        },
        "controls": {
            "common_measurement_cohort": True,
            "common_measurements": int(len(common_ids)),
            "evaluation": "patient-safe repeated stratified 5-fold x20",
            "random_seed": 42,
            "q_range_nm_inv": [2.0, 23.0],
            "error_model": "poisson",
        },
        "controlled_common_cohort_results": common_df.to_dict(orient="records"),
        "end_to_end_full_cohort_results": full_df.to_dict(orient="records"),
        "paired_fold_deltas_vs_npt100_bbox": paired_deltas.to_dict(
            orient="records"
        ),
    }
    (EXPERIMENT_DIR / "npt_method_sweep_results.yaml").write_text(
        yaml.safe_dump(summary, sort_keys=False),
        encoding="utf-8",
    )
    _plot_results(common_df, cohort="common")
    _plot_results(full_df, cohort="full")
    print("\nCONTROLLED COMMON COHORT\n")
    print(common_df.to_string(index=False))
    print("\nEND-TO-END FULL COHORT\n")
    print(full_df.to_string(index=False))


if __name__ == "__main__":
    main()
