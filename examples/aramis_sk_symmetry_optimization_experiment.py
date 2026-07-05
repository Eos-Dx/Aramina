"""Explore stronger SK symmetry features for Aramis research-draft models."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import warnings

import joblib
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from aramis.modeling import LABEL_MAP, profile_matrix
from aramis.training import _logit_average_probability

ROOT = Path(__file__).resolve().parents[1]
INPUT_JOBLIB = (
    ROOT / "examples/outputs/model_input/aramis_biopsy_patients_model_input_v0_1.joblib"
)
OUTPUT_DIR = ROOT / "docs/modeling/results"

PROFILE_COLUMN = "radial_profile_data"
Q_COLUMN = "q_range"
LABEL_COLUMN = "product_status_group"
GROUP_COLUMN = "patientId"
SIDE_COLUMN = "side"
BIOPSY_COLUMN = "biopsy"
AGE_COLUMN = "age"

TARGET_SENSITIVITY = 0.95
RANDOM_STATE = 42
N_SPLITS = 50
TEST_SIZE = 0.30
KFOLD_SPLITS = 5
EPS = 1e-9

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message="'penalty' was deprecated.*",
)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="Inconsistent values: penalty=.*",
)

Q_WINDOWS = {
    "q02_06": (2.0, 6.0),
    "q06_10": (6.0, 10.0),
    "q10_15": (10.0, 15.0),
    "q13_16": (13.0, 16.0),
    "q15_23": (15.0, 23.0),
    "q02_23": (2.0, 23.0),
}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = _load_dataframe(INPUT_JOBLIB)
    feature_df = _build_feature_table(df)
    feature_df = _add_ratio_features(feature_df)

    single_feature_auc = _single_feature_auc(feature_df)
    symmetry_summary = _evaluate_symmetry_only(feature_df)
    best_set = _best_symmetry_feature_set(symmetry_summary)
    fusion_summary = _evaluate_profile_plus_symmetry(df, feature_df, best_set)

    feature_path = OUTPUT_DIR / "sk_symmetry_optimization_features_v0_1.csv"
    single_path = OUTPUT_DIR / "sk_symmetry_optimization_single_features_v0_1.csv"
    symmetry_path = OUTPUT_DIR / "sk_symmetry_optimization_symmetry_only_v0_1.csv"
    fusion_path = OUTPUT_DIR / "sk_symmetry_optimization_profile_plus_sk_v0_1.csv"

    feature_df.to_csv(feature_path, index=False)
    single_feature_auc.to_csv(single_path, index=False)
    symmetry_summary.to_csv(symmetry_path, index=False)
    fusion_summary.to_csv(fusion_path, index=False)
    _write_markdown_summary(
        feature_df=feature_df,
        single_feature_auc=single_feature_auc,
        symmetry_summary=symmetry_summary,
        fusion_summary=fusion_summary,
        best_set=best_set,
    )

    print("Feature table")
    print(feature_df["label_name"].value_counts().to_string())
    print("\nTop single symmetry features")
    print(single_feature_auc.head(12).to_string(index=False))
    print("\nSymmetry-only best rows")
    print(
        symmetry_summary.sort_values(
            ["mode", "roc_auc_mean", "specificity_mean"],
            ascending=[True, False, False],
        )
        .groupby("mode")
        .head(8)
        .to_string(index=False)
    )
    print("\nProfile plus SK")
    print(fusion_summary.to_string(index=False))
    print(f"\nBest symmetry set: {best_set}")


def _load_dataframe(path: Path) -> pd.DataFrame:
    artifact = joblib.load(path)
    return artifact["dataframe"].copy() if isinstance(artifact, dict) else artifact.copy()


def _build_feature_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for patient_id, patient_df in df.groupby(GROUP_COLUMN, sort=True):
        label = _patient_label(patient_df)
        if label is None:
            continue
        target_side = _target_side(patient_df)
        contralateral_side = _contralateral_side(patient_df, target_side)
        if target_side is None or contralateral_side is None:
            continue
        row = {
            "patientId": str(patient_id),
            "label": int(label),
            "label_name": "CANCER" if int(label) == 1 else "BENIGN",
            "target_side": target_side,
            "contralateral_side": contralateral_side,
            "age": _numeric_median(patient_df, AGE_COLUMN),
            "age_available": int(_has_numeric(patient_df, AGE_COLUMN)),
        }
        row.update(
            _all_window_features(
                patient_df,
                target_side=target_side,
                contralateral_side=contralateral_side,
            )
        )
        rows.append(row)
    out = pd.DataFrame(rows)
    if out["label"].nunique() != 2:
        raise ValueError("Feature table must contain BENIGN and CANCER.")
    return out


def _patient_label(patient_df: pd.DataFrame) -> int | None:
    labels = patient_df[LABEL_COLUMN].map(LABEL_MAP).dropna().astype(int)
    if labels.empty:
        return None
    return int((labels == 1).any())


def _target_side(patient_df: pd.DataFrame) -> str | None:
    work = patient_df[[SIDE_COLUMN, LABEL_COLUMN]].copy()
    work["_side"] = work[SIDE_COLUMN].map(_normalize_side)
    work["_label"] = work[LABEL_COLUMN].map(LABEL_MAP)
    if BIOPSY_COLUMN in patient_df.columns:
        work["_biopsy"] = _boolean_series(patient_df[BIOPSY_COLUMN])
    else:
        work["_biopsy"] = False
    for mask in (
        work["_biopsy"] & (work["_label"] == 1),
        work["_biopsy"] & (work["_label"] == 0),
        work["_label"] == 1,
        work["_label"] == 0,
    ):
        sides = sorted(work.loc[mask, "_side"].dropna().unique())
        if sides:
            return str(sides[0])
    return None


def _contralateral_side(patient_df: pd.DataFrame, target_side: str | None) -> str | None:
    if target_side is None:
        return None
    sides = sorted(
        side
        for side in patient_df[SIDE_COLUMN].map(_normalize_side).dropna().unique()
        if side != target_side
    )
    return str(sides[0]) if sides else None


def _all_window_features(
    patient_df: pd.DataFrame,
    *,
    target_side: str,
    contralateral_side: str,
) -> dict[str, float | int]:
    out: dict[str, float | int] = {}
    for name, q_roi in Q_WINDOWS.items():
        metrics = _side_mean_metrics(
            patient_df,
            target_side=target_side,
            contralateral_side=contralateral_side,
            q_roi=q_roi,
        )
        out.update(_window_features(name, metrics))
    out["target_measurements"] = int(
        (patient_df[SIDE_COLUMN].map(_normalize_side) == target_side).sum()
    )
    out["contralateral_measurements"] = int(
        (patient_df[SIDE_COLUMN].map(_normalize_side) == contralateral_side).sum()
    )
    out["min_measurements_per_breast"] = int(
        min(out["target_measurements"], out["contralateral_measurements"])
    )
    return out


def _side_mean_metrics(
    df: pd.DataFrame,
    *,
    target_side: str,
    contralateral_side: str,
    q_roi: tuple[float, float],
) -> dict[str, np.ndarray] | None:
    target_profiles: list[np.ndarray] = []
    contralateral_profiles: list[np.ndarray] = []
    q_common: np.ndarray | None = None
    for row in df.itertuples(index=False):
        side = _normalize_side(getattr(row, SIDE_COLUMN))
        if side not in {target_side, contralateral_side}:
            continue
        q = np.asarray(getattr(row, Q_COLUMN), dtype=float).ravel()
        y = np.asarray(getattr(row, PROFILE_COLUMN), dtype=float).ravel()
        q, y = _profile_roi(q, y, q_roi)
        y = _normalize_profile_near_minimum(q, _smooth_profile(y))
        if q_common is None:
            q_common = q
        y_common = np.interp(q_common, q, y)
        if side == target_side:
            target_profiles.append(y_common)
        else:
            contralateral_profiles.append(y_common)
    if q_common is None or not target_profiles or not contralateral_profiles:
        return None
    target = np.vstack(target_profiles)
    contralateral = np.vstack(contralateral_profiles)
    return {
        "q": q_common,
        "mu_t": np.mean(target, axis=0),
        "mu_c": np.mean(contralateral, axis=0),
        "std_t": _profile_std(target),
        "std_c": _profile_std(contralateral),
    }


def _window_features(
    name: str,
    metrics: dict[str, np.ndarray] | None,
) -> dict[str, float]:
    columns = {
        f"sk_{name}_meanrms": 0.0,
        f"sk_{name}_weightedrms": 0.0,
        f"sk_{name}_sigma_target": 0.0,
        f"sk_{name}_sigma_contralateral": 0.0,
        f"sk_{name}_mahalanobis": 0.0,
        f"sk_{name}_cosine": 0.0,
        f"sk_{name}_wasserstein": 0.0,
        f"sk_{name}_peak": 0.0,
    }
    if metrics is None:
        return columns
    q = metrics["q"]
    mu_t = metrics["mu_t"]
    mu_c = metrics["mu_c"]
    std_t = metrics["std_t"]
    std_c = metrics["std_c"]
    mask = np.isfinite(q)
    columns.update(
        {
            f"sk_{name}_meanrms": _finite_or_zero(_rms_difference(mu_t, mu_c, mask)),
            f"sk_{name}_weightedrms": _finite_or_zero(
                _weighted_rms_difference(mu_t, mu_c, std_t, std_c, mask)
            ),
            f"sk_{name}_sigma_target": _finite_or_zero(_sigma_rms(std_t, mask)),
            f"sk_{name}_sigma_contralateral": _finite_or_zero(
                _sigma_rms(std_c, mask)
            ),
            f"sk_{name}_mahalanobis": _finite_or_zero(
                _mahalanobis_difference(mu_t, mu_c, std_t, std_c, mask)
            ),
            f"sk_{name}_cosine": _finite_or_zero(_cosine_distance(mu_t, mu_c)),
            f"sk_{name}_wasserstein": _finite_or_zero(
                _profile_wasserstein(q, mu_t, mu_c)
            ),
            f"sk_{name}_peak": _finite_or_zero(_peak_intensity(q, mu_t, mu_c)),
        }
    )
    return columns


def _add_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    pairs = [
        ("q15_23", "q10_15"),
        ("q15_23", "q06_10"),
        ("q13_16", "q10_15"),
        ("q10_15", "q06_10"),
    ]
    metrics = ["meanrms", "weightedrms", "mahalanobis", "wasserstein", "cosine"]
    for high, low in pairs:
        for metric in metrics:
            hi = f"sk_{high}_{metric}"
            lo = f"sk_{low}_{metric}"
            if hi not in out or lo not in out:
                continue
            out[f"sk_ratio_{high}_to_{low}_{metric}"] = out[hi] / (out[lo] + EPS)
            out[f"sk_logdiff_{high}_minus_{low}_{metric}"] = np.log1p(
                np.maximum(out[hi], 0.0)
            ) - np.log1p(np.maximum(out[lo], 0.0))
    out["sk_target_sigma_over_contralateral_q10_15"] = out[
        "sk_q10_15_sigma_target"
    ] / (out["sk_q10_15_sigma_contralateral"] + EPS)
    out["sk_target_sigma_over_contralateral_q15_23"] = out[
        "sk_q15_23_sigma_target"
    ] / (out["sk_q15_23_sigma_contralateral"] + EPS)
    return out


def _feature_sets(df: pd.DataFrame) -> dict[str, list[str]]:
    base = [
        "sk_q10_15_meanrms",
        "sk_q10_15_weightedrms",
        "sk_q10_15_sigma_target",
        "sk_q10_15_sigma_contralateral",
        "sk_q10_15_mahalanobis",
        "sk_q15_23_meanrms",
        "sk_q15_23_weightedrms",
        "sk_q15_23_sigma_target",
        "sk_q15_23_sigma_contralateral",
        "sk_q15_23_mahalanobis",
        "sk_q13_16_peak",
        "sk_q02_23_wasserstein",
        "sk_q02_23_cosine",
    ]
    all_windows = [c for c in df.columns if c.startswith("sk_q")]
    ratios = [c for c in df.columns if c.startswith(("sk_ratio_", "sk_logdiff_"))]
    reliability = [
        "target_measurements",
        "contralateral_measurements",
        "min_measurements_per_breast",
    ]
    sigma_ratios = [c for c in df.columns if c.startswith("sk_target_sigma_over_")]
    return {
        "sk_base": base,
        "sk_ratios": [*base, *ratios, *sigma_ratios],
        "sk_windows": all_windows,
        "sk_windows_ratios": [*all_windows, *ratios, *sigma_ratios],
        "sk_windows_reliability": [*all_windows, *reliability],
        "sk_all": [*all_windows, *ratios, *sigma_ratios, *reliability],
    }


def _models() -> dict[str, Pipeline]:
    return {
        "LR_L2": _logistic("l2"),
        "LR_L1": _logistic("l1"),
        "LR_elastic": _logistic("elasticnet"),
        "SVM_poly2": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    SVC(
                        C=1.0,
                        class_weight="balanced",
                        degree=2,
                        gamma="scale",
                        kernel="poly",
                    ),
                ),
            ]
        ),
        "RF_depth2": _random_forest(2),
        "RF_depth3": _random_forest(3),
    }


def _logistic(penalty: str) -> Pipeline:
    kwargs: dict[str, Any] = {
        "class_weight": "balanced",
        "max_iter": 5000,
        "random_state": RANDOM_STATE,
    }
    if penalty == "elasticnet":
        kwargs.update({"penalty": "elasticnet", "solver": "saga", "l1_ratio": 0.5})
    elif penalty == "l1":
        kwargs.update({"penalty": "l1", "solver": "liblinear"})
    else:
        kwargs.update({"penalty": "l2", "solver": "lbfgs"})
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(**kwargs)),
        ]
    )


def _random_forest(depth: int) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestClassifier(
                    class_weight="balanced",
                    max_depth=depth,
                    min_samples_leaf=5,
                    n_estimators=100,
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def _evaluate_symmetry_only(feature_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature_set_name, columns in _feature_sets(feature_df).items():
        for model_name, model in _models().items():
            rows.extend(
                _evaluate_feature_model(
                    feature_df,
                    feature_set_name=feature_set_name,
                    model_name=model_name,
                    model=model,
                    columns=columns,
                )
            )
    return pd.DataFrame(rows)


def _evaluate_feature_model(
    feature_df: pd.DataFrame,
    *,
    feature_set_name: str,
    model_name: str,
    model: Pipeline,
    columns: list[str],
) -> list[dict[str, Any]]:
    rows = []
    x = feature_df[columns]
    y = feature_df["label"].to_numpy(dtype=int)
    rows.append(
        _train_all_row(
            feature_df,
            feature_set_name=feature_set_name,
            model_name=model_name,
            model=model,
            x=x,
            y=y,
        )
    )
    rows.append(
        _repeated_split_row(
            feature_df,
            feature_set_name=feature_set_name,
            model_name=model_name,
            model=model,
            columns=columns,
        )
    )
    rows.append(
        _stratified_kfold_row(
            feature_df,
            feature_set_name=feature_set_name,
            model_name=model_name,
            model=model,
            columns=columns,
        )
    )
    return rows


def _train_all_row(
    feature_df: pd.DataFrame,
    *,
    feature_set_name: str,
    model_name: str,
    model: Pipeline,
    x: pd.DataFrame,
    y: np.ndarray,
) -> dict[str, Any]:
    fitted = _clone_pipeline(model)
    fitted.fit(x, y)
    score = _score_model(fitted, x)
    return _summary_row(
        feature_set_name=feature_set_name,
        model_name=model_name,
        mode="train-all",
        scores=[score],
        labels=[y],
    )


def _repeated_split_row(
    feature_df: pd.DataFrame,
    *,
    feature_set_name: str,
    model_name: str,
    model: Pipeline,
    columns: list[str],
) -> dict[str, Any]:
    y = feature_df["label"].to_numpy(dtype=int)
    splitter = StratifiedShuffleSplit(
        n_splits=N_SPLITS,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )
    metrics = []
    for split_id, (train_idx, test_idx) in enumerate(splitter.split(feature_df, y)):
        train_x = feature_df.iloc[train_idx][columns]
        test_x = feature_df.iloc[test_idx][columns]
        train_y = y[train_idx]
        test_y = y[test_idx]
        fitted = _clone_pipeline(model)
        fitted.fit(train_x, train_y)
        train_score = _score_model(fitted, train_x)
        test_score = _score_model(fitted, test_x)
        threshold = _threshold_for_sensitivity(train_y, train_score)
        metrics.append(_metric_values(test_y, test_score, threshold, split_id))
    return _mean_metric_row(feature_set_name, model_name, "70/30 x50", metrics)


def _stratified_kfold_row(
    feature_df: pd.DataFrame,
    *,
    feature_set_name: str,
    model_name: str,
    model: Pipeline,
    columns: list[str],
) -> dict[str, Any]:
    y = feature_df["label"].to_numpy(dtype=int)
    splitter = StratifiedKFold(
        n_splits=KFOLD_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    metrics = []
    for split_id, (train_idx, test_idx) in enumerate(splitter.split(feature_df, y)):
        train_x = feature_df.iloc[train_idx][columns]
        test_x = feature_df.iloc[test_idx][columns]
        train_y = y[train_idx]
        test_y = y[test_idx]
        fitted = _clone_pipeline(model)
        fitted.fit(train_x, train_y)
        train_score = _score_model(fitted, train_x)
        test_score = _score_model(fitted, test_x)
        threshold = _threshold_for_sensitivity(train_y, train_score)
        metrics.append(_metric_values(test_y, test_score, threshold, split_id))
    return _mean_metric_row(feature_set_name, model_name, "stratified 5-fold", metrics)


def _best_symmetry_feature_set(symmetry_summary: pd.DataFrame) -> str:
    honest = symmetry_summary[
        (symmetry_summary["mode"] == "70/30 x50")
        & (symmetry_summary["model_name"].isin(["LR_L2", "LR_L1", "LR_elastic"]))
    ].copy()
    honest.sort_values(
        ["roc_auc_mean", "specificity_mean"],
        ascending=[False, False],
        inplace=True,
    )
    return str(honest.iloc[0]["feature_set"])


def _evaluate_profile_plus_symmetry(
    df: pd.DataFrame,
    feature_df: pd.DataFrame,
    best_set: str,
) -> pd.DataFrame:
    feature_sets = _feature_sets(feature_df)
    candidate_sets = {
        "M0_profile_only": [],
        "M1_sk_base": feature_sets["sk_base"],
        f"M1_{best_set}": feature_sets[best_set],
        f"M2_{best_set}_age": [*feature_sets[best_set], "age", "age_available"],
    }
    rows = []
    rows.extend(_evaluate_profile_candidates(df, feature_df, candidate_sets, "70/30 x50"))
    rows.extend(
        _evaluate_profile_candidates(
            df,
            feature_df,
            candidate_sets,
            "stratified 5-fold",
        )
    )
    rows.extend(_evaluate_profile_train_all(df, feature_df, candidate_sets))
    return pd.DataFrame(rows)


def _evaluate_profile_candidates(
    df: pd.DataFrame,
    feature_df: pd.DataFrame,
    candidate_sets: dict[str, list[str]],
    mode: str,
) -> list[dict[str, Any]]:
    y = feature_df["label"].to_numpy(dtype=int)
    if mode == "70/30 x50":
        split_pairs = StratifiedShuffleSplit(
            n_splits=N_SPLITS,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
        ).split(feature_df, y)
    else:
        split_pairs = StratifiedKFold(
            n_splits=KFOLD_SPLITS,
            shuffle=True,
            random_state=RANDOM_STATE,
        ).split(feature_df, y)
    metrics: dict[str, list[dict[str, float]]] = {
        name: [] for name in candidate_sets
    }
    for split_id, (train_idx, test_idx) in enumerate(split_pairs):
        train_patients = set(feature_df.iloc[train_idx]["patientId"].astype(str))
        lr1_model = _fit_lr1(df, train_patients)
        train_features = _profile_scores_for_patients(
            df,
            feature_df.iloc[train_idx],
            lr1_model,
        )
        test_features = _profile_scores_for_patients(
            df,
            feature_df.iloc[test_idx],
            lr1_model,
        )
        for candidate_name, symmetry_columns in candidate_sets.items():
            train_score, test_score = _candidate_scores(
                candidate_name,
                train_features,
                test_features,
                symmetry_columns,
                random_state=RANDOM_STATE + split_id,
            )
            threshold = _threshold_for_sensitivity(
                train_features["label"].to_numpy(dtype=int),
                train_score,
            )
            metrics[candidate_name].append(
                _metric_values(
                    test_features["label"].to_numpy(dtype=int),
                    test_score,
                    threshold,
                    split_id,
                )
            )
    return [
        _mean_metric_row("profile_plus_sk", name, mode, model_metrics)
        for name, model_metrics in metrics.items()
    ]


def _evaluate_profile_train_all(
    df: pd.DataFrame,
    feature_df: pd.DataFrame,
    candidate_sets: dict[str, list[str]],
) -> list[dict[str, Any]]:
    lr1_model = _fit_lr1(df, set(feature_df["patientId"].astype(str)))
    features = _profile_scores_for_patients(df, feature_df, lr1_model)
    rows = []
    for candidate_name, symmetry_columns in candidate_sets.items():
        train_score, score = _candidate_scores(
            candidate_name,
            features,
            features,
            symmetry_columns,
            random_state=RANDOM_STATE,
        )
        threshold = _threshold_for_sensitivity(
            features["label"].to_numpy(dtype=int),
            train_score,
        )
        rows.append(
            _summary_row(
                feature_set_name="profile_plus_sk",
                model_name=candidate_name,
                mode="train-all",
                scores=[score],
                labels=[features["label"].to_numpy(dtype=int)],
                threshold=threshold,
            )
        )
    return rows


def _fit_lr1(df: pd.DataFrame, patient_ids: set[str]) -> Pipeline:
    rows = df[df[GROUP_COLUMN].astype(str).isin(patient_ids)].copy()
    rows = rows[_boolean_series(rows[BIOPSY_COLUMN])].copy()
    rows = rows[rows[LABEL_COLUMN].isin(LABEL_MAP)].copy()
    if rows[LABEL_COLUMN].nunique() != 2:
        raise ValueError("LR1 training split must contain BENIGN and CANCER.")
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "logreg",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=5000,
                    random_state=RANDOM_STATE,
                    solver="lbfgs",
                ),
            ),
        ]
    )
    model.fit(profile_matrix(rows, PROFILE_COLUMN), rows[LABEL_COLUMN].map(LABEL_MAP))
    return model


def _profile_scores_for_patients(
    df: pd.DataFrame,
    feature_rows: pd.DataFrame,
    lr1_model: Pipeline,
) -> pd.DataFrame:
    rows = []
    for feature_row in feature_rows.itertuples(index=False):
        patient_id = str(feature_row.patientId)
        target_side = str(feature_row.target_side)
        patient_df = df[df[GROUP_COLUMN].astype(str) == patient_id]
        target_df = patient_df[
            patient_df[SIDE_COLUMN].map(_normalize_side) == target_side
        ].copy()
        if target_df.empty:
            continue
        scores = lr1_model.predict_proba(profile_matrix(target_df, PROFILE_COLUMN))[:, 1]
        row = feature_row._asdict()
        row["profile_p_cancer_logit_average"] = _logit_average_probability(scores)
        row["profile_p_cancer_probability_mean"] = float(np.mean(scores))
        row["profile_p_cancer_n_measurements"] = int(scores.size)
        rows.append(row)
    return pd.DataFrame(rows)


def _candidate_scores(
    candidate_name: str,
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    symmetry_columns: list[str],
    *,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    if candidate_name == "M0_profile_only":
        return (
            train_features["profile_p_cancer_logit_average"].to_numpy(dtype=float),
            test_features["profile_p_cancer_logit_average"].to_numpy(dtype=float),
        )
    columns = ["profile_p_cancer_logit_average", *symmetry_columns]
    model = _logistic("l2")
    model.set_params(model__random_state=random_state)
    model.fit(train_features[columns], train_features["label"].to_numpy(dtype=int))
    return (
        model.predict_proba(train_features[columns])[:, 1],
        model.predict_proba(test_features[columns])[:, 1],
    )


def _single_feature_auc(feature_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    y = feature_df["label"].to_numpy(dtype=int)
    for column in _feature_sets(feature_df)["sk_all"]:
        values = pd.to_numeric(feature_df[column], errors="coerce").fillna(0.0)
        auc = roc_auc_score(y, values)
        rows.append(
            {
                "feature": column,
                "auc_raw": float(auc),
                "auc_oriented": float(max(auc, 1.0 - auc)),
                "direction": 1 if auc >= 0.5 else -1,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["auc_oriented", "feature"], ascending=[False, True]
    )


def _summary_row(
    *,
    feature_set_name: str,
    model_name: str,
    mode: str,
    scores: list[np.ndarray],
    labels: list[np.ndarray],
    threshold: float | None = None,
) -> dict[str, Any]:
    y = np.concatenate(labels)
    score = np.concatenate(scores)
    threshold = _threshold_for_sensitivity(y, score) if threshold is None else threshold
    metric = _metric_values(y, score, threshold, split_id=0)
    return {
        "feature_set": feature_set_name,
        "model_name": model_name,
        "mode": mode,
        "roc_auc_mean": metric["roc_auc"],
        "roc_auc_std": 0.0,
        "pr_auc_mean": metric["pr_auc"],
        "sensitivity_mean": metric["sensitivity"],
        "sensitivity_std": 0.0,
        "specificity_mean": metric["specificity"],
        "specificity_std": 0.0,
        "threshold_mean": metric["threshold"],
        "n_rows": int(len(y)),
        "n_splits": 1,
    }


def _mean_metric_row(
    feature_set_name: str,
    model_name: str,
    mode: str,
    metrics: list[dict[str, float]],
) -> dict[str, Any]:
    frame = pd.DataFrame(metrics)
    return {
        "feature_set": feature_set_name,
        "model_name": model_name,
        "mode": mode,
        "roc_auc_mean": float(frame["roc_auc"].mean()),
        "roc_auc_std": float(frame["roc_auc"].std(ddof=0)),
        "pr_auc_mean": float(frame["pr_auc"].mean()),
        "sensitivity_mean": float(frame["sensitivity"].mean()),
        "sensitivity_std": float(frame["sensitivity"].std(ddof=0)),
        "specificity_mean": float(frame["specificity"].mean()),
        "specificity_std": float(frame["specificity"].std(ddof=0)),
        "threshold_mean": float(frame["threshold"].mean()),
        "n_rows": int(frame["n_rows"].mean()),
        "n_splits": int(len(frame)),
    }


def _metric_values(
    y: np.ndarray,
    score: np.ndarray,
    threshold: float,
    split_id: int,
) -> dict[str, float]:
    pred = (score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "split_id": float(split_id),
        "roc_auc": float(roc_auc_score(y, score)),
        "pr_auc": float(average_precision_score(y, score)),
        "sensitivity": _ratio(tp, tp + fn),
        "specificity": _ratio(tn, tn + fp),
        "threshold": float(threshold),
        "n_rows": float(len(y)),
    }


def _threshold_for_sensitivity(
    y_true: np.ndarray,
    score: np.ndarray,
    target_sensitivity: float = TARGET_SENSITIVITY,
) -> float:
    positive_scores = np.sort(np.asarray(score, dtype=float)[np.asarray(y_true) == 1])
    n_tp = int(np.ceil(float(target_sensitivity) * positive_scores.size))
    return float(positive_scores[max(0, positive_scores.size - n_tp)])


def _score_model(model: Pipeline, x: pd.DataFrame) -> np.ndarray:
    estimator = model.named_steps["model"]
    if hasattr(estimator, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    return model.decision_function(x)


def _clone_pipeline(model: Pipeline) -> Pipeline:
    return clone(model)


def _profile_roi(
    q: np.ndarray,
    y: np.ndarray,
    q_roi: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    mask = (q >= float(q_roi[0])) & (q <= float(q_roi[1]))
    if int(mask.sum()) < 5:
        return q, y
    return q[mask], y[mask]


def _smooth_profile(y: np.ndarray) -> np.ndarray:
    if y.size < 7:
        return y
    window = min(11, y.size if y.size % 2 else y.size - 1)
    if window < 5:
        return y
    return savgol_filter(y, window_length=window, polyorder=min(3, window - 2))


def _normalize_profile_near_minimum(
    q: np.ndarray,
    y: np.ndarray,
    *,
    q0: float = 6.7,
    halfwidth: float = 0.25,
) -> np.ndarray:
    mask = (q >= q0 - halfwidth) & (q <= q0 + halfwidth) & np.isfinite(y)
    baseline = (
        float(np.nanpercentile(y[mask], 5))
        if int(mask.sum()) >= 2
        else float(np.nanpercentile(y, 5))
    )
    if not np.isfinite(baseline) or abs(baseline) < EPS:
        baseline = 1.0
    return y / baseline


def _profile_std(values: np.ndarray) -> np.ndarray:
    if values.shape[0] < 2:
        return np.zeros(values.shape[1])
    return np.std(values, axis=0, ddof=1)


def _rms_difference(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    diff = np.asarray(a - b, dtype=float)
    good = mask & np.isfinite(diff)
    return float(np.sqrt(np.mean(diff[good] ** 2))) if int(good.sum()) >= 5 else np.nan


def _weighted_rms_difference(
    a: np.ndarray,
    b: np.ndarray,
    std_a: np.ndarray,
    std_b: np.ndarray,
    mask: np.ndarray,
) -> float:
    diff = np.asarray(a - b, dtype=float)
    var = np.asarray(std_a**2 + std_b**2, dtype=float)
    good = mask & np.isfinite(diff) & np.isfinite(var)
    if int(good.sum()) < 5:
        return np.nan
    floor = float(np.nanpercentile(var[good], 5))
    weight = 1.0 / np.maximum(var[good], floor + EPS)
    return float(np.sqrt(np.sum(weight * diff[good] ** 2) / np.sum(weight)))


def _mahalanobis_difference(
    a: np.ndarray,
    b: np.ndarray,
    std_a: np.ndarray,
    std_b: np.ndarray,
    mask: np.ndarray,
) -> float:
    diff = np.asarray(a - b, dtype=float)
    var = np.asarray(std_a**2 + std_b**2, dtype=float)
    good = mask & np.isfinite(diff) & np.isfinite(var)
    if int(good.sum()) < 5:
        return np.nan
    return float(np.sqrt(np.sum(diff[good] ** 2 / (var[good] + EPS))))


def _sigma_rms(std: np.ndarray, mask: np.ndarray) -> float:
    good = mask & np.isfinite(std)
    return float(np.sqrt(np.mean(std[good] ** 2))) if int(good.sum()) >= 5 else np.nan


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    good = np.isfinite(a) & np.isfinite(b)
    if int(good.sum()) < 5:
        return np.nan
    av = np.asarray(a[good], dtype=float)
    bv = np.asarray(b[good], dtype=float)
    denom = float(np.linalg.norm(av) * np.linalg.norm(bv))
    if denom <= EPS:
        return np.nan
    return float(1.0 - np.clip(np.dot(av, bv) / denom, -1.0, 1.0))


def _profile_wasserstein(q: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    good = np.isfinite(q) & np.isfinite(a) & np.isfinite(b)
    if int(good.sum()) < 5:
        return np.nan
    qv = q[good]
    av = np.clip(a[good], 0.0, None)
    bv = np.clip(b[good], 0.0, None)
    if float(av.sum()) <= EPS or float(bv.sum()) <= EPS:
        return np.nan
    order = np.argsort(qv)
    qv = qv[order]
    av = av[order] / float(av.sum())
    bv = bv[order] / float(bv.sum())
    return float(np.sum(np.abs(np.cumsum(av)[:-1] - np.cumsum(bv)[:-1]) * np.diff(qv)))


def _peak_intensity(q: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    y = 0.5 * (a + b)
    mask = np.isfinite(y)
    return float(np.nanmax(y[mask])) if int(mask.sum()) >= 3 else np.nan


def _finite_or_zero(value: float) -> float:
    return float(value) if np.isfinite(value) else 0.0


def _numeric_median(df: pd.DataFrame, column: str) -> float:
    if column not in df.columns:
        return 0.0
    values = pd.to_numeric(df[column], errors="coerce")
    return float(values.median()) if values.notna().any() else 0.0


def _has_numeric(df: pd.DataFrame, column: str) -> bool:
    return column in df.columns and pd.to_numeric(df[column], errors="coerce").notna().any()


def _boolean_series(values: pd.Series) -> pd.Series:
    clean = values.astype("object").where(values.notna(), False)
    if clean.dtype == bool:
        return clean
    return clean.astype(str).str.lower().isin(["true", "1", "yes"])


def _normalize_side(value: Any) -> str | None:
    clean = str(value).strip().upper()
    if clean.startswith("LEFT"):
        return "LEFT"
    if clean.startswith("RIGHT"):
        return "RIGHT"
    return None


def _ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def _write_markdown_summary(
    *,
    feature_df: pd.DataFrame,
    single_feature_auc: pd.DataFrame,
    symmetry_summary: pd.DataFrame,
    fusion_summary: pd.DataFrame,
    best_set: str,
) -> None:
    path = ROOT / "docs/modeling/sk_symmetry_optimization_v0_1.md"
    top_symmetry = symmetry_summary.sort_values(
        ["mode", "roc_auc_mean", "specificity_mean"],
        ascending=[True, False, False],
    ).groupby("mode").head(5)
    text = f"""# SK Symmetry Optimization v0.1

Clinical framing: research-draft decision support only; requires radiologist
review. This is a feature-discovery experiment, not clinical validation.

## Dataset

```text
input: {INPUT_JOBLIB.relative_to(ROOT)}
patients: {feature_df['patientId'].nunique()}
BENIGN patients: {(feature_df['label'] == 0).sum()}
CANCER patients: {(feature_df['label'] == 1).sum()}
```

## Feature Families Tested

```text
sk_base
sk_ratios
sk_windows
sk_windows_ratios
sk_windows_reliability
sk_all
```

Windowed SK metrics were computed over:

```text
{Q_WINDOWS}
```

## Best Discovery Feature Set

```text
{best_set}
```

## Top Single Features

{_markdown_table(single_feature_auc.head(15))}

## Symmetry-only Best Rows

{_markdown_table(top_symmetry)}

## Profile Plus SK Candidates

{_markdown_table(fusion_summary)}

## Artifacts

```text
docs/modeling/results/sk_symmetry_optimization_features_v0_1.csv
docs/modeling/results/sk_symmetry_optimization_single_features_v0_1.csv
docs/modeling/results/sk_symmetry_optimization_symmetry_only_v0_1.csv
docs/modeling/results/sk_symmetry_optimization_profile_plus_sk_v0_1.csv
```
"""
    path.write_text(text, encoding="utf-8")


def _markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    rows = df.copy()
    for column in rows.columns:
        if pd.api.types.is_float_dtype(rows[column]):
            rows[column] = rows[column].map(lambda value: f"{float(value):.3f}")
        else:
            rows[column] = rows[column].astype(str)
    header = "| " + " | ".join(rows.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(rows.columns)) + " |"
    body = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in rows.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *body])


if __name__ == "__main__":
    main()
