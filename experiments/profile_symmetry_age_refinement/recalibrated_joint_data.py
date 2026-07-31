"""Patient-safe data construction and provenance for joint-refinement research."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import subprocess
from typing import Any

import joblib
import numpy as np
import pandas as pd

from aramina.patient_features import empty_lr1_scores, patient_feature_table
from aramina.training_config import PRODUCT_MODEL_NAME, resolve_model_definition
from aramina.training_evaluation import _fit_split_feature_tables, _patient_split_pairs


@dataclass(frozen=True)
class CrossFittedFeatures:
    """One patient-safe LR1 OOF feature table and its fold manifest."""

    features: pd.DataFrame
    manifest: pd.DataFrame


@dataclass(frozen=True)
class FullChainMetaPair:
    """Cached features for one strictly nested meta validation fold."""

    fold_id: int
    meta_train_features: pd.DataFrame
    meta_validation_features: pd.DataFrame
    manifest: pd.DataFrame


def load_input_dataframe(path: str | Path) -> pd.DataFrame:
    """Load an experiment input DataFrame from the standard joblib forms."""
    value = joblib.load(Path(path))
    if isinstance(value, pd.DataFrame):
        return value.copy()
    if isinstance(value, dict) and isinstance(value.get("dataframe"), pd.DataFrame):
        return value["dataframe"].copy()
    raise ValueError("Input joblib must contain a DataFrame or {'dataframe': DataFrame}.")


def input_metadata(path: str | Path, *, repository: Path) -> dict[str, str]:
    """Return reproducibility identifiers for a supplied input artifact."""
    source = Path(path).expanduser().resolve()
    return {
        "input_joblib_path": str(source),
        "input_joblib_sha256": sha256(source.read_bytes()).hexdigest(),
        "git_sha": _git_sha(repository),
        "git_worktree_dirty": str(_git_worktree_dirty(repository)).lower(),
    }


def model_columns() -> dict[str, str]:
    """Resolve product data column names without changing product configuration."""
    model = resolve_model_definition(PRODUCT_MODEL_NAME)["model"]
    return {
        name: str(model[name])
        for name in (
            "profile_column",
            "label_column",
            "group_column",
            "specimen_column",
            "side_column",
            "q_column",
            "age_column",
            "biopsy_column",
            "lr1_row_policy",
        )
    }


def base_features(frame: pd.DataFrame, columns: dict[str, str]) -> pd.DataFrame:
    """Build target-case rows with neutral LR1 scores for patient splitting."""
    return patient_feature_table(
        frame,
        empty_lr1_scores(
            frame,
            group_column=columns["group_column"],
            side_column=columns["side_column"],
            label_column=columns["label_column"],
            biopsy_column=columns["biopsy_column"],
        ),
        profile_column=columns["profile_column"],
        label_column=columns["label_column"],
        group_column=columns["group_column"],
        specimen_column=columns["specimen_column"],
        side_column=columns["side_column"],
        q_column=columns["q_column"],
        age_column=columns["age_column"],
        biopsy_column=columns["biopsy_column"],
    )


def split_pairs(
    features: pd.DataFrame,
    *,
    n_splits: int,
    n_repeats: int,
    random_state: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create patient-safe stratified split indices with feasible fold count."""
    patient_labels = features.groupby("patientId")["label"].max()
    actual_splits = min(int(n_splits), int(patient_labels.value_counts().min()))
    if actual_splits < 2:
        raise ValueError("At least two patients of each class are required for splitting.")
    return _patient_split_pairs(
        mode="stratified_kfold",
        base_features=features,
        y_patients=features["label"].to_numpy(dtype=int),
        n_splits=actual_splits,
        n_repeats=int(n_repeats),
        random_state=int(random_state),
    )


def patient_ids(features: pd.DataFrame, index: np.ndarray) -> set[str]:
    """Return patient IDs represented by target-case indexes."""
    return set(features.iloc[index]["patientId"].astype(str))


def raw_patient_subset(
    frame: pd.DataFrame,
    columns: dict[str, str],
    patients: set[str],
) -> pd.DataFrame:
    """Return all measurements for a patient set."""
    return frame[frame[columns["group_column"]].astype(str).isin(patients)].copy()


def fit_feature_pair(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    columns: dict[str, str],
    *,
    lr1_c: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit LR1 on train measurements and create train/validation case features."""
    return _fit_split_feature_tables(
        train_frame,
        validation_frame,
        profile_column=columns["profile_column"],
        label_column=columns["label_column"],
        group_column=columns["group_column"],
        specimen_column=columns["specimen_column"],
        side_column=columns["side_column"],
        q_column=columns["q_column"],
        age_column=columns["age_column"],
        biopsy_column=columns["biopsy_column"],
        lr1_row_policy=columns["lr1_row_policy"],
        lr1_logreg_c=float(lr1_c),
        random_state=int(random_state),
    )


def lr1_cross_fitted_features(
    frame: pd.DataFrame,
    columns: dict[str, str],
    *,
    lr1_c: float,
    n_splits: int,
    random_state: int,
    context: dict[str, Any],
) -> CrossFittedFeatures:
    """Create LR1 OOF rows and explicit train/validation patient manifests."""
    base = base_features(frame, columns)
    pieces: list[pd.DataFrame] = []
    manifest_rows: list[dict[str, Any]] = []
    for lr1_fold, (train_index, validation_index) in enumerate(
        split_pairs(base, n_splits=n_splits, n_repeats=1, random_state=random_state)
    ):
        train_ids = patient_ids(base, train_index)
        validation_ids = patient_ids(base, validation_index)
        if train_ids.intersection(validation_ids):
            raise RuntimeError("Patient leakage in LR1 cross-fitting.")
        _, validation_features = fit_feature_pair(
            raw_patient_subset(frame, columns, train_ids),
            raw_patient_subset(frame, columns, validation_ids),
            columns,
            lr1_c=lr1_c,
            random_state=random_state + lr1_fold,
        )
        part = validation_features.copy()
        part["lr1_crossfit_fold"] = int(lr1_fold)
        pieces.append(part)
        manifest_rows.extend(
            _manifest_rows(
                context=context,
                level="lr1",
                fold_id=lr1_fold,
                role="lr1_fit_train",
                patient_ids=train_ids,
            )
        )
        manifest_rows.extend(
            _manifest_rows(
                context=context,
                level="lr1",
                fold_id=lr1_fold,
                role="lr1_oof_validation",
                patient_ids=validation_ids,
            )
        )
    result = pd.concat(pieces, ignore_index=True)
    if result["target_case_id"].duplicated().any() or len(result) != len(base):
        raise RuntimeError("LR1 OOF construction must return one row per target case.")
    return CrossFittedFeatures(
        features=result.sort_values("target_case_id", kind="stable").reset_index(drop=True),
        manifest=pd.DataFrame(manifest_rows),
    )


def full_chain_meta_pairs(
    outer_train_frame: pd.DataFrame,
    columns: dict[str, str],
    *,
    lr1_c: float,
    meta_splits: int,
    random_state: int,
    outer_split_id: int | str,
    inner_lr1_splits: int,
) -> list[FullChainMetaPair]:
    """Cache fully nested LR1/meta train-validation feature pairs.

    For each meta validation fold, LR1 OOF rows for meta-model training are
    created from meta-train patients only. LR1 then fits all meta-train raw data
    to score the meta-validation patients. Thus no meta-validation patient can
    influence any LR1 model used for that fold's meta-model fitting.
    """
    base = base_features(outer_train_frame, columns)
    pairs: list[FullChainMetaPair] = []
    for meta_fold, (train_index, validation_index) in enumerate(
        split_pairs(base, n_splits=meta_splits, n_repeats=1, random_state=random_state)
    ):
        train_ids = patient_ids(base, train_index)
        validation_ids = patient_ids(base, validation_index)
        if train_ids.intersection(validation_ids):
            raise RuntimeError("Patient leakage in meta-fold split.")
        meta_train_raw = raw_patient_subset(outer_train_frame, columns, train_ids)
        meta_validation_raw = raw_patient_subset(outer_train_frame, columns, validation_ids)
        context = {"outer_split_id": outer_split_id, "meta_fold_id": int(meta_fold)}
        meta_train = lr1_cross_fitted_features(
            meta_train_raw,
            columns,
            lr1_c=lr1_c,
            n_splits=inner_lr1_splits,
            random_state=random_state + meta_fold * 1_000,
            context=context,
        )
        _, meta_validation_features = fit_feature_pair(
            meta_train_raw,
            meta_validation_raw,
            columns,
            lr1_c=lr1_c,
            random_state=random_state + meta_fold * 1_000 + 999,
        )
        if set(meta_train.features["patientId"].astype(str)).intersection(validation_ids):
            raise RuntimeError("Meta validation patient entered LR1 OOF meta training.")
        if set(meta_validation_features["patientId"].astype(str)).intersection(train_ids):
            raise RuntimeError("Meta-train patient entered meta validation feature table.")
        meta_manifest = pd.DataFrame(
            [
                *_manifest_rows(
                    context=context,
                    level="meta",
                    fold_id=meta_fold,
                    role="meta_model_train",
                    patient_ids=train_ids,
                ),
                *_manifest_rows(
                    context=context,
                    level="meta",
                    fold_id=meta_fold,
                    role="meta_model_validation",
                    patient_ids=validation_ids,
                ),
            ]
        )
        pairs.append(
            FullChainMetaPair(
                fold_id=meta_fold,
                meta_train_features=meta_train.features,
                meta_validation_features=meta_validation_features,
                manifest=pd.concat([meta_manifest, meta_train.manifest], ignore_index=True),
            )
        )
    return pairs


def _manifest_rows(
    *,
    context: dict[str, Any],
    level: str,
    fold_id: int,
    role: str,
    patient_ids: set[str],
) -> list[dict[str, Any]]:
    return [
        {**context, "level": level, "fold_id": int(fold_id), "role": role, "patient_id": patient}
        for patient in sorted(patient_ids)
    ]


def _git_sha(repository: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _git_worktree_dirty(repository: Path) -> bool:
    """Record whether the source tree contains uncommitted experiment code."""
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    return bool(result.stdout.strip())
