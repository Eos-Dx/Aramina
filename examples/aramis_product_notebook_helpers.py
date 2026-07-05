from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any

import h5py
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from xrd_preprocessing import (
    H5MeasurementSetAuditTransformer,
    H5SessionSelectorTransformer,
    load_preprocessing_config,
)


LABEL_VALUES = ["BENIGN", "CANCER"]
CANCER_GROUP_VALUES = ["CANCER", "ATYPICAL", "PRE_CANCEROUS"]
NON_NA_SPECIMEN_STATUS_VALUES = [
    "BENIGN",
    "CANCER",
    "NORMAL",
    "ATYPICAL",
    "PRE_CANCEROUS",
]
CALIBRANT_THICKNESS_COLUMN = "calibrant_thickness_mm"
LEGACY_AGBH_THICKNESS_COLUMN = "agbh_thickness_mm"
CALIBRANT_THICKNESS_SOURCE_COLUMNS = (
    CALIBRANT_THICKNESS_COLUMN,
    LEGACY_AGBH_THICKNESS_COLUMN,
)
AGBH_THICKNESS_COLUMN = LEGACY_AGBH_THICKNESS_COLUMN
HEAVY_DETECTOR_COLUMNS = (
    "measurement_data",
    "gfrm_data",
    "raw_data",
    "processed_data",
    "detector_measurements",
)
NON_OUTPUT_PAYLOAD_COLUMNS = (
    *HEAVY_DETECTOR_COLUMNS,
    "faulty_pixel_mask",
    "radial_profile_data_raw",
    "radial_profile_sigma",
)


def ensure_product_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "measurementDate" not in out.columns:
        out["measurementDate"] = out.get("started_at")
    thickness_sources = [
        column
        for column in ["sample_thickness_mm", "sample_thickness", "thickness_raw_mm"]
        if column in out.columns
    ]
    if thickness_sources:
        numeric_sources = [
            pd.to_numeric(out[column], errors="coerce") for column in thickness_sources
        ]
        out["sample_thickness_mm"] = (
            pd.concat(numeric_sources, axis=1).bfill(axis=1).iloc[:, 0]
        )
    elif "sample_thickness_mm" not in out.columns:
        out["sample_thickness_mm"] = np.nan
    if CALIBRANT_THICKNESS_COLUMN not in out.columns:
        for column in CALIBRANT_THICKNESS_SOURCE_COLUMNS:
            if column in out.columns:
                out[CALIBRANT_THICKNESS_COLUMN] = out[column]
                break
    if "product_status_group" not in out.columns:
        out["product_status_group"] = product_status_group(out)
    if "product_diagnosis" not in out.columns:
        out["product_diagnosis"] = canonical_diagnosis(out)
    if "patient_product_diagnosis" not in out.columns and "patientId" in out.columns:
        out["patient_product_diagnosis"] = patient_product_diagnosis(out)
    return out


def max_sessions_from_value(value: int) -> int | None:
    return None if int(value) == 0 else int(value)


def read_agbh_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_aramis_preprocessing_config(path: Path) -> dict[str, Any]:
    return load_preprocessing_config(path)


def resolve_config_path(config_path: Path, value: str) -> Path:
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path
    return (config_path.parent / path).resolve()


def preprocessing_io_paths(
    config_path: Path,
    config: dict[str, Any],
) -> tuple[Path, Path]:
    io = config.get("io", {})
    if not io.get("input_h5_path"):
        raise ValueError(f"Missing io.input_h5_path in {config_path}")
    if not io.get("output_joblib_path"):
        raise ValueError(f"Missing io.output_joblib_path in {config_path}")
    return (
        resolve_config_path(config_path, io["input_h5_path"]),
        resolve_config_path(config_path, io["output_joblib_path"]),
    )


def thickness_settings_from_config(config: dict[str, Any]) -> dict[str, Any]:
    filters = config.get("filters", {})
    integration = config.get("integration", {})
    correction = integration.get("thickness_correction", {})
    filter_thickness = filters.get("thickness", {})
    sample_filter = filter_thickness.get("sample", {})
    calibrant_filter = filter_thickness.get("calibrant", {})
    sample_column = correction.get(
        "sample_thickness_column",
        sample_filter.get("column", "sample_thickness_mm"),
    )
    calibrant_column = correction.get(
        "calibrant_thickness_column",
        calibrant_filter.get("column", CALIBRANT_THICKNESS_COLUMN),
    )
    if sample_filter.get("column") not in {None, sample_column}:
        raise ValueError("Sample-thickness H5 filter column differs from integrator.")
    if calibrant_filter.get("column") not in {None, calibrant_column}:
        raise ValueError("Calibrant-thickness H5 filter column differs from integrator.")
    legacy_range = filters.get(
        "calibrant_thickness_range_mm",
        correction.get("calibrant_thickness_range_mm", [10.0, 40.0]),
    )
    return {
        "enabled": bool(correction.get("enabled", True)),
        "sample_column": sample_column,
        "calibrant_column": calibrant_column,
        "require_sample": bool(
            sample_filter.get(
                "require",
                correction.get(
                    "require_sample_thickness",
                    filters.get("require_sample_thickness_mm", True),
                ),
            )
        ),
        "require_calibrant": bool(
            calibrant_filter.get(
                "require",
                correction.get(
                    "require_calibrant_thickness",
                    filters.get("require_calibrant_thickness_mm", True),
                ),
            )
        ),
        "sample_min_mm": float(sample_filter.get("min_mm", 0.0)),
        "calibrant_min_mm": float(calibrant_filter.get("min_mm", legacy_range[0])),
        "calibrant_max_mm": float(calibrant_filter.get("max_mm", legacy_range[1])),
    }


def thickness_settings_text(settings: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"sample thickness column: {settings['sample_column']}",
            f"sample thickness required: {settings['require_sample']}",
            f"calibrant thickness column: {settings['calibrant_column']}",
            f"calibrant thickness required: {settings['require_calibrant']}",
            "calibrant thickness range, mm: "
            f"{settings['calibrant_min_mm']:g}..{settings['calibrant_max_mm']:g}",
        ]
    )


def validate_thickness_columns_for_integration(
    df: pd.DataFrame,
    settings: dict[str, Any],
) -> None:
    if not settings.get("enabled", True):
        return
    sample_column = settings["sample_column"]
    calibrant_column = settings["calibrant_column"]
    if sample_column not in df.columns:
        raise ValueError(f"Missing required sample thickness column: {sample_column}")
    if calibrant_column not in df.columns:
        raise ValueError(
            f"Missing required calibrant thickness column: {calibrant_column}"
        )
    sample_values = pd.to_numeric(df[sample_column], errors="coerce")
    calibrant_values = pd.to_numeric(df[calibrant_column], errors="coerce")
    if not bool(np.isfinite(sample_values).all()):
        raise ValueError(f"Invalid sample thickness values in {sample_column}")
    if not bool((sample_values > float(settings["sample_min_mm"])).all()):
        raise ValueError(
            f"Sample thickness values in {sample_column} must be > "
            f"{settings['sample_min_mm']:g} mm"
        )
    if not bool(np.isfinite(calibrant_values).all()):
        raise ValueError(f"Invalid calibrant thickness values in {calibrant_column}")
    if not bool(
        calibrant_values.between(
            float(settings["calibrant_min_mm"]),
            float(settings["calibrant_max_mm"]),
        ).all()
    ):
        raise ValueError(
            f"Calibrant thickness values in {calibrant_column} must be "
            f"{settings['calibrant_min_mm']:g}.."
            f"{settings['calibrant_max_mm']:g} mm"
        )


def read_product_versioning(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def agbh_selection(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "accepted_dates": config["selection"]["accepted_dates"],
        "threshold": config["parameters"]["max_score"],
        "selected_batches": config["parameters"].get("selected_batches", []),
        "distance_q_range_policy": config.get("product_distance_q_range_policy", {}),
    }


def _nova_number(value: Any) -> int | None:
    match = re.search(r"Nova[_\s-]*(\d+)", str(value or ""), flags=re.IGNORECASE)
    if match is None:
        return None
    return int(match.group(1))


def _parse_nova_range(value: Any) -> tuple[int, int] | None:
    if value is None:
        return None
    numbers = re.findall(r"\d+", str(value))
    if len(numbers) < 2:
        return None
    return int(numbers[0]), int(numbers[1])


def _batch_lookup(versioning_config: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for batch in versioning_config.get("batch_table", []):
        nova_range = _parse_nova_range(batch.get("nova_range"))
        if nova_range is None:
            continue
        rows.append(
            {
                "data_batch": str(batch.get("data_batch")),
                "nova_min": nova_range[0],
                "nova_max": nova_range[1],
                "calibrant_thickness_mm": batch.get(
                    CALIBRANT_THICKNESS_COLUMN,
                    batch.get(LEGACY_AGBH_THICKNESS_COLUMN),
                ),
            }
        )
    return rows


def _batch_for_row(row: pd.Series, lookup: list[dict[str, Any]]) -> str | None:
    for column in ("human1_data_batch", "product_batch_id", "data_batch"):
        if column in row.index and pd.notna(row[column]):
            value = str(row[column]).strip()
            if value and value.lower() != "nan":
                return value
    for column in ("patientId", "specimenId", "name", "external_id"):
        if column not in row.index:
            continue
        nova = _nova_number(row[column])
        if nova is None:
            continue
        for batch in lookup:
            if batch["nova_min"] <= nova <= batch["nova_max"]:
                return batch["data_batch"]
    return None


def add_agbh_reference_thickness(
    df: pd.DataFrame,
    versioning_config: dict[str, Any],
) -> pd.DataFrame:
    out = df.copy()
    lookup = _batch_lookup(versioning_config)
    thickness_by_batch = {
        batch["data_batch"]: batch["calibrant_thickness_mm"]
        for batch in lookup
        if batch.get("calibrant_thickness_mm") is not None
    }
    batches = []
    thicknesses = []
    sources = []
    existing = (
        pd.to_numeric(out[CALIBRANT_THICKNESS_COLUMN], errors="coerce")
        if CALIBRANT_THICKNESS_COLUMN in out.columns
        else pd.Series(np.nan, index=out.index)
    )
    for _idx, row in out.iterrows():
        batch = _batch_for_row(row, lookup)
        value = existing.loc[_idx]
        if np.isfinite(value):
            source = "h5"
        elif batch in thickness_by_batch:
            value = float(thickness_by_batch[batch])
            source = "aramis_product_versioning_json_batch"
        else:
            value = np.nan
            source = "missing"
        batches.append(batch)
        thicknesses.append(value)
        sources.append(source)
    out["human1_data_batch"] = batches
    out[CALIBRANT_THICKNESS_COLUMN] = pd.to_numeric(thicknesses, errors="coerce")
    out["calibrant_thickness_source"] = sources
    return out


def calibrant_thickness_summary_text(df: pd.DataFrame) -> str:
    if CALIBRANT_THICKNESS_COLUMN not in df.columns:
        return "calibrant_thickness_mm: missing"
    counts = (
        pd.to_numeric(df[CALIBRANT_THICKNESS_COLUMN], errors="coerce")
        .fillna(-1)
        .value_counts()
        .sort_index()
    )
    labels = []
    for value, count in counts.items():
        label = "NA" if value < 0 else f"{value:g} mm"
        labels.append(f"{label}={int(count)}")
    source_counts = (
        df.get("calibrant_thickness_source", pd.Series([], dtype="object"))
        .fillna("NA")
        .astype(str)
        .value_counts()
        .to_dict()
    )
    return "\n".join(
        [
            f"calibrant_thickness_mm counts: {', '.join(labels)}",
            f"calibrant_thickness_source counts: {source_counts}",
        ]
    )


def agbh_thickness_summary_text(df: pd.DataFrame) -> str:
    return calibrant_thickness_summary_text(df)


def patient_product_diagnosis(df: pd.DataFrame) -> pd.Series:
    status = (
        df["specimen_status"]
        if "specimen_status" in df.columns
        else df["product_diagnosis"]
        if "product_diagnosis" in df.columns
        else pd.Series([""] * len(df), index=df.index)
    )
    status = status.fillna("").astype(str).str.strip().str.upper()
    patient = (
        df["patientId"]
        if "patientId" in df.columns
        else pd.Series([None] * len(df), index=df.index)
    )
    labels: dict[object, str] = {}
    for _patient_id, _status_values in status.groupby(patient, dropna=False):
        _unique = set(_status_values)
        if _unique & {
            "CANCER",
            "MALIGNANT",
            "TUMOR",
            "TUMOUR",
            "ATYPICAL",
            "PRE_CANCEROUS",
        }:
            labels[_patient_id] = "CANCER"
        elif _unique & {"BENIGN", "B9"}:
            labels[_patient_id] = "BENIGN"
        else:
            labels[_patient_id] = "NA"
    return patient.map(labels).fillna("NA")


def product_status_group(df: pd.DataFrame) -> pd.Series:
    source = (
        df["specimen_status"]
        if "specimen_status" in df.columns
        else pd.Series([None] * len(df), index=df.index)
    )
    value = source.fillna("").astype(str).str.strip().str.upper()
    out = pd.Series(pd.NA, index=df.index, dtype="object")
    out.loc[value.isin(["BENIGN", "B9"])] = "BENIGN"
    out.loc[
        value.isin(
            [
                "CANCER",
                "MALIGNANT",
                "TUMOR",
                "TUMOUR",
                "ATYPICAL",
                "PRE_CANCEROUS",
            ]
        )
    ] = "CANCER"
    out.loc[value.isin(["NORMAL", "HEALTHY"])] = "NORMAL"
    return out


def canonical_diagnosis(df: pd.DataFrame) -> pd.Series:
    status_group = (
        df["product_status_group"]
        if "product_status_group" in df.columns
        else product_status_group(df)
    )
    out = pd.Series(pd.NA, index=df.index, dtype="object")
    out.loc[status_group.isin(LABEL_VALUES)] = status_group.loc[
        status_group.isin(LABEL_VALUES)
    ]
    return out


def diagnosis_filter_stats(
    before_df: pd.DataFrame,
    after_df: pd.DataFrame,
    *,
    diagnosis_column: str = "product_diagnosis",
) -> dict[str, Any]:
    before_counts = (
        before_df[diagnosis_column].fillna("NA").astype(str).value_counts().to_dict()
        if diagnosis_column in before_df.columns
        else {}
    )
    after_counts = (
        after_df[diagnosis_column].fillna("NA").astype(str).value_counts().to_dict()
        if diagnosis_column in after_df.columns
        else {}
    )
    return {
        "rows_in": int(len(before_df)),
        "rows_pass": int(len(after_df)),
        "rows_dropped": int(len(before_df) - len(after_df)),
        "before_counts": before_counts,
        "after_counts": after_counts,
    }


def filter_valid_thickness(
    df: pd.DataFrame,
    *,
    column: str = "sample_thickness_mm",
) -> tuple[pd.DataFrame, dict[str, int]]:
    out = df.copy()
    values = pd.to_numeric(out[column], errors="coerce")
    valid = values.notna() & np.isfinite(values) & (values > 0)
    out[column] = values
    filtered = out.loc[valid].copy()
    stats = {
        "rows_in": int(len(out)),
        "rows_pass": int(len(filtered)),
        "rows_dropped": int(len(out) - len(filtered)),
    }
    return filtered, stats


def _upper_status(df: pd.DataFrame) -> pd.Series:
    return df["specimen_status"].fillna("").astype(str).str.strip().str.upper()


def _boolean_series(values: pd.Series) -> pd.Series:
    clean = values.astype("object").where(values.notna(), False)
    if clean.dtype == bool:
        return clean
    return clean.astype(str).str.lower().isin(["true", "1", "yes"])


def filter_product_status_group(
    df: pd.DataFrame,
    allowed: list[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = ensure_product_columns(df)
    allowed_set = {value.upper() for value in allowed}
    mask = (
        out["product_status_group"].fillna("").astype(str).str.upper().isin(allowed_set)
    )
    filtered = out.loc[mask].copy()
    stats = diagnosis_filter_stats(
        out,
        filtered,
        diagnosis_column="product_status_group",
    )
    return filtered, stats


def filter_one_to_one_pair_groups(
    df: pd.DataFrame,
    *,
    patient_column: str = "patientId",
    specimen_column: str = "specimenId",
    group_column: str = "product_status_group",
    allowed_pairs: tuple[tuple[str, str], ...] = (
        ("BENIGN", "CANCER"),
        ("BENIGN", "NORMAL"),
        ("CANCER", "NORMAL"),
    ),
) -> tuple[pd.DataFrame, dict[str, Any]]:
    missing = [
        column
        for column in (patient_column, specimen_column, group_column)
        if column not in df.columns
    ]
    if missing:
        raise KeyError(f"Missing one-to-one pair columns: {missing}")
    out = df.copy()
    specimen_groups = (
        out[[patient_column, specimen_column, group_column]]
        .dropna(subset=[patient_column, specimen_column, group_column])
        .drop_duplicates()
        .groupby([patient_column, specimen_column], dropna=False, as_index=False)
        .agg(**{group_column: (group_column, "first")})
    )
    allowed = {tuple(sorted(pair)) for pair in allowed_pairs}
    patient_pairs: dict[Any, tuple[str, ...]] = {}
    for patient_id, rows in specimen_groups.groupby(patient_column, dropna=False):
        groups = tuple(sorted(rows[group_column].astype(str).str.upper().tolist()))
        patient_pairs[patient_id] = groups
    valid_patients = [
        patient_id
        for patient_id, groups in patient_pairs.items()
        if len(groups) == 2 and groups in allowed
    ]
    pair_labels = {
        patient_id: "__".join(patient_pairs[patient_id])
        for patient_id in valid_patients
    }
    filtered = out.loc[out[patient_column].isin(valid_patients)].copy()
    filtered["one_to_one_pair_type"] = filtered[patient_column].map(pair_labels)
    before_counts = {
        "__".join(groups) if groups else "NA": count
        for groups, count in pd.Series(patient_pairs).value_counts().items()
    }
    after_patient_pair_counts = pd.Series(pair_labels).value_counts().to_dict()
    after_row_pair_counts = filtered["one_to_one_pair_type"].value_counts().to_dict()
    stats = {
        "rows_in": int(len(out)),
        "rows_pass": int(len(filtered)),
        "rows_dropped": int(len(out) - len(filtered)),
        "patients_in": int(out[patient_column].nunique()),
        "patients_pass": int(filtered[patient_column].nunique()),
        "specimens_in": int(
            out[[patient_column, specimen_column]].drop_duplicates().shape[0]
        ),
        "specimens_pass": int(
            filtered[[patient_column, specimen_column]].drop_duplicates().shape[0]
        ),
        "allowed_pairs": ["__".join(pair) for pair in allowed_pairs],
        "before_pair_counts": before_counts,
        "after_pair_counts": after_patient_pair_counts,
        "after_pair_row_counts": after_row_pair_counts,
    }
    return filtered, stats


def drop_heavy_detector_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Drop frame-sized detector arrays after azimuthal integration."""
    columns = [column for column in HEAVY_DETECTOR_COLUMNS if column in df.columns]
    if not columns:
        return df.copy(), []
    return df.drop(columns=columns).copy(), columns


def slim_product_preprocessing_frame(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    """Keep normalized profiles plus all H5/product metadata."""
    dropped_columns = [
        column for column in NON_OUTPUT_PAYLOAD_COLUMNS if column in df.columns
    ]
    return df.drop(columns=dropped_columns).copy(), dropped_columns


def build_branch_h5_filters(
    *,
    H5SessionFilter,
    calibrant_thickness_h5_filters,
    filter_h5_sessions,
    archive_path: Path,
    accepted_dates: list[str],
    required_q_max_nm_inv: float,
    thickness_settings: dict[str, Any],
    branch: str,
    normal_as_benign: bool = False,
    require_biopsy_patient: bool = False,
) -> dict[str, Any]:
    date_filter = H5SessionFilter(
        column="started_at",
        op="date in",
        values=accepted_dates,
    )
    q_range_filter = H5SessionFilter(
        column="poni_q_max_nm_inv",
        op=">=",
        value=required_q_max_nm_inv,
    )
    non_na_status_filter = H5SessionFilter(
        column="specimen_status",
        op="in",
        values=NON_NA_SPECIMEN_STATUS_VALUES,
    )
    calibrant_filters = (
        calibrant_thickness_h5_filters(
            column=thickness_settings["calibrant_column"],
            min_mm=thickness_settings["calibrant_min_mm"],
            max_mm=thickness_settings["calibrant_max_mm"],
        )
        if thickness_settings["require_calibrant"]
        else []
    )
    base_filters = [date_filter, q_range_filter]
    _ = filter_h5_sessions
    sample_manifest = H5SessionSelectorTransformer(
        session_category="SAMPLE",
    ).fit_transform(archive_path)
    candidate_manifest = H5SessionSelectorTransformer(
        filters=[*base_filters, *calibrant_filters, non_na_status_filter],
    ).fit_transform(sample_manifest)
    sample_session_df = sample_manifest["session_df"]
    candidate_session_df = candidate_manifest["session_df"]
    status = _upper_status(candidate_session_df)
    lesion_status_values = [*LABEL_VALUES, "ATYPICAL", "PRE_CANCEROUS"]
    product_status_values = (
        [*lesion_status_values, "NORMAL"]
        if normal_as_benign
        else lesion_status_values
    )
    eligible_patient_ids = sorted(
        candidate_session_df.loc[
            status.isin(lesion_status_values),
            "patientId",
        ]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    if branch == "one_to_one":
        branch_filters = [
            non_na_status_filter,
            H5SessionFilter(
                column="patientId",
                op="in",
                values=eligible_patient_ids,
            ),
        ]
        description = (
            "drop NA specimen_status; keep patients with at least one "
            "BENIGN/CANCER/ATYPICAL/PRE_CANCEROUS breast"
        )
    elif branch == "one_to_many":
        branch_filters = []
        if require_biopsy_patient:
            biopsy_patient_ids = sorted(
                candidate_session_df.loc[
                    _boolean_series(candidate_session_df["biopsy"]),
                    "patientId",
                ]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )
            branch_filters.append(
                H5SessionFilter(
                    column="patientId",
                    op="in",
                    values=biopsy_patient_ids,
                )
            )
        branch_filters.append(
            H5SessionFilter(
                column="specimen_status",
                op="in",
                values=product_status_values,
            )
        )
        status_text = "/".join(product_status_values)
        biopsy_text = (
            "keep patients with at least one biopsy row; "
            if require_biopsy_patient
            else ""
        )
        description = (
            f"{biopsy_text}keep raw {status_text} specimen_status at H5 level; "
            "product labels are formed after h5_to_df"
        )
    else:
        raise ValueError(f"Unknown Aramis DataFrame branch: {branch}")

    return {
        "branch": branch,
        "date_filter": date_filter,
        "base_filters": base_filters,
        "calibrant_filters": calibrant_filters,
        "drop_missing_sample_thickness": thickness_settings["require_sample"],
        "thickness_settings": thickness_settings,
        "non_na_status_filter": non_na_status_filter,
        "filters": [*base_filters, *calibrant_filters, *branch_filters],
        "eligible_patient_ids": eligible_patient_ids,
        "sample_session_df": sample_session_df,
        "description": description,
    }


def branch_h5_stage_frames(
    *,
    archive_path: Path,
    filter_h5_sessions,
    list_h5_measurement_sets,
    filter_plan: dict[str, Any],
    max_sessions: int | None,
) -> dict[str, pd.DataFrame]:
    _ = filter_h5_sessions
    _ = list_h5_measurement_sets
    audit_manifest = H5MeasurementSetAuditTransformer(
        stage_filters={
            "before": [],
            "after_date": [filter_plan["date_filter"]],
            "after_q_range": filter_plan["base_filters"],
            "after_calibrant": [
                *filter_plan["base_filters"],
                *filter_plan["calibrant_filters"],
            ],
            "limited_branch": filter_plan["filters"],
        },
        max_sessions_by_stage={"limited_branch": max_sessions},
    ).fit_transform(
        {
            "archive_path": archive_path,
            "all_session_df": filter_plan["sample_session_df"],
        }
    )
    stage_frames = audit_manifest["h5_stage_frames"]
    before = ensure_product_columns(stage_frames["before"])
    after_date = ensure_product_columns(stage_frames["after_date"])
    after_q_range = ensure_product_columns(stage_frames["after_q_range"])
    after_calibrant = ensure_product_columns(stage_frames["after_calibrant"])
    after_thickness = after_calibrant.copy()
    if filter_plan["drop_missing_sample_thickness"]:
        sample_thickness = pd.to_numeric(
            after_thickness["sample_thickness_mm"],
            errors="coerce",
        )
        after_thickness = after_thickness.loc[
            sample_thickness.notna() & np.isfinite(sample_thickness)
        ].copy()
    after_thickness = ensure_product_columns(after_thickness)
    status = after_thickness["specimen_status"].fillna("").astype(str).str.upper()
    after_drop_na = after_thickness.loc[
        status.isin(NON_NA_SPECIMEN_STATUS_VALUES)
    ].copy()
    if filter_plan["branch"] == "one_to_one":
        eligible = after_drop_na["patientId"].astype(str).isin(
            filter_plan["eligible_patient_ids"]
        )
        after_branch = after_drop_na.loc[eligible].copy()
    else:
        status = after_drop_na["specimen_status"].fillna("").astype(str).str.upper()
        lesion_status_values = [*LABEL_VALUES, "ATYPICAL", "PRE_CANCEROUS"]
        after_branch = after_drop_na.loc[status.isin(lesion_status_values)].copy()
    if max_sessions is not None:
        limited_paths = set(stage_frames["limited_branch"]["session_path"].astype(str))
        after_branch = after_branch.loc[
            after_branch["session_path"].astype(str).isin(limited_paths)
        ].copy()
    return {
        "before": before,
        "after_date": after_date,
        "after_q_range": after_q_range,
        "after_calibrant_thickness_filter": after_calibrant,
        "after_thickness_filter": after_thickness,
        "after_drop_na_status": after_drop_na,
        "after_branch_filter": after_branch,
    }


def branch_h5_filter_stage_counts(stage_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return stage_counts(
        [
            ("before_filters", stage_frames["before"]),
            ("after_date_filter", stage_frames["after_date"]),
            ("after_q_range_filter", stage_frames["after_q_range"]),
            (
                "after_calibrant_thickness_filter",
                stage_frames["after_calibrant_thickness_filter"],
            ),
            ("after_thickness_filter", stage_frames["after_thickness_filter"]),
            ("after_drop_na_status", stage_frames["after_drop_na_status"]),
            ("after_branch_filter", stage_frames["after_branch_filter"]),
        ]
    )


def build_h5_filters(
    *,
    H5SessionFilter,
    filter_h5_sessions,
    archive_path: Path,
    accepted_dates: list[str],
    required_q_max_nm_inv: float,
    h5_label_filter_scope: str,
) -> dict[str, Any]:
    date_filter = H5SessionFilter(
        column="started_at",
        op="date in",
        values=accepted_dates,
    )
    q_range_filter = H5SessionFilter(
        column="poni_q_max_nm_inv",
        op=">=",
        value=required_q_max_nm_inv,
    )
    base_filters = [date_filter, q_range_filter]
    candidate_session_df = filter_h5_sessions(
        archive_path,
        filters=base_filters,
        session_category="SAMPLE",
    )
    status = (
        candidate_session_df["specimen_status"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
    )
    eligible_patient_ids = sorted(
        candidate_session_df.loc[
            status.isin(LABEL_VALUES),
            "patientId",
        ]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    if h5_label_filter_scope == "patient":
        label_filters = [
            H5SessionFilter(
                column="patientId",
                op="in",
                values=eligible_patient_ids,
            )
        ]
        description = "patientId has at least one BENIGN/CANCER breast"
    elif h5_label_filter_scope == "specimen":
        label_filters = [
            H5SessionFilter(
                column="specimen_status",
                op="in",
                values=LABEL_VALUES,
            )
        ]
        description = "specimen_status in BENIGN, CANCER"
    elif h5_label_filter_scope == "none":
        label_filters = []
        description = "no H5 BENIGN/CANCER label filter"
    else:
        raise ValueError(f"Unknown H5 label filter scope: {h5_label_filter_scope}")

    return {
        "date_filter": date_filter,
        "base_filters": base_filters,
        "filters": [*base_filters, *label_filters],
        "eligible_patient_ids": eligible_patient_ids,
        "description": description,
    }


def h5_stage_frames(
    *,
    archive_path: Path,
    filter_h5_sessions,
    list_h5_measurement_sets,
    filter_plan: dict[str, Any],
    max_sessions: int | None,
) -> dict[str, pd.DataFrame]:
    before_sessions = filter_h5_sessions(archive_path, session_category="SAMPLE")
    date_sessions = filter_h5_sessions(
        archive_path,
        filters=[filter_plan["date_filter"]],
        session_category="SAMPLE",
    )
    q_range_sessions = filter_h5_sessions(
        archive_path,
        filters=filter_plan["base_filters"],
        session_category="SAMPLE",
    )
    diagnosis_sessions = filter_h5_sessions(
        archive_path,
        filters=filter_plan["filters"],
        session_category="SAMPLE",
        max_sessions=max_sessions,
    )
    before = ensure_product_columns(
        list_h5_measurement_sets(archive_path, session_df=before_sessions)
    )
    after_date = ensure_product_columns(
        list_h5_measurement_sets(archive_path, session_df=date_sessions)
    )
    after_q_range = ensure_product_columns(
        list_h5_measurement_sets(archive_path, session_df=q_range_sessions)
    )
    after_diagnosis = ensure_product_columns(
        list_h5_measurement_sets(archive_path, session_df=diagnosis_sessions)
    )
    after_thickness = ensure_product_columns(
        list_h5_measurement_sets(
            archive_path,
            session_df=diagnosis_sessions,
            drop_missing_sample_thickness=True,
        )
    )
    return {
        "before": before,
        "after_date": after_date,
        "after_q_range": after_q_range,
        "after_diagnosis": after_diagnosis,
        "after_thickness": after_thickness,
    }


def h5_filter_stage_counts(stage_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return stage_counts(
        [
            ("before_filters", stage_frames["before"]),
            ("after_date_filter", stage_frames["after_date"]),
            ("after_q_range_filter", stage_frames["after_q_range"]),
            ("after_diagnosis_filter", stage_frames["after_diagnosis"]),
            ("after_thickness_filter", stage_frames["after_thickness"]),
        ]
    )


def add_interpolation_q_range(
    df: pd.DataFrame,
    *,
    q_min: float = 2.0,
    q_max: float = 23.0,
) -> pd.DataFrame:
    out = df.copy()
    out["interpolation_q_range"] = [(q_min, q_max)] * len(out)
    return out


def make_one_to_many_source(
    ColumnValueFilter,
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    label_filter = ColumnValueFilter(
        "product_diagnosis",
        op="in",
        values=LABEL_VALUES,
    )
    filtered = label_filter.fit_transform(df)
    stats = diagnosis_filter_stats(df, filtered)
    return filtered, stats


def build_product_branch_sources(
    ColumnValueFilter,
    patient_context_df: pd.DataFrame,
) -> dict[str, Any]:
    one_to_many_source_df, one_to_many_stats = make_one_to_many_source(
        ColumnValueFilter,
        patient_context_df,
    )
    return {
        "one_to_many_source_df": one_to_many_source_df,
        "one_to_many_label_filter_stats": one_to_many_stats,
        "one_to_one_source_df": patient_context_df.copy(),
    }


def build_branch_datasets(
    PatientSpecimenValidityFilter,
    *,
    min_measurements_per_specimen: int,
    one_to_many_source_df: pd.DataFrame,
    one_to_one_source_df: pd.DataFrame,
) -> dict[str, Any]:
    one_to_many_filter = PatientSpecimenValidityFilter(
        min_measurements_per_specimen=min_measurements_per_specimen,
        min_specimens_per_patient=1,
    )
    one_to_many_df = one_to_many_filter.fit_transform(one_to_many_source_df)

    one_to_one_filter = PatientSpecimenValidityFilter(
        min_measurements_per_specimen=min_measurements_per_specimen,
        min_specimens_per_patient=2,
    )
    one_to_one_df = one_to_one_filter.fit_transform(one_to_one_source_df)

    return {
        "one_to_many_df": one_to_many_df,
        "one_to_many_stats": one_to_many_filter.stats_,
        "one_to_one_df": one_to_one_df,
        "one_to_one_stats": one_to_one_filter.stats_,
    }


def normalize_branch_datasets(
    QRangeValueNormalizer,
    *,
    one_to_many_df: pd.DataFrame,
    one_to_one_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    one_to_many = QRangeValueNormalizer(
        q_min=6.7,
        q_max=7.1,
        statistic="median",
        save_initial_data=True,
    ).fit_transform(one_to_many_df)
    one_to_one = QRangeValueNormalizer(
        q_min=6.7,
        q_max=7.1,
        statistic="median",
        save_initial_data=True,
    ).fit_transform(one_to_one_df)
    return one_to_many, one_to_one


def apply_profile_value_gate(
    RadialProfileValueFilter,
    *,
    one_to_many_normalized_df: pd.DataFrame,
    one_to_one_normalized_df: pd.DataFrame,
    q_value_nm_inv: float,
    threshold: float,
) -> dict[str, Any]:
    one_to_many_filter = RadialProfileValueFilter(
        q_value_nm_inv=q_value_nm_inv,
        threshold=threshold,
        op=">",
    )
    one_to_many_df = one_to_many_filter.fit_transform(one_to_many_normalized_df)

    one_to_one_filter = RadialProfileValueFilter(
        q_value_nm_inv=q_value_nm_inv,
        threshold=threshold,
        op=">",
    )
    one_to_one_df = one_to_one_filter.fit_transform(one_to_one_normalized_df)

    return {
        "one_to_many_df": one_to_many_df,
        "one_to_many_stats": one_to_many_filter.stats_,
        "one_to_one_df": one_to_one_df,
        "one_to_one_stats": one_to_one_filter.stats_,
    }


def specimen_status_values(df: pd.DataFrame) -> pd.Series:
    if "specimen_status" not in df:
        return pd.Series([], dtype="object")
    values = df["specimen_status"].fillna("NA").astype(str).str.strip().str.upper()
    return values.replace("", "NA")


def measurement_weights(df: pd.DataFrame) -> pd.Series:
    return pd.Series(1, index=df.index, dtype="int64")


def measurement_count(df: pd.DataFrame) -> int:
    return int(measurement_weights(df).sum())


def diagnosis_values(df: pd.DataFrame) -> pd.Series:
    source = None
    for column in (
        "product_status_group",
        "product_diagnosis",
        "specimen_status",
        "patient_product_diagnosis",
    ):
        if column in df:
            source = df[column]
            break
    if source is None:
        return pd.Series([], dtype="object")
    values = source.fillna("NA").astype(str).str.strip().str.upper()
    return values.replace("", "NA")


def specimen_status_counts_text(df: pd.DataFrame) -> str:
    counts = specimen_status_values(df).value_counts()
    if counts.empty:
        return "none"
    return ", ".join(f"{label}={int(value)}" for label, value in counts.items())


def diagnosis_counts_text(df: pd.DataFrame) -> str:
    values = diagnosis_values(df)
    if values.empty:
        return "none"
    weights = measurement_weights(df).reindex(values.index).fillna(0)
    counts = weights.groupby(values).sum().sort_values(ascending=False)
    if counts.empty:
        return "none"
    return ", ".join(f"{label}={int(value)}" for label, value in counts.items())


def stage_counts(stages: list[tuple[str, pd.DataFrame]]) -> pd.DataFrame:
    def _count_label(df: pd.DataFrame, labels: set[str]) -> int:
        values = diagnosis_values(df)
        if values.empty:
            return 0
        weights = measurement_weights(df).reindex(values.index).fillna(0)
        return int(weights.loc[values.isin(labels)].sum())

    return pd.DataFrame(
        [
            {
                "stage": stage,
                "rows": measurement_count(df),
                "patients": int(df["patientId"].nunique()) if "patientId" in df else 0,
                "specimens": int(df["specimenId"].nunique()) if "specimenId" in df else 0,
                "diagnosis_variants": int(diagnosis_values(df).nunique()),
                "diagnosis_counts": diagnosis_counts_text(df),
                "benign": _count_label(df, {"BENIGN", "B9"}),
                "cancer": _count_label(
                    df,
                    {"CANCER", "MALIGNANT", "TUMOR", "TUMOUR"},
                ),
            }
            for stage, df in stages
        ]
    )


def frame_counts(df: pd.DataFrame) -> dict[str, int]:
    diagnosis = diagnosis_values(df)
    weights = measurement_weights(df)
    return {
        "rows": measurement_count(df),
        "patients": int(df["patientId"].nunique()) if "patientId" in df else 0,
        "specimens": int(df["specimenId"].nunique()) if "specimenId" in df else 0,
        "diagnosis_variants": int(diagnosis.nunique()),
        "benign": int(weights.loc[diagnosis.isin(["BENIGN", "B9"])].sum()),
        "cancer": int(
            weights.loc[
                diagnosis.isin(["CANCER", "MALIGNANT", "TUMOR", "TUMOUR"])
            ].sum()
        ),
    }


def stage_summary_text(
    stage: str,
    after_df: pd.DataFrame,
    *,
    before_df: pd.DataFrame | None = None,
) -> str:
    after = frame_counts(after_df)
    if before_df is None:
        return "\n".join(
            [
                f"{stage}",
                f"measurements: {after['rows']}",
                f"patients: {after['patients']}",
                f"specimens/breasts: {after['specimens']}",
                f"diagnosis variants: {after['diagnosis_variants']}",
                f"diagnosis counts: {diagnosis_counts_text(after_df)}",
                f"BENIGN measurements: {after['benign']}",
                f"CANCER measurements: {after['cancer']}",
            ]
        )
    before = frame_counts(before_df)
    return "\n".join(
        [
            f"{stage}",
            f"measurements: {before['rows']} -> {after['rows']} ({after['rows'] - before['rows']:+d})",
            f"patients: {before['patients']} -> {after['patients']} ({after['patients'] - before['patients']:+d})",
            f"specimens/breasts: {before['specimens']} -> {after['specimens']} ({after['specimens'] - before['specimens']:+d})",
            f"diagnosis variants: {before['diagnosis_variants']} -> {after['diagnosis_variants']} ({after['diagnosis_variants'] - before['diagnosis_variants']:+d})",
            f"diagnosis counts before: {diagnosis_counts_text(before_df)}",
            f"diagnosis counts after: {diagnosis_counts_text(after_df)}",
            f"BENIGN measurements: {before['benign']} -> {after['benign']} ({after['benign'] - before['benign']:+d})",
            f"CANCER measurements: {before['cancer']} -> {after['cancer']} ({after['cancer'] - before['cancer']:+d})",
        ]
    )


def counts_text(counts_df: pd.DataFrame) -> str:
    return "\n".join(
        f"{row.stage}: measurements={row.rows}, patients={row.patients}, specimens={row.specimens}, diagnoses={row.diagnosis_variants}, diagnosis_counts=({row.diagnosis_counts})"
        for row in counts_df.itertuples(index=False)
    )


def plot_profiles(
    df: pd.DataFrame,
    *,
    title: str,
    intensity_column: str = "radial_profile_data",
    q_column: str = "q_range",
    max_profiles: int = 120,
    diagnosis_column: str | None = None,
):
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for _, row in df.head(max_profiles).iterrows():
        q = np.asarray(row[q_column], dtype=float)
        y = np.asarray(row[intensity_column], dtype=float)
        n = min(len(q), len(y))
        color = None
        if diagnosis_column is not None and diagnosis_column in df.columns:
            diagnosis = str(row.get(diagnosis_column, "")).upper()
            if diagnosis == "CANCER":
                color = "#dc2626"
            elif diagnosis == "BENIGN":
                color = "#7c3aed"
        ax.plot(q[:n], y[:n], alpha=0.25, linewidth=0.8, color=color)
    if diagnosis_column is not None and diagnosis_column in df.columns:
        ax.plot([], [], color="#7c3aed", label="BENIGN")
        ax.plot([], [], color="#dc2626", label="CANCER")
        ax.legend(loc="upper right", fontsize=8)
    ax.set_title(title)
    ax.set_xlabel("q, nm^-1")
    ax.set_ylabel(intensity_column)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig


def plot_snr(df: pd.DataFrame, *, cutoff_db: float):
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    values = pd.to_numeric(df["snr_db"], errors="coerce").dropna()
    ax.hist(values, bins=30, color="#4c78a8", alpha=0.85)
    ax.axvline(cutoff_db, color="#c43b3b", linewidth=2)
    ax.set_title("Poisson SNR")
    ax.set_xlabel("snr_db")
    ax.set_ylabel("measurements")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig


def plot_stage_counts(counts_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 4.8))
    y = np.arange(len(counts_df))
    ax.barh(y, counts_df["rows"], color="#6d6e9f", alpha=0.88)
    ax.set_yticks(y)
    ax.set_yticklabels(counts_df["stage"])
    ax.invert_yaxis()
    ax.set_xlabel("rows")
    ax.set_title("Rows by preprocessing stage")
    ax.grid(axis="x", alpha=0.2)
    for _idx, _value in enumerate(counts_df["rows"].astype(int)):
        ax.text(_value, _idx, f" {_value}", va="center", fontsize=9)
    fig.tight_layout()
    return fig


def plot_stage_metrics(counts_df: pd.DataFrame):
    metrics = [
        ("rows", "measurements"),
        ("patients", "patients"),
        ("specimens", "specimens/breasts"),
        ("diagnosis_variants", "diagnoses"),
    ]
    fig, ax = plt.subplots(figsize=(13, 5.8))
    x = np.arange(len(counts_df))
    width = 0.18
    offsets = np.linspace(
        -width * (len(metrics) - 1) / 2,
        width * (len(metrics) - 1) / 2,
        len(metrics),
    )
    colors = ["#345995", "#03cea4", "#f5a623", "#d1495b"]
    for _offset, (_column, _label), _color in zip(
        offsets,
        metrics,
        colors,
        strict=True,
    ):
        values = counts_df[_column].astype(int).to_numpy()
        ax.bar(x + _offset, values, width=width, label=_label, color=_color, alpha=0.88)
        for _idx, _value in enumerate(values):
            ax.text(
                x[_idx] + _offset,
                _value,
                str(int(_value)),
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=90 if _value > 99 else 0,
            )
    ax.set_title("Stage statistics")
    ax.set_ylabel("count")
    ax.set_xticks(x)
    ax.set_xticklabels(counts_df["stage"], rotation=25, ha="right")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(loc="upper right", ncols=4, fontsize=8)
    fig.tight_layout()
    return fig


def specimen_measurement_summary(df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "patientId",
        "specimenId",
        "side",
        "product_diagnosis",
        "sample_thickness_mm",
    ]
    available = [column for column in columns if column in df.columns]
    if not {"patientId", "specimenId"}.issubset(available):
        return pd.DataFrame()
    grouped = df.loc[:, available].copy()
    grouped["sample_thickness_mm"] = pd.to_numeric(
        grouped.get("sample_thickness_mm"),
        errors="coerce",
    )
    return (
        grouped.groupby(["patientId", "specimenId"], dropna=False)
        .agg(
            measurements=("specimenId", "size"),
            side=("side", "first") if "side" in grouped.columns else ("specimenId", "first"),
            diagnosis=("product_diagnosis", "first")
            if "product_diagnosis" in grouped.columns
            else ("specimenId", "first"),
            mean_thickness_mm=("sample_thickness_mm", "mean"),
            min_thickness_mm=("sample_thickness_mm", "min"),
            max_thickness_mm=("sample_thickness_mm", "max"),
        )
        .reset_index()
    )


def patient_measurement_summary(df: pd.DataFrame) -> pd.DataFrame:
    specimen_summary = specimen_measurement_summary(df)
    if specimen_summary.empty:
        return pd.DataFrame()
    return (
        specimen_summary.groupby("patientId", dropna=False)
        .agg(
            specimens=("specimenId", "nunique"),
            measurements=("measurements", "sum"),
            mean_measurements_per_specimen=("measurements", "mean"),
            mean_thickness_mm=("mean_thickness_mm", "mean"),
        )
        .reset_index()
    )


def plot_specimen_measurement_histogram(
    specimen_summary: pd.DataFrame,
    *,
    title: str,
):
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    values = pd.to_numeric(specimen_summary["measurements"], errors="coerce").dropna()
    bins = np.arange(0.5, float(values.max() if len(values) else 1) + 1.5, 1.0)
    ax.hist(values, bins=bins, color="#6d6e9f", alpha=0.86, rwidth=0.88)
    ax.set_title(title)
    ax.set_xlabel("measurements per specimen")
    ax.set_ylabel("specimens")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    return fig


def plot_patient_specimen_histogram(
    patient_summary: pd.DataFrame,
    *,
    title: str,
):
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    values = pd.to_numeric(patient_summary["specimens"], errors="coerce").dropna()
    bins = np.arange(0.5, float(values.max() if len(values) else 1) + 1.5, 1.0)
    ax.hist(values, bins=bins, color="#2f6fbb", alpha=0.86, rwidth=0.88)
    ax.set_title(title)
    ax.set_xlabel("valid specimens per patient")
    ax.set_ylabel("patients")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    return fig


def plot_faulty_summary(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    counts = df["faulty_pixel_mask"].apply(lambda value: int(len(value)))
    ax.hist(counts, bins=30, color="#6f9d55", alpha=0.85)
    ax.set_title("Faulty pixels per frame")
    ax.set_xlabel("faulty pixels")
    ax.set_ylabel("measurements")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig


def _decode(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        return value.item()
    return value


def _float_field(text: str, key: str) -> float:
    match = re.search(rf"^{key}:\s*([0-9.eE+-]+)", text, re.MULTILINE)
    if match is None:
        raise ValueError(f"Missing PONI field: {key}")
    return float(match.group(1))


def _detector_config(text: str) -> dict[str, Any]:
    match = re.search(r"^Detector_config:\s*(.+)$", text, re.MULTILINE)
    if match is None:
        raise ValueError("Missing PONI Detector_config")
    return json.loads(match.group(1).replace("'", '"'))


def parse_poni_geometry(poni_text: str) -> dict[str, Any]:
    config = _detector_config(poni_text)
    pixel1_m = float(config["pixel1"])
    pixel2_m = float(config["pixel2"])
    poni1_m = _float_field(poni_text, "Poni1")
    poni2_m = _float_field(poni_text, "Poni2")
    distance_m = _float_field(poni_text, "Distance")
    max_shape = config.get("max_shape") or [np.nan, np.nan]
    return {
        "poni_y_mm": poni1_m * 1000.0,
        "poni_x_mm": poni2_m * 1000.0,
        "poni_distance_mm": distance_m * 1000.0,
        "center_y_px": poni1_m / pixel1_m,
        "center_x_px": poni2_m / pixel2_m,
        "pixel1_um": pixel1_m * 1_000_000.0,
        "pixel2_um": pixel2_m * 1_000_000.0,
        "image_height_px": int(max_shape[0]),
        "image_width_px": int(max_shape[1]),
    }


def _scalar_fields(group: h5py.Group) -> dict[str, Any]:
    return {
        name: _decode(obj[()])
        for name, obj in group.items()
        if isinstance(obj, h5py.Dataset) and obj.shape == ()
    }


def read_calibration_poni_file(path: Path) -> dict[str, Any]:
    with h5py.File(path, "r") as h5:
        session = h5["session"]
        set_name = sorted(session["sets"])[0]
        set_group = session["sets"][set_name]
        acq = _scalar_fields(set_group["acquisition"])
        poni_text = _decode(set_group["artifacts/poni"][()])
        return {
            "file_name": path.name,
            "file_path": str(path),
            "session_uid": _decode(session.attrs["session_uid"]),
            "session_pk": int(session.attrs["session_pk"]),
            "started_at": _decode(session.attrs["started_at"]),
            "completed_at": _decode(session.attrs["completed_at"]),
            "set_name": set_name,
            "set_uid": _decode(set_group.attrs["set_uid"]),
            "measurement_type_name": _decode(set_group.attrs["measurement_type_name"]),
            "acquisition_distance_mm": float(acq["distance"]),
            "exposure_time_s": float(acq["exposure_time"]),
            "voltage_kv": float(acq["voltage"]),
            "current_ua": float(acq["current"]),
            **parse_poni_geometry(poni_text),
        }


def load_calibration_poni_geometry(calibration_dir: Path) -> pd.DataFrame:
    rows = [
        read_calibration_poni_file(_path)
        for _path in sorted(calibration_dir.glob("*.h5"))
    ]
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["started_at"] = pd.to_datetime(df["started_at"], errors="raise")
    df["completed_at"] = pd.to_datetime(df["completed_at"], errors="raise")
    df["day"] = df["started_at"].dt.normalize()
    return df.sort_values("started_at").reset_index(drop=True)


def filter_calibration_poni_geometry(
    df: pd.DataFrame,
    *,
    session_uids: list[str],
) -> pd.DataFrame:
    if df.empty:
        return df
    if not session_uids:
        return df.head(0).copy()
    out = df.copy()
    out = out.loc[out["session_uid"].isin(session_uids)]
    return out.sort_values("started_at").reset_index(drop=True)


def calibration_poni_summary_text(df: pd.DataFrame) -> str:
    if len(df) == 0:
        return "calibration_sessions=0"
    return "\n".join(
        [
            f"calibration_sessions={len(df)}",
            f"calibration_date_min={df['started_at'].min()}",
            f"calibration_date_max={df['started_at'].max()}",
            f"center_x_px={df['center_x_px'].min():.2f}..{df['center_x_px'].max():.2f}",
            f"center_y_px={df['center_y_px'].min():.2f}..{df['center_y_px'].max():.2f}",
            f"poni_distance_mm={df['poni_distance_mm'].min():.2f}..{df['poni_distance_mm'].max():.2f}",
        ]
    )


def plot_calibration_poni_xy(df: pd.DataFrame):
    fig, (ax_center, ax_distance) = plt.subplots(1, 2, figsize=(13.5, 5.4))
    if df.empty:
        ax_center.set_title("PONI center drift")
        ax_distance.set_title("PONI distance by selected calibration")
        fig.tight_layout()
        return fig

    scatter = ax_center.scatter(
        df["center_x_px"],
        df["center_y_px"],
        c=mdates.date2num(df["started_at"]),
        cmap="viridis",
        s=34,
        alpha=0.82,
        edgecolors="none",
    )
    ax_center.plot(
        df["center_x_px"],
        df["center_y_px"],
        color="#1f2933",
        linewidth=1.5,
        alpha=0.75,
    )
    ax_center.set_title("PONI center XY")
    ax_center.set_xlabel("center X, px = Poni2 / pixel2")
    ax_center.set_ylabel("center Y, px = Poni1 / pixel1")
    ax_center.grid(alpha=0.2)
    cbar = fig.colorbar(scatter, ax=ax_center)
    cbar.ax.yaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    cbar.set_label("started_at")

    ax_distance.scatter(
        df["started_at"],
        df["poni_distance_mm"],
        color="#4c78a8",
        alpha=0.45,
        s=24,
        label="session",
    )
    ax_distance.plot(
        df["started_at"],
        df["poni_distance_mm"],
        color="#c43b3b",
        linewidth=2.0,
        marker="o",
        markersize=4,
        label="selected calibration",
    )
    ax_distance.set_title("PONI distance by selected calibration")
    ax_distance.set_xlabel("started_at")
    ax_distance.set_ylabel("Distance, mm")
    ax_distance.legend(loc="best")
    ax_distance.grid(alpha=0.2)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def write_csv(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path
