"""Build full one-patient T130 prediction DataFrames from the source H5 archive."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import re
import tempfile
from typing import Any

import h5py
import pandas as pd

from aramina.pipelines import run_preprocessing_pipeline


def preprocess_manifest_patients(
    source_archive: str | Path,
    manifest: pd.DataFrame,
    prediction_preprocessing_config: dict[str, Any],
) -> pd.DataFrame:
    """Run frozen prediction preprocessing once per manifest patient."""
    pieces: list[pd.DataFrame] = []
    for patient_id, patient_cases in manifest.groupby("patient_id", sort=True):
        record = patient_cases.iloc[0].to_dict()
        with tempfile.TemporaryDirectory(prefix="aramina-t130-patient-") as temp:
            h5_path = extract_one_patient_h5(
                source_archive,
                record,
                target_side=str(record["target_side"]),
                output_path=Path(temp) / "one_patient.h5",
            )
            patient_frame = run_preprocessing_pipeline(
                h5_path,
                prediction_preprocessing_config,
            )
        if set(patient_frame["patientId"].astype(str)) != {str(patient_id)}:
            raise ValueError(f"Prediction preprocessing returned the wrong patient: {patient_id}")
        pieces.append(patient_frame)
    return pd.concat(pieces, ignore_index=True)


def extract_one_patient_h5(
    source_archive: str | Path,
    record: Mapping[str, Any],
    *,
    target_side: str,
    output_path: str | Path,
) -> Path:
    """Copy target and available contralateral sessions into one EOS v0.3 H5."""
    source = Path(source_archive).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    target = _normalize_side(target_side)
    if target is None:
        raise ValueError("target_side must be left or right.")
    target_session_name = _record_text(record, f"{target}_session")
    if not target_session_name:
        raise ValueError(f"Selected patient has no {target} breast session.")
    contralateral_side = "right" if target == "left" else "left"
    contralateral_session_name = _record_text(
        record,
        f"{contralateral_side}_session",
    )
    archive_group = _record_text(record, "archive_group")
    patient_id = _record_text(record, "patient_id")
    if not archive_group or not patient_id:
        raise ValueError("T130 record is missing archive_group or patient_id.")

    output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(source, "r") as source_h5, h5py.File(output, "w") as target_h5:
        source_group = source_h5[archive_group]
        if target_session_name not in source_group:
            raise KeyError(f"Source session is unavailable: {target_session_name}")
        source_group.copy(source_group[target_session_name], target_h5, name="session")
        target_session = target_h5["session"]
        if contralateral_session_name:
            _merge_contralateral_sets(
                source_group,
                target_session,
                contralateral_session_name,
            )
        _set_one_patient_metadata(
            source_group,
            target_session,
            patient_id=patient_id,
            target_session_name=target_session_name,
            contralateral_session_name=contralateral_session_name,
        )
        for key, value in target_session.attrs.items():
            target_h5.attrs[key] = value
        target_h5.attrs["format"] = "xrd-session"
        target_h5.attrs["schema_version"] = "0.3"
        target_h5.attrs["container_type"] = "session"
        target_h5.attrs["container_id"] = f"aramina-t130-{patient_id}"
    return output


def _merge_contralateral_sets(
    source_group: h5py.Group,
    target_session: h5py.Group,
    session_name: str,
) -> None:
    source_sets = source_group[session_name]["sets"]
    target_sets = target_session["sets"]
    for source_name in sorted(source_sets):
        source_sets.copy(
            source_sets[source_name],
            target_sets,
            name=f"contralateral_{source_name}",
        )


def _set_one_patient_metadata(
    source_group: h5py.Group,
    target_session: h5py.Group,
    *,
    patient_id: str,
    target_session_name: str,
    contralateral_session_name: str,
) -> None:
    for key, value in source_group.attrs.items():
        target_session.attrs[key] = value
    session_metadata = {
        target_session_name: _sample_metadata(
            source_group[target_session_name],
            patient_id,
        )
    }
    if contralateral_session_name:
        session_metadata[contralateral_session_name] = _sample_metadata(
            source_group[contralateral_session_name],
            patient_id,
        )
    for set_name, set_group in target_session["sets"].items():
        source_session_name = (
            contralateral_session_name
            if set_name.startswith("contralateral_")
            else target_session_name
        )
        for key, value in session_metadata[source_session_name].items():
            if value is not None:
                set_group.attrs[key] = value
        position = _canonical_position(_text(set_group.attrs.get("position")))
        if position is not None:
            set_group.attrs["position"] = position


def _sample_metadata(session: h5py.Group, patient_id: str) -> dict[str, Any]:
    sample = session["sample"]
    return {
        "patientId": patient_id,
        "specimenId": _dataset_text(sample, "name"),
        "side": _attr_text(sample, "side"),
        "age": _attr_number(sample, "age"),
        "biopsy": _attr_bool(sample, "biopsy"),
        "sample_biopsy": _attr_bool(sample, "biopsy"),
        "specimen_status": _attr_text(sample, "specimen_status"),
        "status": _attr_text(sample, "status"),
    }


def _dataset_text(group: h5py.Group, key: str) -> str:
    if key not in group:
        return ""
    return _text(group[key][()])


def _attr_text(group: h5py.Group, key: str) -> str:
    return _text(group.attrs.get(key))


def _attr_number(group: h5py.Group, key: str) -> float | None:
    value = group.attrs.get(key)
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _attr_bool(group: h5py.Group, key: str) -> bool:
    value = group.attrs.get(key)
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value) if value is not None else False


def _text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return "" if value is None else str(value)


def _normalize_side(value: Any) -> str | None:
    text = _text(value).strip().lower()
    return text if text in {"left", "right"} else None


def _canonical_position(value: str) -> str | None:
    match = re.fullmatch(
        r"P([1-3])(?:_[A-Za-z]+)?",
        value.strip(),
        flags=re.IGNORECASE,
    )
    return None if match is None else f"P{match.group(1)}"


def _record_text(record: Mapping[str, Any], key: str) -> str:
    value = record.get(key)
    return "" if value is None or pd.isna(value) else str(value)
