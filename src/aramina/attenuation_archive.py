"""Raw transmission-archive audit for the attenuation experiment."""

from __future__ import annotations

from collections.abc import Sequence
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd

from .attenuation_contract import (
    ArchiveTransmissionAudit,
    STANDARDIZED_ATTENUATION_POSITIONS,
    VALIDATED_ATTENUATION_STATUS,
    _as_bool,
    _canonical_kind,
    _first_present,
    _has_text,
    _normalize_position,
    _normalize_side,
    _normalized_text,
    _numeric_or_nan,
    _text,
)
from .model_utils import LABEL_MAP


def audit_archive_transmission_metadata(
    archive_path: str | Path,
    *,
    standardized_positions: Sequence[str] = STANDARDIZED_ATTENUATION_POSITIONS,
) -> ArchiveTransmissionAudit:
    """Audit raw transmission metadata without deriving attenuation values."""
    path = Path(archive_path)
    if not path.is_file():
        raise FileNotFoundError(f"Transmission archive does not exist: {path}")
    positions = tuple(
        str(position).strip().upper() for position in standardized_positions
    )
    records: list[dict[str, Any]] = []
    session_records: list[dict[str, Any]] = []
    with h5py.File(path, "r") as archive:
        for calibration_name, calibration_group in archive.items():
            if not isinstance(calibration_group, h5py.Group):
                continue
            for session_name, session in calibration_group.items():
                if not _is_sample_session(session):
                    continue
                session_path = f"/{calibration_name}/{session_name}"
                session_record, set_records = _archive_session_records(
                    session,
                    session_path=session_path,
                )
                session_records.append(session_record)
                records.extend(set_records)
    inventory = pd.DataFrame(records, columns=_ARCHIVE_INVENTORY_COLUMNS)
    coverage = _archive_coverage(
        pd.DataFrame(session_records, columns=_ARCHIVE_SESSION_COLUMNS),
        inventory,
        standardized_positions=positions,
    )
    valid = not inventory.empty and bool(
        coverage["validated_three_point_eligible"].any()
    )
    return ArchiveTransmissionAudit(
        inventory=inventory,
        coverage=coverage,
        status="available" if valid else "unavailable",
        unavailable_reason=(
            "No raw transmission set supplies a validated attenuation value, "
            "formula identifier, reference identifier, provenance status, and units."
            if not valid
            else ""
        ),
    )


def write_archive_audit_artifacts(
    audit: ArchiveTransmissionAudit,
    *,
    archive_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Write reproducible raw inventory, coverage, and status artifacts."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    inventory_path = output / "transmission_inventory.csv"
    coverage_path = output / "transmission_coverage.csv"
    status_path = output / "attenuation_archive_status.json"
    audit.inventory.to_csv(inventory_path, index=False)
    audit.coverage.to_csv(coverage_path, index=False)
    status_path.write_text(
        json.dumps(
            {
                "archive_path": str(Path(archive_path).resolve()),
                "status": audit.status,
                "unavailable_reason": audit.unavailable_reason,
                "transmission_sets": int(len(audit.inventory)),
                "raw_three_point_breast_sessions": int(
                    audit.coverage["raw_three_point_pairing"].sum()
                ),
                "canonical_labelled_raw_three_point_breast_sessions": int(
                    (
                        audit.coverage["product_label"].notna()
                        & audit.coverage["raw_three_point_pairing"].eq(1)
                    ).sum()
                ),
                "validated_three_point_breast_sessions": int(
                    audit.coverage["validated_three_point_eligible"].sum()
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "inventory": inventory_path,
        "coverage": coverage_path,
        "status": status_path,
    }


def _archive_session_records(
    session: h5py.Group,
    *,
    session_path: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    sample = session["sample"]
    patient_id = _dataset_text(sample, "patient_name")
    side = _normalize_side(sample.attrs.get("side"))
    specimen_status = _text(sample.attrs.get("specimen_status", ""))
    set_rows = []
    for set_name, set_group in session["sets"].items():
        if not isinstance(set_group, h5py.Group):
            continue
        set_rows.append(
            {
                "set_name": str(set_name),
                "set_pk": _numeric_or_nan(set_group.attrs.get("set_pk")),
                "measurement_type_name": _text(
                    set_group.attrs.get("measurement_type_name", set_name)
                ),
                "position": _normalize_position(set_group.attrs.get("position")),
                "group": set_group,
            }
        )
    set_rows.sort(key=lambda row: (row["set_pk"], row["set_name"]))
    records = []
    for index, row in enumerate(set_rows):
        if _canonical_kind(row["measurement_type_name"]) != "sampletransmission":
            continue
        group = row["group"]
        next_row = set_rows[index + 1] if index + 1 < len(set_rows) else None
        paired_main = bool(
            next_row
            and _canonical_kind(next_row["measurement_type_name"]) == "samplemain"
            and next_row["position"] == row["position"]
            and next_row["set_pk"] == row["set_pk"] + 1
        )
        metadata = _json_dataset(group, "metadata")
        processing = _json_dataset(group.get("processing"), "config")
        reference_id = _first_present(
            group.attrs,
            metadata,
            processing,
            keys=("reference_set_id", "reference_set_uid", "reference_id"),
        )
        formula_id = _first_present(
            group.attrs,
            metadata,
            processing,
            keys=("attenuation_formula_id", "optical_density_formula_id"),
        )
        attenuation_value = _first_present(
            group.attrs,
            metadata,
            processing,
            keys=("attenuation_value", "attenuation_coefficient", "optical_density"),
        )
        provenance_status = _first_present(
            group.attrs,
            metadata,
            processing,
            keys=("attenuation_provenance_status",),
        )
        units = _first_present(
            group.attrs,
            metadata,
            processing,
            keys=("attenuation_units", "optical_density_units"),
        )
        usable = (
            np.isfinite(_numeric_or_nan(attenuation_value))
            and _normalized_text(provenance_status)
            == _normalized_text(VALIDATED_ATTENUATION_STATUS)
            and _has_text(formula_id)
            and _has_text(reference_id)
            and _has_text(units)
        )
        records.append(
            {
                "session_path": session_path,
                "patientId": patient_id,
                "side": side or "",
                "specimen_status": specimen_status,
                "product_label": LABEL_MAP.get(specimen_status),
                "biopsy": _as_bool(sample.attrs.get("biopsy", False)),
                "set_name": row["set_name"],
                "set_pk": row["set_pk"],
                "position": row["position"] or "",
                "measurement_type_name": row["measurement_type_name"],
                "phase": _text(group.attrs.get("phase", "")),
                "transmission_pct": _numeric_or_nan(group.attrs.get("transmission_pct")),
                "correction_factor": _numeric_or_nan(
                    group.attrs.get("correction_factor")
                ),
                "paired_main_adjacent": int(paired_main),
                "paired_main_set_name": next_row["set_name"] if paired_main else "",
                "explicit_reference_id": _text(reference_id),
                "attenuation_value": _numeric_or_nan(attenuation_value),
                "attenuation_formula_id": _text(formula_id),
                "attenuation_provenance_status": _text(provenance_status),
                "attenuation_units": _text(units),
                "attenuation_input_usable": int(usable),
                "availability_reason": "" if usable else _archive_unavailable_reason(
                    attenuation_value=attenuation_value,
                    provenance_status=provenance_status,
                    formula_id=formula_id,
                    reference_id=reference_id,
                    units=units,
                ),
            }
        )
    return (
        {
            "session_path": session_path,
            "patientId": patient_id,
            "side": side or "",
            "specimen_status": specimen_status,
            "product_label": LABEL_MAP.get(specimen_status),
            "biopsy": _as_bool(sample.attrs.get("biopsy", False)),
        },
        records,
    )


def _archive_coverage(
    sessions: pd.DataFrame,
    inventory: pd.DataFrame,
    *,
    standardized_positions: Sequence[str],
) -> pd.DataFrame:
    rows = []
    for session in sessions.itertuples(index=False):
        session_inventory = inventory[inventory["session_path"] == session.session_path]
        positions = set(
            session_inventory.loc[
                session_inventory["paired_main_adjacent"].eq(1), "position"
            ]
        )
        usable_positions = set(
            session_inventory.loc[
                session_inventory["paired_main_adjacent"].eq(1)
                & session_inventory["attenuation_input_usable"].eq(1),
                "position",
            ]
        )
        paired_complete = set(standardized_positions).issubset(positions)
        validated_complete = set(standardized_positions).issubset(usable_positions)
        rows.append(
            {
                "session_path": session.session_path,
                "patientId": session.patientId,
                "side": session.side,
                "specimen_status": session.specimen_status,
                "product_label": session.product_label,
                "biopsy": int(session.biopsy),
                "standardized_transmission_sets": int(
                    session_inventory["position"].isin(standardized_positions).sum()
                ),
                "raw_paired_positions": ",".join(
                    position
                    for position in standardized_positions
                    if position in positions
                ),
                "raw_three_point_pairing": int(paired_complete),
                "validated_three_point_eligible": int(validated_complete),
                "availability_reason": "" if validated_complete else (
                    "validated_attenuation_input_unavailable"
                ),
            }
        )
    return pd.DataFrame(rows)


def _is_sample_session(session: h5py.Group) -> bool:
    return isinstance(session, h5py.Group) and "sample" in session and "sets" in session


def _dataset_text(group: h5py.Group, name: str) -> str:
    return _text(group[name][()]) if name in group else ""


def _json_dataset(group: h5py.Group | None, name: str) -> dict[str, Any]:
    if group is None or name not in group:
        return {}
    try:
        value = json.loads(_text(group[name][()]))
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _archive_unavailable_reason(
    *,
    attenuation_value: Any,
    provenance_status: Any,
    formula_id: Any,
    reference_id: Any,
    units: Any,
) -> str:
    missing = []
    if not np.isfinite(_numeric_or_nan(attenuation_value)):
        missing.append("attenuation_value_missing")
    if _normalized_text(provenance_status) != _normalized_text(
        VALIDATED_ATTENUATION_STATUS
    ):
        missing.append("provenance_status_not_measured_validated")
    if not _has_text(formula_id):
        missing.append("formula_id_missing")
    if not _has_text(reference_id):
        missing.append("reference_id_missing")
    if not _has_text(units):
        missing.append("units_missing")
    return ";".join(missing)


_ARCHIVE_SESSION_COLUMNS = (
    "session_path",
    "patientId",
    "side",
    "specimen_status",
    "product_label",
    "biopsy",
)
_ARCHIVE_INVENTORY_COLUMNS = (
    "session_path",
    "patientId",
    "side",
    "specimen_status",
    "product_label",
    "biopsy",
    "set_name",
    "set_pk",
    "position",
    "measurement_type_name",
    "phase",
    "transmission_pct",
    "correction_factor",
    "paired_main_adjacent",
    "paired_main_set_name",
    "explicit_reference_id",
    "attenuation_value",
    "attenuation_formula_id",
    "attenuation_provenance_status",
    "attenuation_units",
    "attenuation_input_usable",
    "availability_reason",
)
