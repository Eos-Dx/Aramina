#!/usr/bin/env python3
"""Create three real one-patient v0.3 prediction containers from an archive.

Each fixture is a standalone ``xrd-session`` with both breast sides represented
as measurement sets. Detector frames, PONI geometry and thickness metadata are
copied from the source archive; the archive itself is never modified.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py


FIXTURES = (
    {
        "name": "benign",
        "patient_id": "Nova_227",
        "target_side": "right",
        "historical_target_status": "BENIGN",
        "archive_group": "calib_20250528_085450",
        "sessions": (
            "sample_01_20250528_Nova_227_Right",
            "sample_02_20250528_Nova_227_Left",
        ),
    },
    {
        "name": "cancer",
        "patient_id": "Nova_214",
        "target_side": "left",
        "historical_target_status": "CANCER",
        "archive_group": "calib_20250514_085622",
        "sessions": (
            "sample_01_20250514_Nova_214_Right",
            "sample_02_20250514_Nova_214_Left",
        ),
    },
    {
        "name": "atypical",
        "patient_id": "Nova_257",
        "target_side": "right",
        "historical_target_status": "ATYPICAL",
        "archive_group": "calib_20250724_083952",
        "sessions": (
            "sample_01_20250724_Nova_257_Right",
            "sample_02_20250724_Nova_257_Left",
        ),
    },
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_archive", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent,
    )
    args = parser.parse_args()

    source_archive = args.source_archive.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(source_archive, "r") as source:
        for fixture in FIXTURES:
            output_path = output_dir / f"{fixture['name']}_one_patient.h5"
            _copy_patient_fixture(source, output_path, fixture)
            print(output_path)


def _copy_patient_fixture(
    source: h5py.File,
    output_path: Path,
    fixture: dict[str, object],
) -> None:
    group_name = str(fixture["archive_group"])
    source_group = source[group_name]
    session_names = tuple(str(name) for name in fixture["sessions"])
    missing = [name for name in session_names if name not in source_group]
    if missing:
        raise KeyError(f"{group_name} is missing fixture members: {missing}")

    with h5py.File(output_path, "w") as target:
        target_session_name = next(
            name
            for name in session_names
            if str(source_group[name]["sample"].attrs["side"]).casefold()
            == str(fixture["target_side"]).casefold()
        )
        contralateral_session_name = next(
            name for name in session_names if name != target_session_name
        )
        target_session = _copy_target_session(
            source_group,
            target,
            session_name=target_session_name,
        )
        _merge_contralateral_sets(
            source_group,
            target_session,
            session_name=contralateral_session_name,
        )
        _set_session_measurement_metadata(
            source_group,
            target_session,
            session_names=session_names,
            patient_id=str(fixture["patient_id"]),
            target_side=str(fixture["target_side"]),
        )

        for key, value in target_session.attrs.items():
            target.attrs[key] = value
        target.attrs.update({"format": "xrd-session", "container_type": "session"})
        target.attrs["container_id"] = f"aramina-example-{fixture['name']}"
        target.attrs["fixture_patient_id"] = str(fixture["patient_id"])
        target.attrs["fixture_target_side"] = str(fixture["target_side"])
        target.attrs["fixture_historical_target_status"] = str(
            fixture["historical_target_status"]
        )


def _copy_target_session(
    source_group: h5py.Group,
    target: h5py.File,
    *,
    session_name: str,
) -> h5py.Group:
    source_group.copy(source_group[session_name], target, name="session")
    return target["session"]


def _merge_contralateral_sets(
    source_group: h5py.Group,
    target_session: h5py.Group,
    *,
    session_name: str,
) -> None:
    source_sets = source_group[session_name]["sets"]
    target_sets = target_session["sets"]
    for source_name in sorted(source_sets):
        target_name = f"contralateral_{source_name}"
        source_sets.copy(source_sets[source_name], target_sets, name=target_name)


def _set_session_measurement_metadata(
    source_group: h5py.Group,
    target_session: h5py.Group,
    *,
    session_names: tuple[str, str],
    patient_id: str,
    target_side: str,
) -> None:
    """Promote per-side clinical and calibration fields to merged set metadata."""
    for key, value in source_group.attrs.items():
        target_session.attrs[key] = value

    side_metadata = {}
    for session_name in session_names:
        session = source_group[session_name]
        sample = session["sample"]
        side = str(sample.attrs["side"])
        side_metadata[side.casefold()] = {
            "patientId": patient_id,
            "specimenId": sample["name"][()].decode("utf-8"),
            "side": side,
            "age": sample.attrs.get("age"),
            "biopsy": sample.attrs.get("biopsy"),
            "sample_biopsy": sample.attrs.get("biopsy"),
            "specimen_status": sample.attrs.get("specimen_status"),
            "status": sample.attrs.get("status"),
            "target_side": target_side,
        }

    for set_name, set_group in target_session["sets"].items():
        side = "left" if "_Left_" in set_group.attrs["sample_name"] else "right"
        for key, value in side_metadata[side].items():
            if value is not None:
                set_group.attrs[key] = value


if __name__ == "__main__":
    main()
