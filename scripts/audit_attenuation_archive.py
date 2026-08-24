"""Write reproducible raw-transmission audit artifacts for the experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

from aramina.attenuation_experiment import (
    audit_archive_transmission_metadata,
    write_archive_audit_artifacts,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive_path", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    audit = audit_archive_transmission_metadata(args.archive_path)
    write_archive_audit_artifacts(
        audit,
        archive_path=args.archive_path,
        output_dir=args.output_dir,
    )
    print(audit.status)


if __name__ == "__main__":
    main()
