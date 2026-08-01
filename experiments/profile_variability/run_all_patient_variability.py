"""Preprocess all archive patients and compare within-patient profile variability."""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

from aramina.pipelines import run_preprocessing_pipeline

from .all_patient_variability import (
    all_patient_variability_figure,
    load_all_patient_profile_dataframe,
    run_all_patient_variability_analysis,
    save_all_patient_analysis,
)


def parse_args() -> ArgumentParser:
    """Return the command-line parser for the all-patient experiment."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--input-h5")
    parser.add_argument("--preprocessing-config")
    parser.add_argument("--input-joblib")
    parser.add_argument("--output-joblib")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-measurements", type=int, default=3)
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    parser.add_argument("--random-state", type=int, default=42)
    return parser


def main() -> int:
    """Run preprocessing, save the local all-patient joblib, and analyze it."""
    args = parse_args().parse_args()
    if args.input_joblib:
        if args.input_h5 or args.preprocessing_config or args.output_joblib:
            raise ValueError(
                "--input-joblib cannot be combined with H5 preprocessing arguments."
            )
        frame = load_all_patient_profile_dataframe(args.input_joblib)
    else:
        if not (args.input_h5 and args.preprocessing_config and args.output_joblib):
            raise ValueError(
                "Provide --input-joblib or all of --input-h5, --preprocessing-config, "
                "and --output-joblib."
            )
        output_joblib = Path(args.output_joblib)
        output_joblib.parent.mkdir(parents=True, exist_ok=True)
        frame = run_preprocessing_pipeline(
            args.input_h5,
            args.preprocessing_config,
            output_joblib_path=output_joblib,
            verbose=True,
        )
    analysis = run_all_patient_variability_analysis(
        frame,
        min_measurements=args.min_measurements,
        bootstrap_iterations=args.bootstrap_iterations,
        random_state=args.random_state,
    )
    output = Path(args.output_dir)
    save_all_patient_analysis(analysis, output)
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    all_patient_variability_figure(analysis.cases).savefig(
        figures / "all_patient_variability.png",
        dpi=180,
        bbox_inches="tight",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
