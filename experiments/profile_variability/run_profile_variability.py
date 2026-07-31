"""Run the fixed target-versus-contralateral profile variability analysis."""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

from experiments.profile_variability.profile_variability import (
    load_profile_dataframe,
    paired_scatter_figure,
    q_variability_figure,
    run_variability_analysis,
    save_analysis,
)


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--input-joblib", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-measurements", type=int, default=3)
    parser.add_argument("--include-bilateral-biopsy", action="store_true")
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    frame = load_profile_dataframe(args.input_joblib)
    analysis = run_variability_analysis(
        frame,
        min_measurements=args.min_measurements,
        include_bilateral_biopsy=args.include_bilateral_biopsy,
        bootstrap_iterations=args.bootstrap_iterations,
        random_state=args.random_state,
    )
    output = Path(args.output_dir)
    save_analysis(analysis, output)
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    paired_scatter_figure(analysis.cases).savefig(
        figures / "paired_variability.png",
        dpi=180,
        bbox_inches="tight",
    )
    q_variability_figure(analysis.q_variability).savefig(
        figures / "q_dependent_variability.png",
        dpi=180,
        bbox_inches="tight",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
