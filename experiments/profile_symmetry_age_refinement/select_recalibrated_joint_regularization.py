"""Exploratory train-all regularization selection for joint-refinement research.

The full evaluator performs this selection separately inside each outer train
fold. This helper only writes a transparent train-all selection record.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from recalibrated_joint_data import (
    full_chain_meta_pairs,
    load_input_dataframe,
    model_columns,
)
from recalibrated_joint_selection import ABLATIONS, DEFAULT_C_GRID, select_ablation_regularization


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-joblib", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def run_selection(
    dataframe: pd.DataFrame,
    output_dir: str | Path,
    *,
    candidate_c: tuple[float, ...] = DEFAULT_C_GRID,
    lr1_c: float = 0.1,
    inner_lr1_splits: int = 5,
    meta_splits: int = 4,
    random_state: int = 42,
) -> dict[str, Any]:
    """Select each ablation independently from cached full-chain folds."""
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    pairs = full_chain_meta_pairs(
        dataframe,
        model_columns(),
        lr1_c=lr1_c,
        meta_splits=meta_splits,
        random_state=random_state,
        outer_split_id="train_all_selection",
        inner_lr1_splits=inner_lr1_splits,
    )
    records: list[pd.DataFrame] = []
    selected: dict[str, dict[str, float]] = {}
    for index, ablation in enumerate(ABLATIONS):
        parameters, rows, _ = select_ablation_regularization(
            pairs,
            ablation=ablation,
            candidate_c=candidate_c,
            random_state=random_state + index * 1_000,
        )
        selected[ablation] = parameters
        records.append(rows)
    selection = pd.concat(records, ignore_index=True)
    selection.to_csv(output / "regularization_selection.csv", index=False)
    payload = {
        "experiment": "recalibrated_joint_regularization_selection",
        "status": "research_only_not_independent_validation",
        "selection_protocol": {
            "input_features": "strictly_nested_full_chain_lr1_oof",
            "candidate_c": [float(value) for value in candidate_c],
            "selection": "coordinate_within_each_ablation_only",
            "primary_metric": "full_chain_meta_oof_log_loss",
            "tie_breakers": ["lower_brier_score", "higher_roc_auc", "smaller_c"],
            "selected_grid_boundary_recorded": True,
        },
        "selected_regularization_by_ablation": selected,
        "outputs": {"regularization_selection": "regularization_selection.csv"},
    }
    (output / "regularization_selection.yaml").write_text(
        yaml.safe_dump(payload, sort_keys=False), encoding="utf-8"
    )
    return payload


def main() -> None:
    args = parse_args()
    payload = run_selection(load_input_dataframe(args.input_joblib), args.output_dir)
    print(yaml.safe_dump(payload["selected_regularization_by_ablation"], sort_keys=False))


if __name__ == "__main__":
    main()
