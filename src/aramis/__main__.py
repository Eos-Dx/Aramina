"""Command-line entrypoint for Aramis product workflows."""

from __future__ import annotations

import argparse
from pathlib import Path

from .pipelines import run_preprocessing_from_config
from .prediction import run_prediction_from_config
from .training import run_training_from_config
from .workflows import run_workflow_from_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="aramis")
    subparsers = parser.add_subparsers(dest="command", required=True)

    preprocess = subparsers.add_parser(
        "preprocess",
        help="Build an Aramis preprocessing DataFrame from a YAML config.",
    )
    preprocess.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to Aramis preprocessing YAML.",
    )
    train = subparsers.add_parser(
        "train",
        help="Train an Aramis research-draft model from a YAML config.",
    )
    train.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to Aramis training YAML.",
    )
    run = subparsers.add_parser(
        "run",
        help="Run an Aramis preprocess+train workflow YAML.",
    )
    run.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to Aramis workflow YAML.",
    )
    predict = subparsers.add_parser(
        "predict",
        help="Run one-patient Aramis research-draft decision support.",
    )
    predict.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to Aramis prediction YAML.",
    )

    args = parser.parse_args(argv)
    if args.command == "preprocess":
        df = run_preprocessing_from_config(args.config)
        print(f"rows={len(df)}")
        print(f"columns={len(df.columns)}")
        print(f"config={args.config}")
        return 0
    if args.command == "train":
        artifact = run_training_from_config(args.config)
        print(f"model_type={artifact['model_type']}")
        print(f"branch={artifact['metadata']['branch']}")
        metric_summary = artifact["metric_summary"]
        if "model_name" in metric_summary.columns:
            for row in metric_summary.itertuples(index=False):
                print(
                    f"{row.model_name}: "
                    f"roc_auc_mean={row.roc_auc_mean:.6f} "
                    f"specificity_target_mean={row.specificity_target_mean:.6f}"
                )
        else:
            row = metric_summary.iloc[0]
            print(f"roc_auc_mean={row['roc_auc_mean']:.6f}")
        print(f"config={args.config}")
        return 0
    if args.command == "run":
        result = run_workflow_from_config(args.config)
        preprocessing_df = result["preprocessing_dataframe"]
        training_artifact = result["training_artifact"]
        if preprocessing_df is not None:
            print(f"preprocess_rows={len(preprocessing_df)}")
            print(f"preprocess_columns={len(preprocessing_df.columns)}")
        if training_artifact is not None:
            print(f"model_type={training_artifact['model_type']}")
            print(f"branch={training_artifact['metadata']['branch']}")
        print(f"config={args.config}")
        return 0
    if args.command == "predict":
        report = run_prediction_from_config(args.config)
        print(f"patient_id={report['patient_id']}")
        print(f"target_side={report['target_side']}")
        print(f"model_name={report['model_name']}")
        print(f"p_cancer={report['p_cancer']:.6f}")
        print(f"suggested_class={report['suggested_class']}")
        print(f"reliability={report['reliability']}")
        print(f"config={args.config}")
        return 0
    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
