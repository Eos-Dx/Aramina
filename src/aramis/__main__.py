"""Command-line entrypoint for Aramis product workflows."""

from __future__ import annotations

import argparse
from pathlib import Path

from .pipelines import run_preprocessing_from_config
from .prediction import run_prediction_from_config
from .training import run_training_from_config
from .training_config import available_model_recipes, describe_model_recipe
from .workflows import run_preprocess_train_from_config


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
        type=Path,
        help="Path to Aramis training YAML.",
    )
    train.add_argument(
        "--list-recipes",
        action="store_true",
        help="List model recipes available in this Aramis version.",
    )
    train.add_argument(
        "--describe-recipe",
        metavar="RECIPE_ID",
        help="Print one immutable model recipe.",
    )
    preprocess_train = subparsers.add_parser(
        "preprocess-train",
        help="Run an Aramis preprocess+train workflow YAML.",
    )
    preprocess_train.add_argument(
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
        if args.list_recipes:
            print("\n".join(available_model_recipes()))
            return 0
        if args.describe_recipe:
            print(describe_model_recipe(args.describe_recipe), end="")
            return 0
        if args.config is None:
            train.error("--config is required unless listing or describing recipes")
        artifact = run_training_from_config(args.config)
        print(f"artifact_kind={artifact['kind']}")
        print(f"run_folder={artifact['run_folder']}")
        if artifact["kind"] == "aramis_training_artifact":
            print(f"model_id={artifact['model_id']}")
            print(f"model_path={artifact['model_path']}")
        print(f"config={args.config}")
        return 0
    if args.command == "preprocess-train":
        result = run_preprocess_train_from_config(args.config)
        preprocessing_df = result["preprocessing_dataframe"]
        training_artifact = result["training_artifact"]
        print(f"preprocess_rows={len(preprocessing_df)}")
        print(f"preprocess_columns={len(preprocessing_df.columns)}")
        print(f"training_artifact_kind={training_artifact['kind']}")
        print(f"run_folder={result['run_folder']}")
        print(f"config={args.config}")
        return 0
    if args.command == "predict":
        reports = run_prediction_from_config(args.config)
        external = reports["external_report"]
        internal = reports["internal_report"]
        print(f"patient_id={external['patient_id']}")
        print(f"target_side={external['target_side']}")
        print(f"model_name={internal['model']['name']}")
        print(f"suggested_class={external['suggested_class']}")
        print(f"reliability={external['reliability']}")
        print(f"config={args.config}")
        return 0
    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
