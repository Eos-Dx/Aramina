"""Command-line entrypoint for Aramina product workflows."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .pipelines import run_preprocessing_from_config
from .prediction import run_prediction_from_config
from .promotion import promote_model_run
from .training import run_training_from_config
from .training_config import available_product_models, describe_product_model
from .workflows import run_preprocess_train_from_config


def _add_verbose_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show preprocessing and training progress.",
    )


def _configure_logging(verbose: bool) -> None:
    if verbose:
        logging.basicConfig(
            level=logging.WARNING,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )
        logging.getLogger("aramina").setLevel(logging.INFO)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="aramina")
    subparsers = parser.add_subparsers(dest="command", required=True)

    preprocess = subparsers.add_parser(
        "preprocess",
        help="Build an Aramina preprocessing DataFrame from a YAML config.",
    )
    preprocess.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to Aramina preprocessing YAML.",
    )
    _add_verbose_argument(preprocess)
    train = subparsers.add_parser(
        "train",
        help="Train an Aramina research-draft model from a YAML config.",
    )
    train.add_argument(
        "--config",
        type=Path,
        help="Path to Aramina training YAML.",
    )
    train.add_argument(
        "--list-models",
        action="store_true",
        help="List product models available in this Aramina version.",
    )
    train.add_argument(
        "--describe-model",
        metavar="MODEL_NAME",
        help="Print one fixed product model definition.",
    )
    _add_verbose_argument(train)
    preprocess_train = subparsers.add_parser(
        "preprocess-train",
        help="Run Aramina preprocessing and training from one YAML config.",
    )
    preprocess_train.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to Aramina preprocessing-and-training YAML.",
    )
    _add_verbose_argument(preprocess_train)
    predict = subparsers.add_parser(
        "predict",
        help="Run one-patient Aramina research-draft decision support.",
    )
    predict.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to Aramina prediction YAML.",
    )
    _add_verbose_argument(predict)
    promote = subparsers.add_parser(
        "promote",
        help="Copy one reviewed final-fit run into the immutable models directory.",
    )
    promote.add_argument(
        "--run-folder",
        required=True,
        type=Path,
        help="Completed train or preprocess-train run containing model.joblib.",
    )
    promote.add_argument(
        "--models-root",
        type=Path,
        help="Optional destination root; defaults to the project models directory.",
    )

    args = parser.parse_args(argv)
    _configure_logging(getattr(args, "verbose", False))
    if args.command == "preprocess":
        kwargs = {"verbose": True} if args.verbose else {}
        df = run_preprocessing_from_config(args.config, **kwargs)
        print(f"rows={len(df)}")
        print(f"columns={len(df.columns)}")
        print(f"config={args.config}")
        return 0
    if args.command == "train":
        if args.list_models:
            print("\n".join(available_product_models()))
            return 0
        if args.describe_model:
            print(describe_product_model(args.describe_model), end="")
            return 0
        if args.config is None:
            train.error("--config is required unless listing or describing models")
        artifact = run_training_from_config(args.config)
        print(f"artifact_kind={artifact['kind']}")
        print(f"run_folder={artifact['run_folder']}")
        if artifact["kind"] == "aramina_training_artifact":
            print(f"model_id={artifact['model_id']}")
            print(f"model_path={artifact['model_path']}")
        print(f"config={args.config}")
        return 0
    if args.command == "preprocess-train":
        kwargs = {"verbose": True} if args.verbose else {}
        result = run_preprocess_train_from_config(args.config, **kwargs)
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
        print(f"biopsy_required={external['biopsy_required']}")
        print(f"risk_probability={external['risk_probability']:.5f}")
        print(f"reliability={external['reliability']}")
        print(f"config={args.config}")
        return 0
    if args.command == "promote":
        promoted = promote_model_run(args.run_folder, models_root=args.models_root)
        print(f"model_id={promoted['model_id']}")
        print(f"artifact_sha256={promoted['artifact_sha256']}")
        print(f"model_folder={promoted['model_folder']}")
        return 0
    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
