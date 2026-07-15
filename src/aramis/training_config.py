"""Strict public contracts for Aramis model training."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


TRAINING_CONTRACT = "aramis_training_config_v0_1"
RECIPE_REGISTRY_CONTRACT = "aramis_model_recipe_registry_v0_1"
DEFAULT_RECIPE_REGISTRY = Path(__file__).with_name("model_recipes.yaml")
PRODUCT_EVALUATION = {
    "method": "repeated_stratified_kfold",
    "folds": 5,
    "repeats": 20,
    "random_seed": 42,
}


def load_training_config(path: str | Path) -> tuple[dict[str, Any], str]:
    """Load and strictly validate one public training YAML."""
    source = Path(path).expanduser().resolve()
    text = source.read_text(encoding="utf-8")
    config = yaml.safe_load(text)
    validate_training_config(config, source)
    return config, text


def validate_training_config(config: Any, source: str | Path) -> None:
    """Reject unknown or missing training contract fields."""
    source = Path(source)
    if not isinstance(config, dict):
        raise TypeError(f"Training config must be a mapping: {source}")
    _exact_keys(
        config,
        required={"contract", "training", "input", "output", "model", "evaluation"},
        allowed={"contract", "training", "input", "output", "model", "evaluation"},
        where="training config",
    )
    if config["contract"] != TRAINING_CONTRACT:
        raise ValueError(f"Unsupported training contract: {config['contract']!r}")
    _exact_keys(
        config["training"],
        required={
            "name",
            "version",
            "created_by",
            "created_at",
            "clinical_stage",
            "intended_use",
            "mode",
        },
        allowed={
            "name",
            "version",
            "created_by",
            "created_at",
            "clinical_stage",
            "intended_use",
            "mode",
        },
        where="training",
    )
    if config["training"]["mode"] not in {"evaluation", "final_fit"}:
        raise ValueError("training.mode must be 'evaluation' or 'final_fit'.")
    _exact_keys(
        config["input"],
        required={"dataframe_joblib_path"},
        allowed={"dataframe_joblib_path"},
        where="input",
    )
    _exact_keys(
        config["output"],
        required={"folder"},
        allowed={"folder"},
        where="output",
    )
    _exact_keys(
        config["model"],
        required={"recipe"},
        allowed={"recipe"},
        where="model",
    )
    _validate_evaluation(config["evaluation"])


def load_recipe_registry(
    path: str | Path = DEFAULT_RECIPE_REGISTRY,
) -> tuple[dict[str, Any], Path]:
    """Load the versioned model recipe registry."""
    source = Path(path).expanduser().resolve()
    registry = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(registry, dict):
        raise TypeError(f"Recipe registry must be a mapping: {source}")
    _exact_keys(
        registry,
        required={"contract", "recipes"},
        allowed={"contract", "recipes"},
        where="recipe registry",
    )
    if registry["contract"] != RECIPE_REGISTRY_CONTRACT:
        raise ValueError(f"Unsupported recipe registry contract: {registry['contract']!r}")
    if not isinstance(registry["recipes"], dict) or not registry["recipes"]:
        raise ValueError("Recipe registry requires at least one recipe.")
    return registry, source


def resolve_training_recipe(
    recipe_id: str,
    *,
    registry_path: str | Path = DEFAULT_RECIPE_REGISTRY,
) -> tuple[dict[str, Any], Path]:
    """Return one validated immutable recipe and its registry path."""
    registry, source = load_recipe_registry(registry_path)
    if recipe_id not in registry["recipes"]:
        raise ValueError(
            f"Unknown model recipe {recipe_id!r}; available: "
            f"{sorted(registry['recipes'])}"
        )
    recipe = deepcopy(registry["recipes"][recipe_id])
    _validate_recipe(recipe_id, recipe)
    return recipe, source


def available_model_recipes(
    registry_path: str | Path = DEFAULT_RECIPE_REGISTRY,
) -> list[str]:
    """Return public recipe IDs."""
    registry, _ = load_recipe_registry(registry_path)
    return sorted(registry["recipes"])


def describe_model_recipe(
    recipe_id: str,
    *,
    registry_path: str | Path = DEFAULT_RECIPE_REGISTRY,
) -> str:
    """Return one recipe as readable YAML."""
    recipe, _ = resolve_training_recipe(recipe_id, registry_path=registry_path)
    return yaml.safe_dump({recipe_id: recipe}, sort_keys=False)


def resolved_recipe_path(value: str, registry_path: Path) -> Path:
    """Resolve a recipe-owned path relative to the registry file."""
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    packaged = (registry_path.parent / path).resolve()
    if packaged.exists():
        return packaged
    return (registry_path.parents[2] / path).resolve()


def _validate_evaluation(evaluation: Any) -> None:
    if not isinstance(evaluation, dict):
        raise TypeError("evaluation must be a mapping.")
    _exact_keys(
        evaluation,
        required={"method", "folds", "repeats", "random_seed"},
        allowed={"method", "folds", "repeats", "random_seed"},
        where="evaluation",
    )
    if evaluation["method"] != PRODUCT_EVALUATION["method"]:
        raise ValueError(
            "evaluation.method must be 'repeated_stratified_kfold' for this "
            "training contract."
        )
    _validate_int_at_least(evaluation["folds"], 2, "evaluation.folds")
    _validate_int_at_least(evaluation["repeats"], 1, "evaluation.repeats")
    _validate_int_at_least(evaluation["random_seed"], 0, "evaluation.random_seed")


def _validate_int_at_least(value: Any, minimum: int, where: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{where} must be an integer.")
    if value < minimum:
        raise ValueError(f"{where} must be >= {minimum}.")


def _validate_recipe(recipe_id: str, recipe: Any) -> None:
    if not isinstance(recipe, dict):
        raise TypeError(f"Recipe {recipe_id!r} must be a mapping.")
    _exact_keys(
        recipe,
        required={
            "description",
            "model",
            "target_sensitivity",
            "prediction_preprocessing_config_path",
            "prediction_contract",
        },
        allowed={
            "description",
            "model",
            "target_sensitivity",
            "prediction_preprocessing_config_path",
            "prediction_contract",
        },
        where=f"recipe {recipe_id}",
    )
    target = float(recipe["target_sensitivity"])
    if not 0.0 < target <= 1.0:
        raise ValueError(f"Recipe {recipe_id!r} target_sensitivity must be in (0, 1].")
    model = recipe["model"]
    if not isinstance(model, dict) or model.get("selected_models") != ["M2Q"]:
        raise ValueError(f"Recipe {recipe_id!r} must select only M2Q.")
    for key in ("lr1_logreg_c", "lr2_logreg_c"):
        if float(model.get(key, 0.0)) <= 0.0:
            raise ValueError(f"Recipe {recipe_id!r} {key} must be positive.")
    contract = recipe["prediction_contract"]
    for section in ("container", "reporting", "decision"):
        if not isinstance(contract.get(section), dict):
            raise ValueError(
                f"Recipe {recipe_id!r} prediction_contract requires {section}."
            )


def _exact_keys(
    value: Any,
    *,
    required: set[str],
    allowed: set[str],
    where: str,
) -> None:
    if not isinstance(value, dict):
        raise TypeError(f"{where} must be a mapping.")
    missing = sorted(required.difference(value))
    if missing:
        raise ValueError(f"Missing {where} fields: {missing}")
    unknown = sorted(set(value).difference(allowed))
    if unknown:
        raise ValueError(f"Unknown {where} fields: {unknown}")
