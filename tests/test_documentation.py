from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).parents[1]
MARKDOWN_LINK = re.compile(r"\[[^]]+\]\(([^)]+)\)")


def _markdown_files() -> list[Path]:
    return [
        *ROOT.glob("*.md"),
        *ROOT.joinpath("docs").rglob("*.md"),
        *ROOT.joinpath("config").rglob("README.md"),
        *ROOT.joinpath("contracts").rglob("README.md"),
        *ROOT.joinpath("examples").rglob("README.md"),
        ROOT / "models" / "README.md",
    ]


def test_local_markdown_links_resolve() -> None:
    missing = []
    for source in _markdown_files():
        for target in MARKDOWN_LINK.findall(source.read_text(encoding="utf-8")):
            if target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            relative = target.split("#", maxsplit=1)[0]
            if relative and not (source.parent / relative).resolve().exists():
                missing.append(f"{source.relative_to(ROOT)} -> {target}")
    assert missing == []


def test_product_and_test_modules_remain_reviewable() -> None:
    oversized = []
    paths = [
        *ROOT.joinpath("src", "aramina").glob("*.py"),
        *ROOT.joinpath("tests").glob("test_*.py"),
    ]
    for path in paths:
        lines = len(path.read_text(encoding="utf-8").splitlines())
        if lines > 700:
            oversized.append(f"{path.relative_to(ROOT)}: {lines}")
    assert oversized == []


def test_prediction_contract_documents_yaml_only_reports() -> None:
    paths = [
        ROOT / "docs" / "contracts" / "prediction_config_v0_1.md",
        ROOT / "contracts" / "prediction" / "README.md",
    ]
    for path in paths:
        text = path.read_text(encoding="utf-8")
        normalized = " ".join(text.split())
        assert "YAML/JSON" not in text
        assert "JSON files are not written" in normalized
