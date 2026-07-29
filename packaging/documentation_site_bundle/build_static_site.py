#!/usr/bin/env python3
"""Build a dependency-free, offline technical documentation site."""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import markdown


@dataclass(frozen=True)
class Page:
    repository: str
    source: Path
    output: Path
    title: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aramina-root", type=Path, required=True)
    parser.add_argument("--xrd-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--aramina-commit", required=True)
    parser.add_argument("--xrd-commit", required=True)
    return parser.parse_args()


def title_from_markdown(path: Path) -> str:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return path.stem.replace("_", " ").replace("-", " ").title()


def document_paths(root: Path, repository: str) -> list[Path]:
    paths: set[Path] = set(root.glob("*.md"))
    if repository == "Aramina":
        for relative in (
            "docs",
            "contracts",
            "config",
            "examples",
            "models",
            "packaging",
        ):
            base = root / relative
            if base.exists():
                paths.update(base.rglob("README.md"))
        paths.update((root / "docs").rglob("*.md"))
        paths.update((root / "contracts").rglob("*.md"))
        paths.update((root / "packaging").rglob("API_CONTRACT.md"))
    else:
        paths.update((root / "src" / "xrd_preprocessing" / "docs").glob("*.md"))
    return sorted(path for path in paths if path.is_file())


def output_for_doc(root: Path, repository: str, path: Path) -> Path:
    relative = path.relative_to(root)
    if relative == Path("README.md"):
        return Path(repository.lower()) / "index.html"
    return Path(repository.lower()) / relative.with_suffix(".html")


def output_for_source(root: Path, repository: str, path: Path) -> Path:
    relative = path.relative_to(root / "src")
    return Path("source") / repository.lower() / relative.with_suffix(".html")


def relative_url(source: Path, target: Path) -> str:
    return os.path.relpath(target, start=source.parent).replace(os.sep, "/")


def stylesheet() -> str:
    return """
:root { --ink: #18212b; --muted: #5d6b78; --edge: #d9e0e6; --blue: #0b5e8e;
  --panel: #f6f8fa; --code: #17212b; --accent: #e87d24; }
* { box-sizing: border-box; }
body { margin: 0; color: var(--ink); background: #fff; font-family: Arial, Helvetica, sans-serif;
  line-height: 1.55; }
header { border-bottom: 1px solid var(--edge); padding: 18px 28px; display: flex; gap: 24px;
  align-items: baseline; background: #fff; position: sticky; top: 0; z-index: 2; }
header a { color: var(--blue); text-decoration: none; font-weight: 700; }
header small { color: var(--muted); }
main { max-width: 1180px; margin: 0 auto; padding: 30px 28px 64px; }
h1, h2, h3 { line-height: 1.2; color: #102a3c; }
h1 { font-size: 30px; margin-top: 0; } h2 { margin-top: 32px; }
a { color: var(--blue); } p, li { max-width: 920px; }
table { border-collapse: collapse; width: 100%; margin: 18px 0; }
th, td { border: 1px solid var(--edge); padding: 8px 10px; text-align: left; vertical-align: top; }
th { background: var(--panel); } code { background: var(--panel); padding: 1px 4px; }
pre { overflow-x: auto; background: var(--code); color: #edf4f8; padding: 16px; border-radius: 4px; }
pre code { background: transparent; padding: 0; } blockquote { border-left: 4px solid var(--accent);
  margin-left: 0; padding-left: 14px; color: var(--muted); }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 16px; }
.card { border: 1px solid var(--edge); padding: 18px; border-radius: 6px; background: #fff; }
.meta { color: var(--muted); font-size: 14px; }
.source-path { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; color: var(--muted); }
.search { width: 100%; max-width: 780px; font-size: 18px; padding: 12px; border: 1px solid var(--edge); }
.search-result { margin: 14px 0; padding: 14px; border-bottom: 1px solid var(--edge); }
.search-result p { margin: 5px 0 0; color: var(--muted); }
details { border: 1px solid var(--edge); border-radius: 5px; padding: 10px 14px; margin: 10px 0; }
summary { cursor: pointer; font-weight: 700; } .footer { color: var(--muted); margin-top: 48px;
  border-top: 1px solid var(--edge); padding-top: 16px; font-size: 13px; }
@media (max-width: 640px) { header { padding: 14px; gap: 12px; } main { padding: 22px 16px; } }
""".strip()


def shell(
    title: str,
    body: str,
    current: Path,
    *,
    source_path: str | None = None,
    navigation: str = "",
) -> str:
    root = relative_url(current, Path("index.html"))
    css = relative_url(current, Path("assets") / "site.css")
    aramina_docs = relative_url(current, Path("aramina") / "index.html")
    xrd_docs = relative_url(current, Path("xrd-preprocessing") / "index.html")
    source_note = (
        f'<p class="source-path">Source: {html.escape(source_path)}</p>' if source_path else ""
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)} | Aramina Technical Documentation</title><link rel="stylesheet" href="{css}"></head>
<body><header><a href="{root}">Aramina Technical Documentation</a><a href="{aramina_docs}">Aramina docs</a>
<a href="{xrd_docs}">XRD-preprocessing docs</a><a href="{relative_url(current, Path('search.html'))}">Search</a>
<small>Fixed offline snapshot</small></header><main><h1>{html.escape(title)}</h1>{source_note}{navigation}{body}
<p class="footer">Research-draft decision-support documentation. Not for autonomous diagnosis.</p></main></body></html>"""


def markdown_html(text: str, page: Page, by_source: dict[Path, Page]) -> str:
    renderer = markdown.Markdown(extensions=["fenced_code", "tables", "toc", "sane_lists"])
    rendered = renderer.convert(text)

    def replace_link(match: re.Match[str]) -> str:
        href = match.group(1)
        destination, separator, fragment = href.partition("#")
        if not destination.endswith(".md"):
            return match.group(0)
        candidate = (page.source.parent / destination).resolve()
        target = by_source.get(candidate)
        if target is None:
            return match.group(0)
        updated = relative_url(page.output, target.output)
        return f'href="{updated}{separator}{fragment}"'

    return re.sub(r'href="([^"]+)"', replace_link, rendered)


def source_html(path: Path) -> str:
    return f"<pre><code>{html.escape(path.read_text(encoding='utf-8'))}</code></pre>"


def write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def documentation_index(root: Path, pages: list[Page], repository: str) -> str:
    """Render direct links to every documentation page for one repository."""
    groups: dict[str, list[Page]] = {}
    for page in pages:
        if page.repository != repository or page.source == root / "README.md":
            continue
        relative = page.source.relative_to(root)
        group = str(relative.parent) if relative.parent != Path(".") else "project root"
        groups.setdefault(group, []).append(page)

    sections = [
        "<h2>Documentation index</h2>",
        "<p>Every Markdown document in this fixed repository snapshot is listed below. "
        "Select any title to read it; use Search for full-text lookup across documents and Python source.</p>",
    ]
    for group in sorted(groups):
        sections.append(f"<details open><summary>{html.escape(group)}</summary><ul>")
        for page in sorted(groups[group], key=lambda item: item.source.name.lower()):
            relative = page.source.relative_to(root).as_posix()
            sections.append(
                f'<li><a href="{relative_url(Path(repository.lower()) / "index.html", page.output)}">'
                f"{html.escape(page.title)}</a> <span class=\"source-path\">{html.escape(relative)}</span></li>"
            )
        sections.append("</ul></details>")
    return "\n".join(sections)


def page_navigation(page: Page, pages: list[Page]) -> str:
    """Provide direct previous/next navigation inside one repository's docs."""
    repository_pages = sorted(
        (item for item in pages if item.repository == page.repository),
        key=lambda item: item.source.as_posix().lower(),
    )
    position = repository_pages.index(page)
    links = [
        f'<a href="{relative_url(page.output, Path(page.repository.lower()) / "index.html")}">'
        "Documentation index</a>"
    ]
    if position:
        previous = repository_pages[position - 1]
        links.append(f'<a href="{relative_url(page.output, previous.output)}">Previous: {html.escape(previous.title)}</a>')
    if position + 1 < len(repository_pages):
        following = repository_pages[position + 1]
        links.append(f'<a href="{relative_url(page.output, following.output)}">Next: {html.escape(following.title)}</a>')
    return '<p class="meta">Browse: ' + " | ".join(links) + "</p>"


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    if output.exists():
        shutil.rmtree(output)
    (output / "assets").mkdir(parents=True)
    (output / "assets" / "site.css").write_text(stylesheet(), encoding="utf-8")

    repositories = (("Aramina", args.aramina_root.resolve()), ("XRD-preprocessing", args.xrd_root.resolve()))
    pages: list[Page] = []
    for name, root in repositories:
        for source in document_paths(root, name):
            pages.append(Page(name, source.resolve(), output_for_doc(root, name, source), title_from_markdown(source)))
    by_source = {page.source: page for page in pages}

    search_entries: list[dict[str, str]] = []
    for page in pages:
        content = page.source.read_text(encoding="utf-8")
        rendered = markdown_html(content, page, by_source)
        if page.repository == "Aramina" and page.source == args.aramina_root / "README.md":
            rendered += documentation_index(args.aramina_root, pages, "Aramina")
        if page.repository == "XRD-preprocessing" and page.source == args.xrd_root / "README.md":
            rendered += documentation_index(args.xrd_root, pages, "XRD-preprocessing")
        write(
            output / page.output,
            shell(
                page.title,
                rendered,
                page.output,
                source_path=str(page.source),
                navigation=page_navigation(page, pages),
            ),
        )
        search_entries.append({"title": page.title, "url": page.output.as_posix(), "text": content[:1000]})

    source_entries: list[tuple[str, Path, Path]] = []
    for name, root in repositories:
        package = root / "src" / ("aramina" if name == "Aramina" else "xrd_preprocessing")
        for source in sorted(package.rglob("*.py")):
            out = output_for_source(root, name, source)
            source_entries.append((name, source, out))
            write(output / out, shell(source.name, source_html(source), out, source_path=str(source.relative_to(root))))
            search_entries.append({"title": f"{name} source: {source.name}", "url": out.as_posix(), "text": str(source.relative_to(root))})

    cards = []
    for name, root in repositories:
        root_page = next(page for page in pages if page.repository == name and page.source == root / "README.md")
        docs = [page for page in pages if page.repository == name]
        cards.append(
            f'<section class="card"><h2>{html.escape(name)}</h2><p>{len(docs)} documentation pages and '
            f'{sum(1 for item in source_entries if item[0] == name)} Python source files.</p>'
            f'<p><a href="{root_page.output.as_posix()}">Open documentation</a></p></section>'
        )
    overview = f"""
<p class="meta">Built {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')} from fixed <code>main</code> commits.</p>
<div class="grid">{''.join(cards)}</div>
<h2>Fixed source revisions</h2>
<table><thead><tr><th>Repository</th><th>Commit</th><th>Remote</th></tr></thead><tbody>
<tr><td>Aramina</td><td><code>{args.aramina_commit}</code></td><td><code>https://github.com/Eos-Dx/Aramina.git</code></td></tr>
<tr><td>XRD-preprocessing</td><td><code>{args.xrd_commit}</code></td><td><code>https://github.com/Eos-Dx/XRD-preprocessing.git</code></td></tr>
</tbody></table>
<h2>Searchable source</h2><p>Use <a href="search.html">Search</a> to find contracts, API fields, preprocessing steps, or source modules.</p>
<h2>Python source index</h2><details><summary>Aramina source files</summary><ul>"""
    overview += "".join(
        f'<li><a href="{out.as_posix()}">{html.escape(str(source.relative_to(args.aramina_root)))}</a></li>'
        for name, source, out in source_entries if name == "Aramina"
    )
    overview += "</ul></details><details><summary>XRD-preprocessing source files</summary><ul>"
    overview += "".join(
        f'<li><a href="{out.as_posix()}">{html.escape(str(source.relative_to(args.xrd_root)))}</a></li>'
        for name, source, out in source_entries if name == "XRD-preprocessing"
    )
    overview += "</ul></details>"
    write(output / "index.html", shell("Aramina and XRD-preprocessing", overview, Path("index.html")))

    search_data = json.dumps(search_entries).replace("</", "<\\/")
    search_body = f"""
<p>Search titles, contracts, technical documentation, and Python source paths in this fixed snapshot.</p>
<input class="search" id="query" placeholder="e.g. prediction config, azimuthal integration, H5" autofocus>
<div id="results"></div>
<script>const entries={search_data}; const results=document.getElementById('results');
function escapeHtml(value){{return value.replace(/[&<>\"]/g, char => ({{'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;'}}[char]));}}
function render(){{const term=document.getElementById('query').value.trim().toLowerCase(); const found=entries.filter(e => !term || (e.title+' '+e.text).toLowerCase().includes(term)).slice(0,80);
results.innerHTML=found.map(e => `<div class="search-result"><a href="${{e.url}}"><strong>${{escapeHtml(e.title)}}</strong></a><p>${{escapeHtml(e.text.replace(/\\s+/g,' ').slice(0,260))}}</p></div>`).join('') || '<p>No matching documentation or source page.</p>';}}
document.getElementById('query').addEventListener('input',render); render();</script>"""
    write(output / "search.html", shell("Search", search_body, Path("search.html")))

    manifest = {
        "contract": "aramina_offline_documentation_site_v0_1",
        "aramina": {"branch": "main", "commit": args.aramina_commit, "remote": "https://github.com/Eos-Dx/Aramina.git"},
        "xrd_preprocessing": {"branch": "main", "commit": args.xrd_commit, "remote": "https://github.com/Eos-Dx/XRD-preprocessing.git"},
        "documentation_pages": len(pages),
        "source_code_pages": len(source_entries),
    }
    (output / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
