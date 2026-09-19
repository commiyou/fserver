"""Print fserver URLs for local files and directories."""

from __future__ import annotations

import argparse
import asyncio
import json
import socket
import sys
from pathlib import Path
from urllib.parse import quote

TABLE_EXTENSIONS = {".csv", ".json", ".ndjson", ".tsv", ".xls", ".xlsx"}


def default_host() -> str:
    """Return a browser-friendly default hostname for this machine."""
    hostname = socket.gethostname()
    fqdn = socket.getfqdn()
    if sys.platform == "darwin" and fqdn.endswith((".in-addr.arpa", ".ip6.arpa")):
        # On macOS, getfqdn() can reverse-resolve 127.0.0.1 to the unusable
        # DNS record 1.0.0.127.in-addr.arpa. The Bonjour hostname is usable.
        return hostname
    if fqdn.endswith((".in-addr.arpa", ".ip6.arpa")):
        return hostname
    return fqdn


def export_markdown_html(path: Path) -> Path:
    """Render a Markdown file and write a sibling .html file."""
    from fserver import render_markdown_file

    output_path = path.with_suffix(".html")
    html = asyncio.run(render_markdown_file(path))
    output_path.write_text(html, encoding="utf-8")
    return output_path


def _url(base_url: str, endpoint: str, path: Path, query: str = "") -> str:
    # Keep the leading slash in an absolute filesystem path. This produces
    # /md//absolute/path, which the existing catch-all routes resolve correctly.
    encoded_path = quote(path.as_posix(), safe="/")
    return f"{base_url}/{endpoint}/{encoded_path}{query}"


def build_links(path: Path, base_url: str) -> list[tuple[str, str]]:
    """Return the useful fserver links for one absolute path."""
    if path.is_dir():
        return [("list", _url(base_url, "list", path))]

    links = [
        ("list", _url(base_url, "list", path.parent)),
        ("download", _url(base_url, "download", path)),
    ]
    suffix = path.suffix.lower()
    if suffix == ".md":
        links.extend(
            [
                ("markdown", _url(base_url, "md", path)),
                ("html", _url(base_url, "md", path, "?download=1")),
            ],
        )
    elif suffix in TABLE_EXTENSIONS:
        links.extend(
            [
                ("table", _url(base_url, "tsv", path, "?reload=1")),
                ("excel", _url(base_url, "excel", path)),
            ],
        )

    return links


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fp",
        description="Print absolute-path URLs for files and directories served by fserver.",
    )
    parser.add_argument("paths", nargs="*", metavar="PATH", help="file or directory (defaults to the current directory)")
    parser.add_argument("--host", default=default_host(), help="server hostname or IP (default: local hostname)")
    parser.add_argument("--port", type=int, default=8113, help="server port (default: 8113)")
    parser.add_argument("--scheme", choices=("http", "https"), default="http", help="URL scheme (default: http)")
    parser.add_argument("--format", choices=("text", "json"), default="text", dest="output_format")
    parser.add_argument(
        "--html",
        "--export-html",
        action="store_true",
        dest="export_html",
        help="render each Markdown input to a sibling .html file",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    input_paths = args.paths or ["."]
    base_url = f"{args.scheme}://{args.host}:{args.port}"
    results: list[dict[str, object]] = []
    exit_code = 0

    for raw_path in input_paths:
        path = Path(raw_path).expanduser().resolve()
        if not path.exists():
            print(f"can NOT find {raw_path}", file=sys.stderr)
            exit_code = 1
            continue
        html_path: Path | None = None
        if args.export_html:
            if path.is_file() and path.suffix.lower() == ".md":
                try:
                    html_path = export_markdown_html(path)
                except Exception as error:  # noqa: BLE001
                    print(f"can NOT export {raw_path}: {error}", file=sys.stderr)
                    exit_code = 1
            else:
                print(f"can NOT export {raw_path}: expected a Markdown file", file=sys.stderr)
                exit_code = 1
        links = build_links(path, base_url)
        result: dict[str, object] = {
            "input": raw_path,
            "path": str(path),
            "type": "directory" if path.is_dir() else "file",
            "links": [{"kind": kind, "url": url} for kind, url in links],
        }
        if html_path is not None:
            result["html_path"] = str(html_path)
        results.append(result)

    if args.output_format == "json":
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        for index, item in enumerate(results):
            if len(results) > 1:
                print(f"[{item['path']}]")
            for link in item["links"]:  # type: ignore[union-attr]
                print(f"{link['kind']:<8} {link['url']}")  # type: ignore[index]
            if "html_path" in item:
                print(f"{'exported':<8} {item['html_path']}")
            if index < len(results) - 1:
                print()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
