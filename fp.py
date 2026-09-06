"""Print fserver URLs for local files and directories."""

from __future__ import annotations

import argparse
import json
import socket
import sys
from pathlib import Path
from urllib.parse import quote

TABLE_EXTENSIONS = {".csv", ".json", ".ndjson", ".tsv", ".xls", ".xlsx"}


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
    parser.add_argument("--host", default=socket.getfqdn(), help="server hostname or IP (default: local FQDN)")
    parser.add_argument("--port", type=int, default=8113, help="server port (default: 8113)")
    parser.add_argument("--scheme", choices=("http", "https"), default="http", help="URL scheme (default: http)")
    parser.add_argument("--format", choices=("text", "json"), default="text", dest="output_format")
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
        links = build_links(path, base_url)
        results.append(
            {
                "input": raw_path,
                "path": str(path),
                "type": "directory" if path.is_dir() else "file",
                "links": [{"kind": kind, "url": url} for kind, url in links],
            },
        )

    if args.output_format == "json":
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        for index, item in enumerate(results):
            if len(results) > 1:
                print(f"[{item['path']}]")
            for link in item["links"]:  # type: ignore[union-attr]
                print(f"{link['kind']:<8} {link['url']}")  # type: ignore[index]
            if index < len(results) - 1:
                print()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
