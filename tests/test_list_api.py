from __future__ import annotations

import asyncio
import json
import unittest
from pathlib import Path

import fserver
from starlette.requests import Request


class ListApiTests(unittest.TestCase):
    def test_directory_api_returns_entries_and_navigation(self):
        response = asyncio.run(fserver.api_list("tests/fixtures"))
        payload = json.loads(response.body)

        self.assertEqual(payload["path"], "tests/fixtures")
        self.assertTrue(payload["absolute_path"].endswith("/tests/fixtures"))
        self.assertGreaterEqual(payload["total"], 2)
        self.assertEqual(payload["breadcrumbs"][-1]["url"], "/list/tests/fixtures")

        files = {item["name"]: item for item in payload["items"]}
        self.assertEqual(files["wide.tsv"]["type"], "file")
        self.assertEqual(
            files["wide.tsv"]["url"],
            "/tsv/tests/fixtures/wide.tsv",
        )
        self.assertEqual(
            files["wide.tsv"]["preview_url"],
            "/tsv/tests/fixtures/wide.tsv",
        )
        self.assertEqual(
            files["wide.tsv"]["download_url"],
            "/download/tests/fixtures/wide.tsv",
        )
        self.assertEqual(
            files["simple.csv"]["url"],
            "/tsv/tests/fixtures/simple.csv?delim=%2C",
        )
        self.assertEqual(
            files["plain.txt"]["url"],
            "/download/tests/fixtures/plain.txt",
        )
        self.assertNotIn("preview_url", files["plain.txt"])

    def test_list_page_uses_the_tabulator_template(self):
        request = Request(
            {
                "type": "http",
                "method": "GET",
                "path": "/list/tests/fixtures",
                "raw_path": b"/list/tests/fixtures",
                "headers": [],
                "query_string": b"",
                "scheme": "http",
                "server": ("testserver", 80),
                "client": ("testclient", 1),
            }
        )
        response = asyncio.run(fserver.list_files(request, Path("tests/fixtures")))
        self.assertEqual(response.template.name, "list.html")
        self.assertIn("copy-path-button", response.body.decode())
        self.assertIn(str(Path("tests/fixtures").resolve()), response.body.decode())

    def test_absolute_paths_do_not_percent_encode_separators(self):
        absolute_directory = Path("tests/fixtures").resolve()
        files = {
            item["name"]: item
            for item in fserver.get_directory_contents(absolute_directory)
        }
        markdown_file = files["wide-table.md"]

        self.assertNotIn("%2F", markdown_file["url"])
        self.assertNotIn("%2F", markdown_file["preview_url"])
        self.assertNotIn("%2F", markdown_file["download_url"])
        self.assertTrue(markdown_file["url"].startswith("/md//"))
        self.assertTrue(markdown_file["preview_url"].startswith("/md//"))
        self.assertTrue(markdown_file["download_url"].startswith("/download//"))

        breadcrumbs = fserver.get_breadcrumbs(absolute_directory)
        self.assertTrue(breadcrumbs[-1]["url"].startswith("/list//"))
        self.assertNotIn("%2F", breadcrumbs[-1]["url"])

    def test_legacy_percent_encoded_list_urls_redirect_to_canonical_form(self):
        absolute_directory = Path("tests/fixtures").resolve()
        encoded_path = str(absolute_directory).replace("/", "%2F", 1)
        request = Request(
            {
                "type": "http",
                "method": "GET",
                "path": str(absolute_directory),
                "raw_path": f"/list/{encoded_path}".encode(),
                "headers": [],
                "query_string": b"",
                "scheme": "http",
                "server": ("testserver", 80),
                "client": ("testclient", 1),
            }
        )
        response = asyncio.run(fserver.list_files(request, absolute_directory))

        self.assertEqual(response.status_code, 308)
        self.assertEqual(
            response.headers["location"],
            f"/list//{str(absolute_directory).lstrip('/')}",
        )

    def test_unsupported_file_urls_default_to_download(self):
        file_path = Path("tests/fixtures/plain.txt")
        request = Request(
            {
                "type": "http",
                "method": "GET",
                "path": "/list/tests/fixtures/plain.txt",
                "raw_path": b"/list/tests/fixtures/plain.txt",
                "headers": [],
                "query_string": b"",
                "scheme": "http",
                "server": ("testserver", 80),
                "client": ("testclient", 1),
            }
        )
        response = asyncio.run(fserver.list_files(request, file_path))

        self.assertEqual(response.status_code, 307)
        self.assertEqual(
            response.headers["location"],
            "/download/tests/fixtures/plain.txt",
        )


if __name__ == "__main__":
    unittest.main()
