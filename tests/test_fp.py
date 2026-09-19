from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import fp


class DefaultHostTests(unittest.TestCase):
    @patch.object(fp.sys, "platform", "darwin")
    @patch.object(fp.socket, "gethostname", return_value="mbp-baidu.local")
    @patch.object(fp.socket, "getfqdn", return_value="1.0.0.127.in-addr.arpa")
    def test_macos_uses_hostname_instead_of_reverse_loopback_name(self, _getfqdn, _gethostname):
        self.assertEqual(fp.default_host(), "mbp-baidu.local")

    @patch.object(fp.sys, "platform", "darwin")
    @patch.object(fp.socket, "gethostname", return_value="mbp-baidu.local")
    @patch.object(fp.socket, "getfqdn", return_value="mbp.example.com")
    def test_macos_keeps_valid_fqdn(self, _getfqdn, _gethostname):
        self.assertEqual(fp.default_host(), "mbp.example.com")

    @patch.object(fp.sys, "platform", "linux")
    @patch.object(fp.socket, "gethostname", return_value="devbox")
    @patch.object(fp.socket, "getfqdn", return_value="devbox.example.com")
    def test_other_platforms_keep_fqdn(self, _getfqdn, _gethostname):
        self.assertEqual(fp.default_host(), "devbox.example.com")


class HtmlExportTests(unittest.TestCase):
    def test_html_flag_writes_sibling_html_file(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            markdown_path = Path(temporary_directory) / "示例.md"
            markdown_path.write_text("# 标题\n\n正文。\n", encoding="utf-8")

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exit_code = fp.main(["--html", "--host", "localhost", str(markdown_path)])

            html_path = markdown_path.with_suffix(".html")
            resolved_html_path = markdown_path.resolve().with_suffix(".html")
            self.assertEqual(exit_code, 0)
            self.assertTrue(html_path.is_file())
            html = html_path.read_text(encoding="utf-8")
            self.assertIn("<!DOCTYPE html>", html)
            self.assertIn("<h1", html)
            self.assertIn("标题</h1>", html)
            self.assertIn(f"exported {resolved_html_path}", stdout.getvalue())

    def test_html_flag_is_reported_in_json_output(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            markdown_path = Path(temporary_directory) / "example.md"
            markdown_path.write_text("# Example\n", encoding="utf-8")

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exit_code = fp.main(
                    ["--html", "--format", "json", "--host", "localhost", str(markdown_path)],
                )

            result = json.loads(stdout.getvalue())
            self.assertEqual(exit_code, 0)
            self.assertEqual(result[0]["html_path"], str(markdown_path.resolve().with_suffix(".html")))

    def test_html_flag_rejects_non_markdown_files(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            text_path = Path(temporary_directory) / "example.txt"
            text_path.write_text("plain text\n", encoding="utf-8")

            with redirect_stdout(io.StringIO()):
                exit_code = fp.main(["--html", "--host", "localhost", str(text_path)])

            self.assertEqual(exit_code, 1)
            self.assertFalse(text_path.with_suffix(".html").exists())


if __name__ == "__main__":
    unittest.main()
