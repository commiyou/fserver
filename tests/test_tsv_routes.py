from __future__ import annotations

import asyncio
import unittest
from pathlib import Path

import fserver
from starlette.requests import Request


class TsvRouteTests(unittest.TestCase):
    def test_tsv_route_is_registered_without_tsv2_alias(self):
        route_names = {getattr(route, "name", None) for route in fserver.app.routes}
        self.assertIn("tsv", route_names)
        self.assertNotIn("tsv2", route_names)

    def test_only_the_tabular_template_exists(self):
        template_dir = Path(__file__).resolve().parents[1] / "templates"
        self.assertTrue((template_dir / "tsv.html").is_file())
        self.assertFalse((template_dir / "tsv2.html").exists())

    def test_tsv_route_uses_the_tabulator_template(self):
        request = Request(
            {
                "type": "http",
                "method": "GET",
                "path": "/tsv/tests/fixtures/wide.tsv",
                "headers": [],
                "query_string": b"",
                "scheme": "http",
                "server": ("testserver", 80),
                "client": ("testclient", 1),
            }
        )
        response = asyncio.run(
            fserver.read_tsv(request, "tests/fixtures/wide.tsv")
        )
        self.assertEqual(response.template.name, "tsv.html")
        self.assertIn("copy-path-button", response.body.decode())


if __name__ == "__main__":
    unittest.main()
