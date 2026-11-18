from __future__ import annotations

import asyncio
import unittest
from pathlib import Path

import fserver


def render(text: str) -> str:
    return fserver._new_md_converter().convert(text)


class MarkdownRenderTests(unittest.TestCase):
    def test_mermaid_source_is_kept_out_of_math_and_code(self):
        html = render(
            "```mermaid\n"
            'flowchart LR\n    A[("库")] --> B["x<br/>y"]\n'
            "```\n"
        )
        self.assertIn('<div class="mermaid">', html)
        self.assertIn("A[(&quot;库&quot;)] --&gt; B[&quot;x&lt;br/&gt;y&quot;]", html)
        self.assertNotIn("arithmatex", html)
        self.assertNotIn("codehilite", html)

    def test_math_is_marked_by_parser(self):
        html = render("inline $a+b$ and\n\n$$\nx^2\n$$\n")
        self.assertIn('<span class="arithmatex">\\(a+b\\)</span>', html)
        self.assertIn('<div class="arithmatex">\\[', html)

    def test_dollars_and_brackets_in_code_or_text_are_not_math(self):
        html = render("price (a) [b] `$x$`\n\n```sh\necho $HOME $PATH\n```\n")
        self.assertNotIn("arithmatex", html)
        self.assertIn("codehilite", html)

    def test_markdown_tables_are_wrapped_for_scrolling_and_resizing(self):
        html = render("| alpha | beta |\n| --- | --- |\n| one | two |\n")
        wrapped = fserver._wrap_markdown_tables(html)
        self.assertIn('class="markdown-table-scroll"', wrapped)
        self.assertIn('tabindex="0"', wrapped)
        self.assertIn('data-markdown-table="1"', wrapped)
        self.assertIn("<table>", wrapped)

    def test_each_markdown_table_gets_a_unique_wrapper(self):
        html = (
            "<table><thead><tr><th>A</th></tr></thead></table>"
            "<p>between</p>"
            "<table><thead><tr><th>B</th></tr></thead></table>"
        )
        wrapped = fserver._wrap_markdown_tables(html)
        self.assertIn('data-markdown-table="1"', wrapped)
        self.assertIn('data-markdown-table="2"', wrapped)
        self.assertEqual(wrapped.count('class="markdown-table-scroll"'), 2)

    def test_markdown_page_uses_shared_breadcrumb_component(self):
        html = asyncio.run(
            fserver.render_markdown_file(Path("tests/fixtures/wide-table.md"))
        )
        self.assertIn('class="breadcrumbs"', html)
        self.assertIn('href="/list/"', html)
        self.assertIn(
            '<span class="current" title="wide-table.md">wide-table.md</span>',
            html,
        )
        self.assertIn("copy-path-button", html)
        self.assertIn(str(Path("tests/fixtures/wide-table.md").resolve()), html)


if __name__ == "__main__":
    unittest.main()
