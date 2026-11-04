from __future__ import annotations

import unittest

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


if __name__ == "__main__":
    unittest.main()
