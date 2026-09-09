"""Regression tests for the TUI display-width truncation helpers.

The width helpers (``_truncate_display_width`` / ``_display_width``) live in
``benchmark/cli.py``; their per-character width lookup was replaced with the
``wcwidth`` package in commit ``ae828db`` while the hand-rolled grapheme
clustering was kept (``wcwidth`` does not provide grapheme segmentation). This
file pins the ASCII, CJK, and mixed ASCII+CJK truncation cases that
``tests/test_tui_cells.py`` does not already cover (that file owns the
emoji, keycap, combining-mark, and ZWJ-cluster cases).
"""
import unittest

from benchmark import cli


class TestTruncate(unittest.TestCase):
    """``_truncate_display_width`` clips to a wcwidth-measured display width."""

    def test_truncate_uses_wcwidth(self):
        """Truncation yields the expected prefix and never exceeds the budget.

        Each case asserts the exact clipped prefix (pinning the grapheme
        boundary behaviour) and that the result's measured display width --
        computed via ``wcwidth``/``wcswidth`` -- stays within ``max_width``.
        The cases below are the ASCII, CJK, and mixed ASCII+CJK shapes that
        ``tests/test_tui_cells.py`` does not already cover (it owns the
        emoji, keycap, combining-mark, and ZWJ-cluster cases).
        """
        cases = [
            # (text, max_width, expected_prefix)
            ("hello world", 5, "hello"),            # ASCII, one column each
            ("日本語のテキスト", 4, "日本"),      # CJK ideographs, two columns each
            ("ab界cd", 4, "ab界"),                # mixed ASCII + CJK
        ]
        for text, max_width, expected in cases:
            with self.subTest(text=text, max_width=max_width):
                result = cli._truncate_display_width(text, max_width)
                self.assertEqual(result, expected)
                self.assertLessEqual(cli._display_width(result), max_width)
                # The clipped prefix must be a true prefix of the input.
                self.assertTrue(text.startswith(result))
