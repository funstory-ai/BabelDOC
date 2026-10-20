"""Regression tests for word wrapping in narrow translated PDF columns."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import pymupdf
from babeldoc.format.pdf.document_il import Box
from babeldoc.format.pdf.document_il import PdfCharacter
from babeldoc.format.pdf.document_il import PdfStyle
from babeldoc.format.pdf.document_il.il_version_1 import PdfParagraph
from babeldoc.format.pdf.document_il.midend.typesetting import Typesetting
from babeldoc.format.pdf.document_il.midend.typesetting import TypesettingUnit


class LineBreakTest(unittest.TestCase):
    """Exercise the real typesetter without models, PDF assets, or network calls."""

    def test_long_identifier_wraps_without_vertical_character_columns(self) -> None:
        """A long word must wrap horizontally instead of moving each suffix."""
        engine = object.__new__(Typesetting)
        engine.font_mapper = SimpleNamespace(base_font=pymupdf.Font("helv"))
        text = "model sentence-transformers/distiluse-base-multilingual-cased-v2 parameters"
        units = [
            TypesettingUnit(
                char=PdfCharacter(
                    char_unicode=char,
                    box=Box(0, 0, 5, 10),
                    pdf_style=PdfStyle(font_id="fixture", font_size=10),
                ),
            )
            for char in text
        ]
        paragraph = PdfParagraph(box=Box(0, 0, 100, 1000))
        with patch.object(
            engine,
            "_get_width_before_next_break_point",
            wraps=engine._get_width_before_next_break_point,
        ) as lookahead:
            laid_out, fits = engine._layout_typesetting_units(
                units,
                paragraph.box,
                1,
                1.3,
                paragraph,
            )
        self.assertLessEqual(lookahead.call_count, 6)
        self.assertTrue(fits)
        self.assertEqual(
            "".join(str(unit) for unit in laid_out),
            text,
        )
        rows = {round(unit.box.y, 3) for unit in laid_out}
        self.assertLessEqual(len(rows), 6)
        self.assertTrue(
            all(unit.box.x >= 0 and unit.box.x2 <= 100 for unit in laid_out),
        )


if __name__ == "__main__":
    unittest.main()
