import pytest
from babeldoc.format.pdf.document_il.il_version_1 import Page
from babeldoc.format.pdf.document_il.il_version_1 import PdfFont
from babeldoc.format.pdf.document_il.il_version_1 import PdfXobject
from babeldoc.format.pdf.document_il.utils.formular_helper import (
    collect_page_formula_font_ids,
)
from babeldoc.format.pdf.document_il.utils.formular_helper import is_formulas_font


@pytest.mark.parametrize("formula_pattern", [None, r".*Math.*"])
@pytest.mark.parametrize(
    ("font_name", "is_formula"),
    [
        ("NewCM10-Regular", False),
        ("NewCM10-Bold", False),
        ("NewCM10-Italic", False),
        ("ABCDEF+NewCM10-Regular", False),
        ("NewComputerModern10-Regular", False),
        ("NewCMMath-Book", True),
        ("NewCMMath-Bold", True),
        ("ABCDEF+NewCMMath-Book", True),
        ("NewComputerModernMath-Book", True),
        ("LatinModernMath-Regular", True),
        ("CMR10", False),
    ],
)
def test_new_computer_modern_font_classification(
    font_name: str, is_formula: bool, formula_pattern: str | None
):
    assert is_formulas_font(font_name, formula_pattern) is is_formula


@pytest.mark.parametrize("formula_pattern", [None, r".*Math.*"])
def test_newcm_formula_font_ids_respect_page_and_xobject_scopes(
    formula_pattern: str | None,
):
    page = Page(
        pdf_font=[
            PdfFont(name="NewCM10-Regular", font_id="F1"),
            PdfFont(name="NewCM10-Bold", font_id="F2"),
            PdfFont(name="NewCM10-Italic", font_id="F3"),
            PdfFont(name="NewCMMath-Book", font_id="F4"),
        ],
        pdf_xobject=[
            PdfXobject(
                xobj_id=1,
                pdf_font=[
                    PdfFont(name="NewCMMath-Bold", font_id="F1"),
                    PdfFont(name="NewCM10-Regular", font_id="F4"),
                ],
            ),
        ],
    )

    page_ids, xobject_ids = collect_page_formula_font_ids(page, formula_pattern)

    assert page_ids == {"F4"}
    assert xobject_ids == {1: {"F1"}}
