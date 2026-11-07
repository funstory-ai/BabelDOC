"""Schema-side sharing and view declarations for the owned Rust IR.

No input distribution participates in these declarations. Pools are semantic:
font definitions and their bounds, and the persistent graphic-state catalogue.
"""

POOLS = {"GraphicState", "PdfFont", "PdfFontCharBoundingBox"}
# These schema object types are shared or retained and serially mutated by stages.
# Their tree edges address a document table; the remaining structural payloads
# are immutable owned subtrees, with Arc only to preserve explicit sharing.
TABLE_TYPES = {
    "Box",
    "GraphicState",
    "PdfFont",
    "PdfFontCharBoundingBox",
    "Page",
    "PageLayout",
    "PdfParagraph",
    "PdfParagraphComposition",
    "PdfLine",
    "PdfFormula",
    "PdfSameStyleCharacters",
    "PdfSameStyleUnicodeCharacters",
    "PdfCharacter",
    "PdfStyle",
    "VisualBbox",
    "PdfRectangle",
    "PdfCurve",
    "PdfForm",
    "PdfXobject",
}
# Priorities now only preserve which path is named by handoff validation errors.
OWNER_PRIORITIES = {
    "PdfParagraph.box": 100,
    "PdfFormula.box": 90,
    "PdfCharacter.box": 80,
    "VisualBbox.box": 70,
    "PdfRectangle.box": 10,
    "PdfParagraph.pdf_style": 100,
    "PdfSameStyleUnicodeCharacters.pdf_style": 80,
    "PdfSameStyleCharacters.pdf_style": 70,
    "PdfCharacter.pdf_style": 60,
    "Page.pdf_paragraph": 100,
    "Page.pdf_character": 90,
    "Page.pdf_curve": 90,
    "Page.pdf_form": 90,
    "Page.pdf_font": 90,
    "PdfXobject.pdf_font": 80,
}

# Stage views retain compile-time write sets. ReadOnly is the narrowing target.
VIEWS = {
    "WriterView": {
        "write": ["Page", "PdfXobject", "PdfFont"],
        "narrow": "ReadOnlyView",
    },
    "LayoutParserView": {
        "write": [
            "Page",
            "PageLayout",
            "PdfParagraph",
            "PdfStyle",
            "GraphicState",
            "Box",
        ],
        "narrow": "ReadOnlyView",
    },
    "ParagraphFinderView": {
        "write": [
            "Page",
            "PageLayout",
            "PdfParagraph",
            "PdfParagraphComposition",
            "PdfLine",
            "PdfFormula",
            "PdfSameStyleCharacters",
            "PdfCharacter",
            "PdfStyle",
            "GraphicState",
            "Box",
            "VisualBbox",
        ],
        "narrow": "ReadOnlyView",
    },
    "StylesAndFormulasView": {
        "write": [
            "Page",
            "PdfParagraph",
            "PdfParagraphComposition",
            "PdfLine",
            "PdfCharacter",
            "PdfFormula",
            "PdfCurve",
            "PdfForm",
            "PdfStyle",
            "GraphicState",
        ],
        "narrow": "ReadOnlyView",
    },
    "AddDebugInformationView": {
        "write": ["Page", "PdfParagraph", "PdfStyle", "GraphicState", "Box"],
        "narrow": "ReadOnlyView",
    },
    "TranslationView": {
        "write": [
            "Page",
            "PdfParagraph",
            "PdfParagraphComposition",
            "PdfFormula",
            "PdfSameStyleCharacters",
            "PdfSameStyleUnicodeCharacters",
            "PdfStyle",
            "GraphicState",
            "Box",
        ],
        "narrow": "ReadOnlyView",
    },
    "ScanDetectionView": {
        "write": ["Page", "PdfCharacter", "PdfStyle", "GraphicState"],
        "narrow": "ReadOnlyView",
    },
    "TypesettingView": {"write": "all", "narrow": "ReadOnlyView"},
    "ReadOnlyView": {"write": [], "narrow": None},
    "GeometryView": {"write": ["Box"], "narrow": "ReadOnlyView"},
}
