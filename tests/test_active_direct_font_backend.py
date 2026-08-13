from babeldoc.format.pdf.new_parser import active_direct_font_backend


class _SubsetFontFace:
    """Minimal FreeType-like face for an unencoded Type1 subset font."""

    def get_first_char(self):
        return 0x37, 1

    def get_next_char(self, _char_code, _glyph_index):
        return 0, 0

    def get_glyph_name(self, _glyph_index):
        return b"G55"


class _CustomCharmapSubsetFontFace(_SubsetFontFace):
    charmaps = (object(),)

    def __init__(self):
        self.charmap_selected = False

    def set_charmap(self, _charmap):
        self.charmap_selected = True

    def get_first_char(self):
        return (0x37, 1) if self.charmap_selected else (0, 0)


def test_recovers_ascii_from_unencoded_subset_glyph_names():
    recovered = active_direct_font_backend.recover_unencoded_subset_map(
        _SubsetFontFace()
    )

    assert recovered == {0x37: "T"}


def test_recovers_glyph_names_from_custom_charmap():
    recovered = active_direct_font_backend.recover_unencoded_subset_map(
        _CustomCharmapSubsetFontFace()
    )

    assert recovered == {0x37: "T"}


def test_uses_recovered_map_for_unencoded_arial_subset(monkeypatch):
    monkeypatch.setattr(
        active_direct_font_backend,
        "_recover_unencoded_core_font_subset_map",
        lambda _descriptor, _font_name: {0x37: "T"},
    )

    backend = active_direct_font_backend._build_simple_font_backend(
        {
            "Subtype": "Type1",
            "BaseFont": "Arial_Fett0350.625",
            "FontDescriptor": {"FontName": "Arial_Fett0350.625"},
        },
        try_font_metrics=False,
    )

    assert backend.unicode_text(0x37, "(cid:55)") == "T"
