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


class _ConflictingCharmapsSubsetFontFace:
    charmaps = ("first", "second")

    def __init__(self):
        self.selected_charmap = None

    def set_charmap(self, charmap):
        self.selected_charmap = charmap

    def get_first_char(self):
        return 0x37, 1

    def get_next_char(self, _char_code, _glyph_index):
        return 0, 0

    def get_glyph_name(self, _glyph_index):
        return b"G55" if self.selected_charmap == "first" else b"G56"


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


def test_uses_first_charmap_with_a_recoverable_mapping():
    recovered = active_direct_font_backend.recover_unencoded_subset_map(
        _ConflictingCharmapsSubsetFontFace()
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


def test_does_not_recover_an_arial_subset_with_an_explicit_encoding(monkeypatch):
    recovered_called = False

    def recover(_descriptor, _font_name):
        nonlocal recovered_called
        recovered_called = True
        return {0x37: "T"}

    monkeypatch.setattr(
        active_direct_font_backend,
        "_recover_unencoded_core_font_subset_map",
        recover,
    )

    active_direct_font_backend._build_simple_font_backend(
        {
            "Subtype": "Type1",
            "BaseFont": "Arial_Fett0350.625",
            "Encoding": "StandardEncoding",
            "FontDescriptor": {"FontName": "Arial_Fett0350.625"},
        },
        try_font_metrics=False,
    )

    assert recovered_called is False


def test_does_not_recover_an_arial_subset_with_a_tounicode_map(monkeypatch):
    recovered_called = False

    def recover(_descriptor, _font_name):
        nonlocal recovered_called
        recovered_called = True
        return {0x37: "T"}

    monkeypatch.setattr(
        active_direct_font_backend,
        "_recover_unencoded_core_font_subset_map",
        recover,
    )
    monkeypatch.setattr(
        active_direct_font_backend,
        "build_simple_unicode_map",
        lambda _spec, **_kwargs: None,
    )

    active_direct_font_backend._build_simple_font_backend(
        {
            "Subtype": "Type1",
            "BaseFont": "Arial_Fett0350.625",
            "ToUnicode": object(),
            "FontDescriptor": {"FontName": "Arial_Fett0350.625"},
        },
        try_font_metrics=False,
    )

    assert recovered_called is False


def test_does_not_recover_a_non_core_font_subset(monkeypatch):
    face_load_attempted = False

    def face_loader(_data):
        nonlocal face_load_attempted
        face_load_attempted = True
        raise AssertionError("non-core fonts must not be loaded for recovery")

    monkeypatch.setattr(
        active_direct_font_backend,
        "freetype",
        type("FreeType", (), {"Face": staticmethod(face_loader)}),
    )

    recovered = active_direct_font_backend._recover_unencoded_core_font_subset_map(
        {"FontFile": object()},
        "Helvetica_Subset",
    )

    assert face_load_attempted is False
    assert recovered == {}


def test_preserves_embedded_type1_encoding_before_recovery(monkeypatch):
    class _FontFile:
        def __getitem__(self, key):
            assert key == "Length1"
            return 1

        def get_data(self):
            return b"x"

    class _HeaderParser:
        def __init__(self, _data):
            pass

        def get_encoding(self):
            return {0x37: "7"}

    recovered_called = False

    def recover(_descriptor, _font_name):
        nonlocal recovered_called
        recovered_called = True
        return {0x37: "T"}

    monkeypatch.setattr(active_direct_font_backend, "stream_value", lambda value: value)
    monkeypatch.setattr(active_direct_font_backend, "Type1FontHeaderParser", _HeaderParser)
    monkeypatch.setattr(
        active_direct_font_backend,
        "_recover_unencoded_core_font_subset_map",
        recover,
    )

    backend = active_direct_font_backend._build_simple_font_backend(
        {
            "Subtype": "Type1",
            "BaseFont": "Arial_Fett0350.625",
            "FontDescriptor": {
                "FontName": "Arial_Fett0350.625",
                "FontFile": _FontFile(),
            },
        },
        try_font_metrics=False,
    )

    assert recovered_called is False
    assert backend.unicode_text(0x37, "(cid:55)") == "7"
