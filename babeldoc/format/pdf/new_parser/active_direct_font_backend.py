from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from io import BytesIO

import freetype

from babeldoc.format.pdf.new_parser.active_value_access import dict_value
from babeldoc.format.pdf.new_parser.active_value_access import int_value
from babeldoc.format.pdf.new_parser.active_value_access import list_value
from babeldoc.format.pdf.new_parser.active_value_access import literal_name
from babeldoc.format.pdf.new_parser.active_value_access import num_value
from babeldoc.format.pdf.new_parser.active_value_access import stream_value
from babeldoc.format.pdf.new_parser.font_spec_primitives import classify_font_subtype
from babeldoc.format.pdf.new_parser.font_types import PdfRuntimeFontLike
from babeldoc.format.pdf.new_parser.object_model import ActiveLiteral
from babeldoc.format.pdf.new_parser.runtime.cid_cmap_runtime import build_cid_cmap
from babeldoc.format.pdf.new_parser.runtime.cid_cmap_runtime import (
    build_cid_unicode_map,
)
from babeldoc.format.pdf.new_parser.runtime.font_data_runtime import FontMetricsDB
from babeldoc.format.pdf.new_parser.runtime.font_data_runtime import TrueTypeFont
from babeldoc.format.pdf.new_parser.runtime.font_data_runtime import (
    Type1FontHeaderParser,
)
from babeldoc.format.pdf.new_parser.runtime.font_encoding_runtime import (
    STANDARD_ENCODING_NAME,
)
from babeldoc.format.pdf.new_parser.runtime.font_encoding_runtime import EncodingDB
from babeldoc.format.pdf.new_parser.runtime.font_encoding_runtime import name2unicode
from babeldoc.format.pdf.new_parser.runtime.font_unicode_maps import (
    build_simple_unicode_map,
)
from babeldoc.format.pdf.new_parser.runtime.font_widths import get_widths
from babeldoc.format.pdf.new_parser.runtime.font_widths import get_widths2
from babeldoc.format.pdf.new_parser.runtime.object_primitives_runtime import resolve1
from babeldoc.format.pdf.new_parser.state import apply_matrix_norm


def recover_unencoded_subset_map(font_face: object) -> dict[int, str]:
    """Recover character codes from ``G<glyph-id>`` names in subset fonts.

    Some Type1/CFF subset fonts omit both ``/Encoding`` and ``/ToUnicode``.
    Their embedded program can still expose a custom cmap whose glyph names
    retain the original glyph IDs. This is the convention used by legacy
    Microsoft core fonts, where glyph ID 55, for example, represents ``T``.
    """
    try:
        charmaps = getattr(font_face, "charmaps", ()) or (None,)
        for charmap in charmaps:
            recovered: dict[int, str] = {}
            if charmap is not None:
                font_face.set_charmap(charmap)
            char_code, glyph_index = font_face.get_first_char()
            while glyph_index:
                glyph_name = font_face.get_glyph_name(glyph_index).decode("ascii")
                match = re.fullmatch(r"G(\d+)", glyph_name)
                if match:
                    original_glyph_id = int(match.group(1))
                    if 3 <= original_glyph_id <= 97:
                        recovered[char_code] = chr(original_glyph_id + 29)
                char_code, glyph_index = font_face.get_next_char(
                    char_code,
                    glyph_index,
                )
            if recovered:
                return recovered
    except Exception:
        return {}
    return {}


def _recover_unencoded_core_font_subset_map(
    descriptor: Mapping[object, object],
    font_name: str | bytes,
) -> dict[int, str]:
    """Recover an omitted encoding from an embedded Arial/Times subset font."""
    if not re.search(r"Arial|Times[_ ]?New[_ ]?Roman", str(font_name)):
        return {}

    for file_key in ("FontFile", "FontFile2", "FontFile3"):
        font_file = descriptor.get(file_key)
        if font_file is None:
            continue
        try:
            font_face = freetype.Face(BytesIO(stream_value(font_file).get_data()))
            recovered = recover_unencoded_subset_map(font_face)
            if recovered:
                return recovered
        except Exception:
            continue
    return {}


def _normalize_font_name_like_pdfminer(value: object) -> str | bytes:
    if isinstance(value, ActiveLiteral):
        value = value.name
    if hasattr(value, "name"):
        return literal_name(value)
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError:
            return str(value)
    if isinstance(value, str):
        try:
            return value.encode("latin-1").decode("utf-8")
        except UnicodeDecodeError:
            return str(value.encode("latin-1"))
    return str(value)


@dataclass(slots=True)
class ActiveSimpleRuntimeFontBackend:
    fontname: str | bytes
    widths: dict[int | str, float]
    cid2unicode: dict[int, str]
    unicode_map: object | None
    ascent: float
    descent: float
    default_width: float = 0

    def decode(self, data: bytes) -> object:
        return bytearray(data)

    def unicode_text(self, cid: int, fallback_text: str) -> str:
        if self.unicode_map is not None:
            get_unichr = getattr(self.unicode_map, "get_unichr", None)
            if callable(get_unichr):
                try:
                    return get_unichr(cid)
                except KeyError:
                    pass
        return self.cid2unicode.get(cid, fallback_text)

    def is_multibyte(self) -> bool:
        return False

    def is_vertical(self) -> bool:
        return False

    def char_width(self, cid: int) -> float:
        cid_width = self.widths.get(cid)
        if cid_width is not None:
            return float(cid_width) * 0.001
        try:
            unicode_text = self.unicode_text(cid, f"(cid:{cid})")
        except Exception:
            unicode_text = None
        if unicode_text is not None:
            unicode_width = self.widths.get(unicode_text)
            if unicode_width is not None:
                return float(unicode_width) * 0.001
        return float(self.default_width) * 0.001

    def char_disp(self, cid: int) -> float | tuple[float | None, float]:
        _ = cid
        return 0

    def get_descent(self) -> float:
        return self.descent * 0.001

    def runtime_identity(self) -> int:
        return id(self)

    def compute_encoding_length(self, *, mupdf: object, xref_id: int) -> int:
        _ = (mupdf, xref_id)
        return 1


@dataclass(slots=True)
class ActiveType3RuntimeFontBackend:
    fontname: str | bytes
    widths: dict[int | str, float]
    cid2unicode: dict[int, str]
    unicode_map: object | None
    ascent: float
    descent: float
    hscale: float
    vscale: float
    default_width: float = 0

    def decode(self, data: bytes) -> object:
        return bytearray(data)

    def unicode_text(self, cid: int, fallback_text: str) -> str:
        if self.unicode_map is not None:
            get_unichr = getattr(self.unicode_map, "get_unichr", None)
            if callable(get_unichr):
                try:
                    return get_unichr(cid)
                except KeyError:
                    pass
        return self.cid2unicode.get(cid, fallback_text)

    def is_multibyte(self) -> bool:
        return False

    def is_vertical(self) -> bool:
        return False

    def char_width(self, cid: int) -> float:
        cid_width = self.widths.get(cid)
        if cid_width is not None:
            return float(cid_width) * self.hscale
        try:
            unicode_text = self.unicode_text(cid, f"(cid:{cid})")
        except Exception:
            unicode_text = None
        if unicode_text is not None:
            unicode_width = self.widths.get(unicode_text)
            if unicode_width is not None:
                return float(unicode_width) * self.hscale
        return float(self.default_width) * self.hscale

    def char_disp(self, cid: int) -> float | tuple[float | None, float]:
        _ = cid
        return 0

    def get_descent(self) -> float:
        return self.descent * self.vscale

    def runtime_identity(self) -> int:
        return id(self)

    def compute_encoding_length(self, *, mupdf: object, xref_id: int) -> int:
        _ = (mupdf, xref_id)
        return 1


@dataclass(slots=True)
class ActiveCIDRuntimeFontBackend:
    fontname: str | bytes
    cmap: object
    unicode_map: object | None
    widths: dict[int | str, float]
    default_width: float
    disps: dict[int, tuple[float | None, float]]
    default_disp: float | tuple[float | None, float]
    vertical: bool
    cidcoding: str
    ascent: float
    descent: float
    encoding_length_hint: int = 1
    has_encoding: bool = False
    cid_encoding: object | None = None

    def decode(self, data: bytes) -> object:
        if self.has_encoding and self.cid_encoding is not None:
            try:
                decoded = self.cid_encoding.decode(data)
                if decoded is not None and all(x > 0 for x in decoded):
                    return decoded
            except Exception:
                pass
        return self.cmap.decode(data)

    def unicode_text(self, cid: int, fallback_text: str) -> str:
        if self.unicode_map is not None:
            get_unichr = getattr(self.unicode_map, "get_unichr", None)
            if callable(get_unichr):
                try:
                    return get_unichr(cid)
                except KeyError:
                    pass
        return fallback_text

    def to_unichr(self, cid: int) -> str:
        text = self.unicode_text(cid, f"(cid:{cid})")
        if text == f"(cid:{cid})":
            raise KeyError(cid)
        return text

    def is_multibyte(self) -> bool:
        return True

    def is_vertical(self) -> bool:
        return self.vertical

    def char_width(self, cid: int) -> float:
        cid_width = self.widths.get(cid)
        if cid_width is not None:
            return float(cid_width) * 0.001
        try:
            unicode_text = self.unicode_text(cid, f"(cid:{cid})")
        except Exception:
            unicode_text = None
        if unicode_text is not None:
            unicode_width = self.widths.get(unicode_text)
            if unicode_width is not None:
                return float(unicode_width) * 0.001
        return float(self.default_width) * 0.001

    def char_disp(self, cid: int) -> float | tuple[float | None, float]:
        return self.disps.get(cid, self.default_disp)

    def get_descent(self) -> float:
        return self.descent * 0.001

    def runtime_identity(self) -> int:
        return id(self)

    def compute_encoding_length(self, *, mupdf: object, xref_id: int) -> int:
        encoding_length = self.encoding_length_hint
        try:
            _, encoding = mupdf.xref_get_key(xref_id, "Encoding")
            if encoding == "/Identity-H" or encoding == "/Identity-V":
                encoding_length = 2
            elif encoding == "/WinAnsiEncoding":
                encoding_length = 1
            else:
                _, to_unicode_id = mupdf.xref_get_key(xref_id, "ToUnicode")
                if to_unicode_id is not None:
                    to_unicode_bytes = mupdf.xref_stream(
                        int(to_unicode_id.split(" ")[0])
                    )
                    code_range = (
                        __import__("re")
                        .search(
                            b"begincodespacerange\n?.*<(\\d+?)>.*",
                            to_unicode_bytes,
                        )
                        .group(1)
                    )
                    encoding_length = len(code_range) // 2
        except Exception:
            cid_map = getattr(self.unicode_map, "cid2unichr", None)
            if cid_map and max(cid_map.keys()) > 255:
                encoding_length = 2
        return encoding_length


def construct_active_direct_runtime_font(
    runtime_spec: dict[object, object],
) -> PdfRuntimeFontLike | None:
    return _construct_active_direct_pdfminer_font(runtime_spec)


def _construct_active_direct_pdfminer_font(runtime_spec: dict[object, object]):
    subtype = classify_font_subtype(runtime_spec)
    if subtype in ("Type1", "MMType1"):
        return _construct_type1_font(runtime_spec)
    if subtype == "TrueType":
        return _construct_truetype_font(runtime_spec)
    if subtype == "Type3":
        return _construct_type3_font(runtime_spec)
    if subtype in ("CIDFontType0", "CIDFontType2"):
        return _construct_cid_font(runtime_spec)
    if subtype == "Type0":
        return _construct_type0_font(runtime_spec)
    return None


def _construct_type1_font(runtime_spec: dict[object, object]):
    return _build_simple_font_backend(
        runtime_spec,
        try_font_metrics=True,
    )


def _construct_truetype_font(runtime_spec: dict[object, object]):
    return _build_simple_font_backend(
        runtime_spec,
        try_font_metrics=True,
    )


def _construct_type3_font(runtime_spec: dict[object, object]):
    return _build_type3_font_backend(runtime_spec)


def _construct_cid_font(runtime_spec: dict[object, object]):
    return _build_cid_font_backend(runtime_spec)


def _construct_type0_font(runtime_spec: dict[object, object]):
    descendant_fonts = list_value(runtime_spec["DescendantFonts"])
    subspec = dict_value(descendant_fonts[0]).copy()
    for key in ("Encoding", "ToUnicode"):
        if key in runtime_spec:
            subspec[key] = runtime_spec[key]
    descendant_font = _construct_active_direct_pdfminer_font(subspec)
    if descendant_font is not None:
        return descendant_font

    # Legacy/pdfminer falls back to a Type1-shaped default when subtype is
    # missing on the descendant spec instead of forcing CID semantics.
    fallback_spec = dict(subspec)
    fallback_spec.setdefault("Subtype", "Type1")
    return _construct_type1_font(fallback_spec)


def _build_simple_font_backend(
    runtime_spec: Mapping[object, object], *, try_font_metrics: bool
) -> ActiveSimpleRuntimeFontBackend:
    spec = dict(runtime_spec)
    descriptor = dict_value(spec.get("FontDescriptor", {}))
    basefont = _normalize_font_name_like_pdfminer(
        descriptor.get("FontName", spec.get("BaseFont", "unknown"))
    )

    widths: dict[int | str, float]
    if try_font_metrics:
        try:
            metrics_descriptor, metrics_widths = FontMetricsDB.get_metrics(basefont)
            descriptor = dict(metrics_descriptor)
            widths = dict(metrics_widths)
            basefont = _normalize_font_name_like_pdfminer(
                descriptor.get("FontName", basefont)
            )
        except KeyError:
            widths = _build_spec_widths(spec)
    else:
        widths = _build_spec_widths(spec)

    cid2unicode = _build_simple_encoding_map(
        spec,
        encoding_db=EncodingDB,
        literal_name=literal_name,
    )
    unicode_map = build_simple_unicode_map(
        spec,
        stream_value=stream_value,
    )

    has_embedded_type1_encoding = False
    if "Encoding" not in spec and "FontFile" in descriptor:
        try:
            fontfile = stream_value(descriptor.get("FontFile"))
            length1 = int_value(fontfile["Length1"])
            data = fontfile.get_data()[:length1]
            offset = 0
            if b"/Encoding" in data:
                offset = data.index(b"/Encoding")
            parser = Type1FontHeaderParser(BytesIO(data[offset:]))
            embedded_encoding = parser.get_encoding()
            if embedded_encoding:
                cid2unicode = embedded_encoding
                has_embedded_type1_encoding = True
        except Exception:
            pass

    if (
        "Encoding" not in spec
        and "ToUnicode" not in spec
        and not has_embedded_type1_encoding
    ):
        recovered = _recover_unencoded_core_font_subset_map(descriptor, basefont)
        if recovered:
            cid2unicode = recovered

    descent = num_value(descriptor.get("Descent", 0))
    ascent = num_value(descriptor.get("Ascent", 0))
    if descent > 0:
        descent = -descent
    default_width = num_value(descriptor.get("MissingWidth", 0))

    return ActiveSimpleRuntimeFontBackend(
        fontname=basefont,
        widths=widths,
        cid2unicode=cid2unicode,
        unicode_map=unicode_map,
        ascent=ascent,
        descent=descent,
        default_width=default_width,
    )


def _build_type3_font_backend(
    runtime_spec: Mapping[object, object],
) -> ActiveType3RuntimeFontBackend:
    spec = dict(runtime_spec)
    firstchar = int_value(spec.get("FirstChar", 0))
    width_list = list_value(spec.get("Widths", [0] * 256))
    widths: dict[int | str, float] = {
        i + firstchar: width for (i, width) in enumerate(width_list)
    }

    if "FontDescriptor" in spec:
        descriptor = dict_value(spec["FontDescriptor"])
    else:
        descriptor = {"Ascent": 0, "Descent": 0, "FontBBox": spec["FontBBox"]}

    basefont = _normalize_font_name_like_pdfminer(
        descriptor.get("FontName", spec.get("BaseFont", "unknown"))
    )

    cid2unicode = _build_simple_encoding_map(
        spec,
        encoding_db=EncodingDB,
        literal_name=literal_name,
    )
    unicode_map = build_simple_unicode_map(
        spec,
        stream_value=stream_value,
    )

    matrix = tuple(list_value(spec.get("FontMatrix")))
    # Match pdfminer's Type3 semantics: once a FontDescriptor exists, do not
    # silently fall back to the top-level FontBBox. Malformed/missing descriptor
    # FontBBox should degrade to a zero bbox instead of leaking spec-level
    # bounds into ascent/descent.
    bbox_source = (
        descriptor.get("FontBBox", (0, 0, 0, 0))
        if "FontDescriptor" in spec
        else spec.get("FontBBox", (0, 0, 0, 0))
    )
    try:
        bbox = tuple(list_value(bbox_source))[:4]
    except Exception:
        bbox = (0.0, 0.0, 0.0, 0.0)
    if len(bbox) < 4:
        bbox = (0.0, 0.0, 0.0, 0.0)
    _, descent, _, ascent = bbox
    hscale, vscale = apply_matrix_norm(matrix, (1, 1))
    default_width = num_value(descriptor.get("MissingWidth", 0))

    return ActiveType3RuntimeFontBackend(
        fontname=basefont,
        widths=widths,
        cid2unicode=cid2unicode,
        unicode_map=unicode_map,
        ascent=ascent,
        descent=descent,
        hscale=hscale,
        vscale=vscale,
        default_width=default_width,
    )


def _build_cid_font_backend(
    runtime_spec: Mapping[object, object],
) -> ActiveCIDRuntimeFontBackend:
    from babeldoc.format.pdf.babelpdf.cmap import CharacterMap

    spec = dict(runtime_spec)
    descriptor = dict_value(spec.get("FontDescriptor", {}))
    try:
        basefont = _normalize_font_name_like_pdfminer(
            descriptor.get("FontName", spec.get("BaseFont", b"unknown"))
        )
    except Exception:
        basefont = "unknown"

    cidsysteminfo = dict_value(spec.get("CIDSystemInfo", {}))
    cid_registry_value = resolve1(cidsysteminfo.get("Registry", b"unknown"))
    cid_ordering_value = resolve1(cidsysteminfo.get("Ordering", b"unknown"))
    cid_registry = (
        cid_registry_value.decode("latin1")
        if isinstance(cid_registry_value, bytes)
        else str(cid_registry_value)
    )
    cid_ordering = (
        cid_ordering_value.decode("latin1")
        if isinstance(cid_ordering_value, bytes)
        else str(cid_ordering_value)
    )
    cidcoding = f"{cid_registry.strip()}-{cid_ordering.strip()}"

    cmap_name, cmap = build_cid_cmap(
        spec,
        literal_name=literal_name,
    )

    ttf = None
    has_encoding = False
    cid_encoding = None
    try:
        if "Encoding" in spec:
            encoding_part = spec["Encoding"]
            if hasattr(encoding_part, "get_data"):
                has_encoding = True
                cid_encoding = CharacterMap(encoding_part.get_data().decode("U8"))
    except Exception:
        has_encoding = False
        cid_encoding = None

    if "FontFile2" in descriptor:
        try:
            fontfile = stream_value(descriptor.get("FontFile2"))
            ttf = TrueTypeFont(basefont, BytesIO(fontfile.get_data()))
        except Exception:
            ttf = None

    unicode_map = build_cid_unicode_map(
        spec,
        cid_ordering=cidcoding,
        ttf=ttf,
        cmap=cmap,
        stream_value=stream_value,
        literal_name=literal_name,
    )

    vertical = cmap.is_vertical()
    if vertical:
        widths2 = get_widths2(list_value(spec.get("W2", [])))
        disps = {cid: (vx, vy) for (cid, (_, (vx, vy))) in widths2.items()}
        vy, width_default = spec.get("DW2", [880, -1000])
        default_disp = (None, vy)
        widths = {cid: width for (cid, (width, _)) in widths2.items()}
        default_width = width_default
    else:
        disps = {}
        default_disp = 0
        widths = get_widths(list_value(spec.get("W", [])))
        default_width = spec.get("DW", 1000)

    encoding_length_hint = 1
    if cmap_name in ("Identity-H", "Identity-V"):
        encoding_length_hint = 2
    elif unicode_map is not None:
        cid_map = getattr(unicode_map, "cid2unichr", None)
        if cid_map and max(cid_map.keys()) > 255:
            encoding_length_hint = 2

    ascent = 0.0
    descent = 0.0
    if descriptor:
        try:
            ascent = num_value(descriptor.get("Ascent", 0))
            descent = num_value(descriptor.get("Descent", 0))
            if descent > 0:
                descent = -descent
        except Exception:
            pass

    return ActiveCIDRuntimeFontBackend(
        fontname=basefont,
        cmap=cmap,
        unicode_map=unicode_map,
        widths=widths,
        default_width=default_width,
        disps=disps,
        default_disp=default_disp,
        vertical=vertical,
        cidcoding=cidcoding,
        ascent=ascent,
        descent=descent,
        encoding_length_hint=encoding_length_hint,
        has_encoding=has_encoding,
        cid_encoding=cid_encoding,
    )


def _build_spec_widths(spec: Mapping[object, object]) -> dict[int | str, float]:
    firstchar = int_value(spec.get("FirstChar", 0))
    width_list = list_value(spec.get("Widths", [0] * 256))
    return {i + firstchar: w for (i, w) in enumerate(width_list)}


def _build_simple_encoding_map(
    spec: Mapping[object, object], *, encoding_db, literal_name
) -> dict[int, str]:
    if "Encoding" in spec:
        encoding = spec["Encoding"]
    else:
        encoding = STANDARD_ENCODING_NAME
    if isinstance(encoding, dict):
        name = literal_name(encoding.get("BaseEncoding", STANDARD_ENCODING_NAME))
        cid2unicode = encoding_db.get_encoding(name).copy()
        diff = list_value(encoding.get("Differences", []))
        if diff:
            cid = 0
            for item in diff:
                if isinstance(item, int):
                    cid = item
                    continue
                try:
                    cid2unicode[cid] = name2unicode(literal_name(item))
                except (KeyError, ValueError):
                    pass
                cid += 1
        return cid2unicode
    return encoding_db.get_encoding(literal_name(encoding))
