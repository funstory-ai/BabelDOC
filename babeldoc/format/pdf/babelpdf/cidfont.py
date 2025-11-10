import re
from io import BytesIO

import freetype


def indirect(obj):
    if isinstance(obj, tuple) and obj[0] == "xref":
        return int(obj[1].split(" ")[0])


def get_xref(doc, xref, key):
    obj = doc.xref_get_key(xref, key)
    if obj[0] == "xref":
        return indirect(obj)


def get_font_file(doc, xref):
    if idx := get_xref(doc, xref, "FontFile"):
        return doc.xref_stream(idx)
    if idx := get_xref(doc, xref, "FontFile2"):
        return doc.xref_stream(idx)
    if idx := get_xref(doc, xref, "FontFile3"):
        return doc.xref_stream(idx)


def get_font_descriptor(doc, xref):
    if idx := get_xref(doc, xref, "FontDescriptor"):
        return get_font_file(doc, idx)


def get_descendant_fonts(doc, xref):
    obj = doc.xref_get_key(xref, "DescendantFonts")
    array_text = ""
    if obj[0] == "xref":
        array_text = doc.xref_object(indirect(obj))
    elif obj[0] == "array":
        array_text = obj[1]
    if m := re.search(r"\d+", array_text):
        return get_font_descriptor(doc, int(m.group(0)))


def get_glyph_bbox(face, g):
    try:
        face.load_glyph(g, freetype.FT_LOAD_NO_SCALE)
        cbox = face.glyph.outline.get_bbox()
        return cbox.xMin, cbox.yMin, cbox.xMax, cbox.yMax
    except Exception:
        return 0, 0, 0, 0


def get_face_bbox(blob):
    face = freetype.Face(BytesIO(blob))
    scale = 1000 / face.units_per_EM
    bbox_list = [get_glyph_bbox(face, code) for code in range(face.num_glyphs)]
    bbox_list = [[v * scale for v in bbox] for bbox in bbox_list]
    return bbox_list


def get_cidfont_bbox(doc, xref):
    if doc.xref_get_key(xref, "Subtype")[1] == "/Type0":
        if blob := get_descendant_fonts(doc, xref):
            return get_face_bbox(blob)
