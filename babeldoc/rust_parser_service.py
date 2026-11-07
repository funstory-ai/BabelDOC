"""Parse PDFs through the Rust extension for maintainer-facing Python tools."""

from __future__ import annotations

import json
from dataclasses import fields
from dataclasses import is_dataclass
from pathlib import Path
from typing import get_args

import pymupdf
from xsdata.formats.dataclass.parsers import JsonParser

from babeldoc.format.pdf.document_il import il_version_1
from babeldoc.rust_debug_service import RustDebugSerializerService


def _page_indexes(pages: str | None, page_count: int) -> list[int] | None:
    if not pages:
        return None

    ranges: list[tuple[int, int]] = []
    for part in pages.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-")
            ranges.append((int(start) if start else 1, int(end) if end else -1))
        else:
            page = int(part)
            ranges.append((page, page))

    return [
        page_index
        for page_index in range(page_count)
        if any(
            start <= page_index + 1 and (end == -1 or page_index + 1 <= end)
            for start, end in ranges
        )
    ]


def _dataclass_type(annotation):
    if isinstance(annotation, type) and is_dataclass(annotation):
        return annotation
    for argument in get_args(annotation):
        dataclass_type = _dataclass_type(argument)
        if dataclass_type is not None:
            return dataclass_type
    return None


def _use_il_json_field_names(value, schema_type):
    """Translate Rust's serde field names to the IL schema's JSON aliases."""
    if isinstance(value, list):
        item_type = _dataclass_type(schema_type)
        return [_use_il_json_field_names(item, item_type) for item in value]
    if isinstance(value, dict):
        if schema_type is None:
            return {
                key: _use_il_json_field_names(item, None) for key, item in value.items()
            }
        schema_fields = {item.name: item for item in fields(schema_type)}
        normalized = {}
        for key, item in value.items():
            schema_field = schema_fields.get(key)
            alias = schema_field.metadata.get("name", key) if schema_field else key
            child_type = _dataclass_type(schema_field.type) if schema_field else None
            normalized[alias] = _use_il_json_field_names(item, child_type)
        return normalized
    return value


def parse_pdf_to_il(
    pdf_path: str | Path,
    *,
    pages: str | None = None,
    working_dir: str | Path | None = None,
    debug: bool = False,
) -> il_version_1.Document:
    """Return the legacy IL dataclasses from the Rust parser artifact."""
    from babeldoc import _rust_il

    pdf_path = Path(pdf_path)
    if pages:
        with pymupdf.open(pdf_path) as pdf:
            page_indexes = _page_indexes(pages, pdf.page_count)
    else:
        page_indexes = None

    rust_document = _rust_il.parse_prepared_pdf(
        str(pdf_path), json.dumps({"pages": page_indexes})
    )
    artifact_json = rust_document.artifact()
    service = RustDebugSerializerService()
    document_data = service.materialize_document(artifact_json)
    if debug and working_dir is not None:
        debug_path = Path(working_dir) / "create_il.debug.json"
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        service.write_debug_json(artifact_json, str(debug_path))
    document_data = _use_il_json_field_names(document_data, il_version_1.Document)
    return JsonParser().from_string(json.dumps(document_data), il_version_1.Document)
