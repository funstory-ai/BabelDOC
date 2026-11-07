"""Materialize Rust debug snapshots as the legacy JSON tree at stage boundaries."""

from __future__ import annotations

import json
import struct


def _decode_python_values(value):
    if isinstance(value, list):
        return [_decode_python_values(item) for item in value]
    if not isinstance(value, dict):
        return value
    tag = value.get("$py")
    payload = value.get("value")
    if tag in {"tuple", "set", "frozenset", "list"}:
        return [_decode_python_values(item) for item in payload]
    if tag == "dict":
        return {
            _decode_python_values(key): _decode_python_values(item)
            for key, item in payload
        }
    if tag == "int":
        return int(payload, 0)
    if tag == "float":
        return struct.unpack(">d", bytes.fromhex(value["bits"]))[0]
    if tag == "unicode":
        return "".join(chr(codepoint) for codepoint in payload)
    if tag in {"bytes", "bytearray"}:
        return bytes(payload).decode("utf-8")
    return {key: _decode_python_values(item) for key, item in value.items()}


def _restore_legacy_fallback_layout_confidence(value):
    """Match the integer confidence emitted by Python's fallback layout stage."""
    if isinstance(value, list):
        return [_restore_legacy_fallback_layout_confidence(item) for item in value]
    if not isinstance(value, dict):
        return value
    result = {
        key: _restore_legacy_fallback_layout_confidence(item)
        for key, item in value.items()
    }
    if (
        result.get("class_name") == "fallback_line"
        and result.get("conf") == 1.0
        and "box" in result
        and "id" in result
    ):
        result["conf"] = 1
    return result


def _materialize_instruction_views(document, node_records):
    from babeldoc.format.pdf.document_il.frontend.il_creater_active_support import (
        CtmAwarePassthroughArg,
    )
    from babeldoc.format.pdf.document_il.frontend.il_creater_active_support import (
        LazyPassthroughInstruction,
    )
    from babeldoc.format.pdf.document_il.frontend.il_creater_active_support import (
        PassthroughSnapshot,
    )
    from babeldoc.format.pdf.document_il.frontend.il_creater_active_support import (
        _PassthroughNode,
    )

    nodes = []
    for record in node_records:
        arg = _decode_python_values(record["arg"])
        if isinstance(arg, dict):
            arg = CtmAwarePassthroughArg(arg["value"], tuple(arg["ctm"]))
        parent_index = record.get("parent_ordinal")
        nodes.append(
            _PassthroughNode(
                None if parent_index is None else nodes[parent_index],
                (record["op"], arg),
                record.get("replace_key"),
                record["version"],
                record["length"],
            )
        )

    def visit(value):
        if isinstance(value, list):
            return [visit(item) for item in value]
        if not isinstance(value, dict):
            return value
        if value.get("kind") == "view":
            tail_ref = value.get("tail_ref")
            tail = None if tail_ref is None else nodes[tail_ref]
            snapshot = PassthroughSnapshot(
                tail,
                tuple(tuple(item) for item in value["latest_versions"]),
                max((node["version"] for node in node_records), default=0) + 1,
            )
            return LazyPassthroughInstruction(
                snapshot,
                include_clipping=value["include_clipping"],
                suffix_parts=value["suffix_parts"],
            ).materialize()
        return {key: visit(item) for key, item in value.items()}

    return visit(document)


class RustDebugSerializerService:
    cleanup_methods = frozenset()

    def materialize_document(self, artifact_json: str):
        artifact = json.loads(artifact_json)
        document = _materialize_instruction_views(
            artifact["document"], artifact.get("graphic_state_nodes", [])
        )
        document = _decode_python_values(document)
        return _restore_legacy_fallback_layout_confidence(document)

    def write_debug_json(self, artifact_json: str, path: str):
        from babeldoc.format.pdf.document_il.xml_converter import XMLConverter

        document = self.materialize_document(artifact_json)
        XMLConverter().write_json(document, path)
