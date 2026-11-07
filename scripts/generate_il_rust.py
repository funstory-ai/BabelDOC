#!/usr/bin/env python3
"""Generate the Rust IL and direct PyO3 field readers from the Python schema."""

from __future__ import annotations

import argparse
import dataclasses
import pathlib
import sys
import typing

# Runtime-only forms observed in 0.6.4 parser output. Keep this explicit and
# evidence-linked: the Python schema remains untouched.
RUNTIME_TYPE_OVERRIDES = {
    "PdfCurve.debug_info": {
        "rust": "Option<crate::CurveDebugInfo>",
        "evidence": (
            "babeldoc/format/pdf/document_il/frontend/il_creater_active.py:1257",
            "babeldoc/format/pdf/document_il/frontend/il_creater_active.py:1285",
            "runtime-type evidence:45",
        ),
        "reason": "the active parser emits both a string marker and a bool",
    },
    "PdfFont.bold": {
        "rust": "Option<crate::RuntimeBool>",
        "evidence": (
            "babeldoc/format/pdf/document_il/utils/fontmap.py:300",
            "runtime-type evidence:413",
        ),
        "reason": "PyMuPDF font flags are runtime integers 0/1",
    },
    "PdfFont.italic": {
        "rust": "Option<crate::RuntimeBool>",
        "evidence": (
            "babeldoc/format/pdf/document_il/utils/fontmap.py:301",
            "runtime-type evidence:453",
        ),
        "reason": "PyMuPDF font flags are runtime integers 0/1",
    },
    "PdfFont.monospace": {
        "rust": "Option<crate::RuntimeBool>",
        "evidence": (
            "babeldoc/format/pdf/document_il/utils/fontmap.py:302",
            "runtime-type evidence:493",
        ),
        "reason": "PyMuPDF font flags are runtime integers 0/1",
    },
    "PdfFont.serif": {
        "rust": "Option<crate::RuntimeBool>",
        "evidence": (
            "babeldoc/format/pdf/document_il/utils/fontmap.py:303",
            "runtime-type evidence:530",
        ),
        "reason": "PyMuPDF font flags are runtime integers 0/1",
    },
    "GraphicState.passthrough_per_char_instruction": {
        "rust": "Option<crate::PassthroughValue>",
        "evidence": (
            "babeldoc/format/pdf/document_il/frontend/il_creater_active.py:1008",
            "babeldoc/format/pdf/document_il/frontend/il_creater_active_support.py:53",
            "runtime-type evidence:82",
        ),
        "reason": "the field is an eager string or a lazy persistent-log view",
    },
}

SHARED_TYPES = {"GraphicState", "PdfFont", "PdfCurve", "PdfForm"}


def rust_type(annotation: object, *, owner: str, field_name: str) -> str:
    override = RUNTIME_TYPE_OVERRIDES.get(f"{owner}.{field_name}")
    if override:
        return override["rust"]
    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)
    if origin in (typing.Union, __import__("types").UnionType):
        non_none = [arg for arg in args if arg is not type(None)]
        if len(non_none) == 1 and len(non_none) != len(args):
            inner = rust_type(non_none[0], owner=owner, field_name=field_name)
            is_dataclass = isinstance(non_none[0], type) and dataclasses.is_dataclass(
                non_none[0]
            )
            if field_name == "graphic_state" and inner == "GraphicState":
                return "Option<crate::Shared<GraphicState>>"
            return (
                f"Option<std::boxed::Box<{inner}>>"
                if is_dataclass
                else f"Option<{inner}>"
            )
        raise TypeError(f"unsupported union: {annotation!r}")
    if origin is list:
        if len(args) != 1:
            raise TypeError(f"unsupported list: {annotation!r}")
        item = rust_type(args[0], owner=owner, field_name=field_name)
        if item in {"PdfFont", "PdfCurve", "PdfForm"}:
            return f"Vec<crate::Shared<{item}>>"
        return f"Vec<{item}>"
    if annotation is str:
        return "String"
    if annotation is int:
        return "i64"
    if annotation is float:
        return "f64"
    if annotation is bool:
        return "bool"
    if annotation is object:
        return "crate::PythonValue"
    if isinstance(annotation, type) and dataclasses.is_dataclass(annotation):
        return annotation.__name__
    raise TypeError(f"unsupported annotation for {owner}.{field_name}: {annotation!r}")


def generate(schema: pathlib.Path, mode: str) -> str:
    repo_root = schema.parents[4]
    sys.path.insert(0, str(repo_root))
    from babeldoc.format.pdf.document_il import il_version_1

    classes = [
        value
        for value in vars(il_version_1).values()
        if isinstance(value, type)
        and dataclasses.is_dataclass(value)
        and value.__module__ == il_version_1.__name__
    ]
    if mode in {"m-core", "m-reader"}:
        from il_rust_m import generate_core
        from il_rust_m import generate_reader

        return (generate_core if mode == "m-core" else generate_reader)(
            classes, rust_type
        )
    # Owning Document served only the retired stages/production transfer.
    # Shared helpers still require Page and its nested value types.
    if mode == "types":
        classes = [cls for cls in classes if cls.__name__ != "Document"]
    lines = [
        "// Generated from babeldoc/format/pdf/document_il/il_version_1.py; do not edit.",
        "",
    ]
    if mode in {"types", "combined"}:
        lines.insert(
            1,
            "use serde::{Deserialize, Serialize};"
            if mode == "combined"
            else "use serde::Serialize;",
        )
    if mode in {"combined"}:
        lines.insert(2, "use pyo3::types::PyAnyMethods;")
    if mode in {"types", "combined"}:
        for cls in classes:
            hints = typing.get_type_hints(cls)
            lines.extend(
                [
                    "#[derive(Clone, Debug, Default, Deserialize, Serialize)]"
                    if mode == "combined"
                    else "#[derive(Clone, Debug, Default, Serialize)]",
                    "#[serde(default)]",
                    f"pub struct {cls.__name__} {{",
                ]
            )
            for field in dataclasses.fields(cls):
                override = RUNTIME_TYPE_OVERRIDES.get(f"{cls.__name__}.{field.name}")
                if override:
                    lines.append(
                        "    // Runtime override: "
                        + override["reason"]
                        + "; evidence: "
                        + ", ".join(override["evidence"])
                    )
                typ = rust_type(
                    hints[field.name], owner=cls.__name__, field_name=field.name
                )
                # The independent historical F1 measurement DTO retains its
                # original JSON carrier; production M uses PythonValue.
                if mode == "combined":
                    typ = typ.replace("crate::PythonValue", "serde_json::Value")
                rust_name = "r#" + field.name if field.name in {"box"} else field.name
                lines.append(f"    pub {rust_name}: {typ},")
            lines.extend(["}", ""])

    if mode in {"types", "combined"}:
        field_count = sum(len(dataclasses.fields(cls)) for cls in classes)
        lines.append(f"pub const IL_CLASS_COUNT: usize = {len(classes)};")
        lines.append(f"pub const IL_FIELD_COUNT: usize = {field_count};")
        lines.append(
            f"pub const IL_RUNTIME_OVERRIDE_COUNT: usize = {len(RUNTIME_TYPE_OVERRIDES)};"
        )
    if mode in {"combined"}:
        for cls in classes:
            hints = typing.get_type_hints(cls)
            lines.extend(
                [
                    f"impl crate::FromPyValue for {cls.__name__} {{",
                    "    fn from_py(value: &pyo3::Bound<'_, pyo3::PyAny>, context: &mut crate::PyBuildContext) -> pyo3::PyResult<Self> {",
                    "        Ok(Self {",
                ]
            )
            for field in dataclasses.fields(cls):
                typ = rust_type(
                    hints[field.name], owner=cls.__name__, field_name=field.name
                )
                # The independent historical F1 measurement DTO retains its
                # original JSON carrier; production M uses PythonValue.
                if mode == "combined":
                    typ = typ.replace("crate::PythonValue", "serde_json::Value")
                rust_name = "r#" + field.name if field.name in {"box"} else field.name
                lines.append(
                    f'            {rust_name}: <{typ} as crate::FromPyValue>::from_py(&value.getattr("{field.name}")?, context)?,'
                )
            lines.extend(["        })", "    }", "}", ""])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--schema", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("types", "combined", "m-core", "m-reader"),
        required=True,
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(generate(args.schema, args.mode), encoding="utf-8")


if __name__ == "__main__":
    main()
