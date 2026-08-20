import json

from babeldoc.format.pdf.document_il.midend.il_translator_llm_only import (
    ILTranslatorLLMOnly,
)


def _clean(llm_output: str) -> str:
    return ILTranslatorLLMOnly._clean_json_output(None, llm_output)


def test_clean_json_output_then_loads_accepts_embedded_c0_controls():
    """LLM JSON copied from PDF text may embed C0 controls that json.loads rejects.

    After cleaning, the payload must parse and keep the translated text so the
    translate path does not fall back and leak residual source fragments.
    """
    raw = '[{"id": 0, "output": "这是示例段落\x00\x01\x08\x0b\x0c\x1f"}]'

    parsed = json.loads(_clean(raw))

    assert parsed == [{"id": 0, "output": "这是示例段落"}]


def test_clean_json_output_preserves_json_legal_whitespace_and_wrappers():
    raw = """<json>
[
  {\r
    "id": 0,\t
    "output": "ok"
  }
]
</json>"""

    parsed = json.loads(_clean(raw))

    assert parsed == [{"id": 0, "output": "ok"}]
