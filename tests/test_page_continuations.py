import json

import numpy as np
import pymupdf
import pytest
from babeldoc.assets.assets import get_font_and_metadata
from babeldoc.docvision.base_doclayout import YoloBox
from babeldoc.docvision.base_doclayout import YoloResult
from babeldoc.format.pdf.high_level import translate
from babeldoc.format.pdf.translation_config import TranslationConfig
from babeldoc.format.pdf.translation_config import WatermarkOutputMode

SOURCE = (
    "The observation confirms that the measured response depends on",
    "the temperature of the sample during the initial experiment.",
)
TARGET = (
    "Les observations montrent que la reponse de l'echantillon depend",
    "de sa temperature au debut de l'experience.",
)


def request_inputs(prompt):
    return [
        item["input"]
        for item in json.loads(prompt.split("## Here is the input:\n\n", 1)[1])
    ]


class PageLayout:
    def handle_document(self, pages, pdf, config, save_debug_image):
        for page in pages:
            blocks = pdf[page.page_number].get_text("blocks")
            yield (
                page,
                YoloResult(
                    ["text"],
                    boxes=[
                        YoloBox(xyxy=block[:4], conf=np.float32(0.99), cls=0)
                        for block in blocks
                    ],
                ),
            )


class Translator:
    def __init__(self, use_llm, source=SOURCE, target=TARGET, incomplete=False):
        self.use_llm = use_llm
        self.requests = []
        self.translations = dict(zip(source, target, strict=True))
        self.translations[" ".join(source)] = " ".join(target)
        self.incomplete = incomplete

    def do_llm_translate(self, text):
        if not self.use_llm:
            raise NotImplementedError

    def llm_translate(self, prompt, rate_limit_params=None):
        self.requests.append(prompt)
        if not rate_limit_params.get("request_json_mode"):
            text = prompt.split("Now translate the following text:\n\n", 1)[1]
            return self.translations[text]
        inputs = request_inputs(prompt)
        if self.incomplete:
            inputs = inputs[:-1]
        return json.dumps(
            [
                {"id": index, "output": self.translations[text]}
                for index, text in enumerate(inputs)
            ]
        )

    def translate(self, text, rate_limit_params=None):
        self.requests.append(text)
        return self.translations[text]

    def get_formular_placeholder(self, index):
        return f"{{v{index}}}", r"\{v" + str(index) + r"\}"

    def get_rich_text_left_placeholder(self, index):
        return f"<b{index}>", f"<b{index}>"

    def get_rich_text_right_placeholder(self, index):
        return f"</b{index}>", f"</b{index}>"


@pytest.fixture
def translate_pdf(tmp_path, monkeypatch):
    font_id = "NotoSans-Regular.ttf"
    font_path, metadata = get_font_and_metadata(font_id)
    family = {
        "normal": [font_id],
        "script": [],
        "fallback": [font_id],
        "base": [font_id],
    }
    monkeypatch.setattr(
        "babeldoc.rust_run_context._font_assets",
        lambda _config: (
            family,
            {font_id: str(font_path)},
            {font_id: metadata["sha3_256"]},
        ),
    )

    def run(translator, pages, heights=None, **kwargs):
        source = tmp_path / "source.pdf"
        with pymupdf.open() as pdf:
            for index, lines in enumerate(pages):
                page = pdf.new_page(
                    width=600, height=heights[index] if heights else 800
                )
                for x, y, text, *colors in lines:
                    page.insert_text(
                        (x, y), text, fontsize=12, color=next(iter(colors), None)
                    )
            pdf.save(source)

        result = translate(
            TranslationConfig(
                translator=translator,
                input_file=source,
                lang_in="en",
                lang_out="fr",
                doc_layout_model=PageLayout(),
                output_dir=tmp_path,
                no_dual=True,
                auto_extract_glossary=False,
                skip_scanned_detection=True,
                watermark_output_mode=WatermarkOutputMode.NoWatermark,
                **kwargs,
            )
        )
        return result.mono_pdf_path

    return run


@pytest.mark.parametrize("use_llm", [True, False])
def test_page_continuation(translate_pdf, use_llm):
    translator = Translator(use_llm)
    output = translate_pdf(
        translator,
        [[(54, 740, SOURCE[0])], [(54, 70, SOURCE[1])]],
    )
    assert len(translator.requests) == 1
    if use_llm:
        assert request_inputs(translator.requests[0]) == [" ".join(SOURCE)]
    else:
        assert translator.requests == [" ".join(SOURCE)]

    with pymupdf.open(output) as pdf:
        assert len(pdf) == 2
        assert "Les observations" in pdf[0].get_text()
        assert "l'experience." in pdf[1].get_text()
        assert pdf[0].get_text("words")[0][1] > 710
        assert pdf[1].get_text("words")[0][1] < 90
        first_spans = [
            page.get_text("dict")["blocks"][0]["lines"][0]["spans"][0] for page in pdf
        ]
        assert first_spans[0]["size"] == pytest.approx(first_spans[1]["size"])
        assert all(
            span["origin"][0] == pytest.approx(54, abs=3) for span in first_spans
        )
        assert " ".join(" ".join(page.get_text().split()) for page in pdf) == " ".join(
            TARGET
        )


def test_three_page_continuation_includes_short_fragment(translate_pdf):
    middle = (
        "the temperature of the sample during the initial\n"
        "stage of the experiment, as recorded by the sensors\n"
        "at intervals chosen to capture the changes in the\n"
        "response after the latest adjustment to the\n"
        "measurement settings for this"
    )
    source = (SOURCE[0], " ".join(middle.split()), "test.")
    target = (
        TARGET[0],
        "de sa temperature pendant l'experience, selon les mesures enregistrees "
        "par les capteurs pour suivre les effets du dernier ajustement au cours du",
        "test.",
    )
    translator = Translator(True, source, target)
    output = translate_pdf(
        translator,
        [[(54, 740, source[0])], [(54, 30, middle)], [(54, 70, source[2])]],
        heights=(800, 120, 800),
        min_text_length=7,
    )
    # The middle fragment fills a page in reading order, and the last piece
    # is shorter than the usual translation admission limit.
    with pymupdf.open(output) as pdf:
        assert len(pdf) == 3
        assert pdf[2].get_text().strip().endswith(target[-1])
        assert " ".join(" ".join(page.get_text().split()) for page in pdf) == " ".join(
            target
        )
    assert len(translator.requests) == 1
    assert request_inputs(translator.requests[0]) == [" ".join(source)]


def test_complete_paragraphs_remain_independent(translate_pdf):
    source = ("The initial experiment is complete.", "A new experiment begins now.")
    translator = Translator(True, source)
    translate_pdf(translator, [[(54, 740, source[0])], [(54, 70, source[1])]])
    assert len(translator.requests) == 1
    assert request_inputs(translator.requests[0]) == list(source)


def test_two_column_continuation(translate_pdf):
    source = (
        "The recorded signal depends on",
        "the temperature of the sample.",
        "We used a different sample.",
        "The second result is comparable.",
    )
    target = (
        "Le signal mesure depend",
        "de la temperature de l'echantillon.",
        "Nous avons utilise un autre specimen.",
        "Le second resultat est comparable.",
    )
    translator = Translator(True, source, target)
    translator.translations[" ".join(source[:2])] = " ".join(target[:2])
    output = translate_pdf(
        translator,
        [
            [(54, 400, source[2]), (315, 740, source[0])],
            [(54, 70, source[1]), (315, 400, source[3])],
        ],
    )
    joined = [" ".join(source[:2])]
    assert [request_inputs(prompt) for prompt in translator.requests].count(joined) == 1
    with pymupdf.open(output) as pdf:
        fragments = []
        for index, (x, y) in enumerate(((315, 740), (54, 70))):
            words = [
                word[4]
                for word in pdf[index].get_text("words")
                if word[0] >= x - 3 and abs(word[1] - y) < 30
            ]
            assert words
            fragments.extend(words)
        assert " ".join(fragments) == " ".join(target[:2])


def test_incomplete_response_retries_the_joined_paragraph(translate_pdf):
    translator = Translator(True, incomplete=True)
    output = translate_pdf(translator, [[(54, 740, SOURCE[0])], [(54, 70, SOURCE[1])]])
    assert len(translator.requests) == 2
    assert request_inputs(translator.requests[0]) == [" ".join(SOURCE)]
    assert translator.requests[1].split("Now translate the following text:\n\n", 1)[
        1
    ] == (" ".join(SOURCE))
    with pymupdf.open(output) as pdf:
        assert " ".join(" ".join(page.get_text().split()) for page in pdf) == " ".join(
            TARGET
        )


@pytest.mark.parametrize("use_llm", [True, False])
def test_unsplittable_translation_keeps_both_source_pages(translate_pdf, use_llm):
    translator = Translator(use_llm)
    translator.translations[" ".join(SOURCE)] = "Incomplete"
    output = translate_pdf(translator, [[(54, 740, SOURCE[0])], [(54, 70, SOURCE[1])]])
    assert len(translator.requests) == (2 if use_llm else 1)
    with pymupdf.open(output) as pdf:
        assert [" ".join(page.get_text().split()) for page in pdf] == list(SOURCE)


def test_long_continuation_stays_in_its_page_region(translate_pdf):
    target = (
        "Les observations realisees lors de l'experience montrent que, "
        "dans les conditions de mesure et avec les reglages prevus, "
        "la reponse de l'echantillon depend",
        TARGET[1],
    )
    translator = Translator(True, target=target)
    output = translate_pdf(translator, [[(54, 740, SOURCE[0])], [(54, 70, SOURCE[1])]])
    with pymupdf.open(output) as pdf:
        assert " ".join(" ".join(page.get_text().split()) for page in pdf) == " ".join(
            target
        )
        first_page_words = pdf[0].get_text("words")
        assert max(word[2] for word in first_page_words) <= 408
        assert max(word[3] for word in first_page_words) <= 745


def test_continuation_preserves_styles_on_both_pages(translate_pdf):
    source = (
        "<b1>The observation confirms that </b1><b3>red</b3><b5> depends on</b5>",
        "<b1>the temperature of the sample and </b1><b3>blue</b3>"
        "<b5> during the initial experiment.</b5>",
    )
    target = (
        "<b1>L'analyse presente de longs resultats avant de revenir sur la mesure du </b1>"
        "<b3>rouge</b3><b5> pour comparer</b5>",
        "<b1>les mesures du </b1><b3>bleu</b3><b5> sur l'echantillon.</b5>",
    )

    def styled_line(y, before, word, after, color):
        x = 54 + pymupdf.get_text_length(before, fontsize=12)
        end = x + pymupdf.get_text_length(word, fontsize=12)
        return [(54, y, before), (x, y, word, color), (end, y, after)]

    translator = Translator(True, source, target)
    output = translate_pdf(
        translator,
        [
            styled_line(
                740, "The observation confirms that ", "red", " depends on", (1, 0, 0)
            ),
            styled_line(
                70,
                "the temperature of the sample and ",
                "blue",
                " during the initial experiment.",
                (0, 0, 1),
            ),
        ],
    )
    assert len(translator.requests) == 1
    assert request_inputs(translator.requests[0]) == [" ".join(source)]
    with pymupdf.open(output) as pdf:
        for page, (word, color) in zip(
            pdf, [("rouge", 0xFF0000), ("bleu", 0x0000FF)], strict=True
        ):
            assert any(
                span["text"].strip() == word and span["color"] == color
                for block in page.get_text("dict")["blocks"]
                for line in block["lines"]
                for span in line["spans"]
            )
