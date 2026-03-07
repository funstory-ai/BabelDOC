import pytest

from babeldoc.main import create_parser
from babeldoc.main import should_create_term_extraction_translator
from babeldoc.main import validate_openai_temperature_args


def _parse(*extra_args: str):
    parser = create_parser()
    args = parser.parse_args(
        [
            "--openai",
            "--openai-api-key",
            "dummy-key",
            "--files",
            "dummy.pdf",
            *extra_args,
        ]
    )
    return parser, args


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--openai-temperature", "-0.1"),
        ("--openai-temperature", "2.1"),
        ("--openai-term-extraction-temperature", "-0.1"),
        ("--openai-term-extraction-temperature", "2.1"),
    ],
)
def test_temperature_out_of_range_raises_parser_error(flag: str, value: str):
    parser, args = _parse(flag, value)

    with pytest.raises(SystemExit):
        validate_openai_temperature_args(parser, args)


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--openai-temperature", "0.0"),
        ("--openai-temperature", "2.0"),
        ("--openai-term-extraction-temperature", "0.0"),
        ("--openai-term-extraction-temperature", "2.0"),
    ],
)
def test_temperature_range_boundaries_are_allowed(flag: str, value: str):
    parser, args = _parse(flag, value)

    validate_openai_temperature_args(parser, args)


@pytest.mark.parametrize(
    "extra_args",
    [
        ["--openai-term-extraction-temperature", "0.3"],
        ["--openai-term-extraction-reasoning", "low"],
    ],
)
def test_term_extraction_translator_created_for_temperature_or_reasoning_overrides(
    extra_args: list[str],
):
    _, args = _parse(*extra_args)

    assert should_create_term_extraction_translator(args)
