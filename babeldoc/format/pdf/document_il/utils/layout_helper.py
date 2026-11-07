import regex

from babeldoc.format.pdf.document_il.il_version_1 import PdfCharacter

HEIGHT_NOT_USFUL_CHAR_IN_CHAR = (None,)


def formular_height_ignore_char(char: PdfCharacter) -> bool:
    return (
        char.pdf_character_id is None
        or char.char_unicode in HEIGHT_NOT_USFUL_CHAR_IN_CHAR
    )


SPACE_REGEX = regex.compile(r"\s+", regex.UNICODE)
