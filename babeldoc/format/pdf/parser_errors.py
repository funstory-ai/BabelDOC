class UnsupportedOperatorError(ValueError):
    pass


class TokenizerError(ValueError):
    pass


class PDFException(Exception):  # noqa: N818 - preserve the parser exception name.
    pass


class PDFTypeError(PDFException, TypeError):
    pass
