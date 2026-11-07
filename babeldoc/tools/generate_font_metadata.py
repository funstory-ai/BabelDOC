# This script is used to automatically generate the following files:
# https://github.com/funstory-ai/BabelDOC-Assets/blob/main/font_metadata.json


import argparse
import hashlib
import logging
import re
import tempfile
from pathlib import Path

import orjson
import pymupdf
from babeldoc.format.pdf.document_il import PdfFont
from rich.logging import RichHandler

logger = logging.getLogger(__name__)

serif_keywords = [
    "serif",
]
sans_serif_keywords = ["sans", "GoNotoKurrent"]
serif_regex = "|".join(serif_keywords)
sans_serif_regex = "|".join(sans_serif_keywords)


def get_font_metadata(
    font_path,
    *,
    working_dir: str | Path | None = None,
) -> PdfFont:
    from babeldoc.rust_parser_service import parse_pdf_to_il

    font_path = Path(font_path)
    temp_root = Path(working_dir) if working_dir is not None else None
    if temp_root is not None:
        temp_root.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(
        dir=temp_root,
        prefix="babeldoc-font-metadata-",
    ) as temp_dir:
        temp_dir_path = Path(temp_dir)
        probe_pdf_path = temp_dir_path / "font_probe.pdf"
        parse_working_dir = temp_dir_path / "parse_working"

        _write_font_probe_pdf(font_path, probe_pdf_path)
        il = parse_pdf_to_il(probe_pdf_path, working_dir=parse_working_dir)

    il_page = il.page[0]
    if not il_page.pdf_font:
        raise RuntimeError(
            "new parser did not produce font metadata from the generated probe PDF"
        )
    return il_page.pdf_font[0]


def _write_font_probe_pdf(font_path: Path, output_path: Path) -> None:
    doc = pymupdf.open()
    try:
        page = doc.new_page(width=1000, height=1000)
        page.insert_font("test_font", fontfile=str(font_path))
        page.insert_text(
            (72, 120),
            "BabelDOC font metadata probe",
            fontname="test_font",
            fontsize=32,
        )
        doc.save(output_path)
    finally:
        doc.close()


def _stable_json_number(value: int | float) -> int | float:
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def main():
    logging.basicConfig(level=logging.INFO, handlers=[RichHandler()])
    parser = argparse.ArgumentParser(description="Get font metadata.")
    parser.add_argument("assets_repo_path", type=str, help="Path to the font file.")
    parser.add_argument(
        "--working-dir",
        help="Optional directory for temporary generated probe PDFs.",
    )
    args = parser.parse_args()
    repo_path = Path(args.assets_repo_path)
    assert repo_path.exists(), f"Assets repo path {repo_path} does not exist."
    assert (repo_path / "README.md").exists(), (
        f"Assets repo path {repo_path} does not contain a README.md file."
    )
    assert (repo_path / "fonts").exists(), (
        f"Assets repo path {repo_path} does not contain a fonts folder."
    )
    logger.info(f"Getting font metadata for {repo_path}")

    metadatas = {}
    for font_path in list((repo_path / "fonts").glob("**/*.ttf")):
        logger.info(f"Getting font metadata for {font_path}")
        with Path(font_path).open("rb") as f:
            # Read the file in chunks to handle large files efficiently
            hash_ = hashlib.sha3_256()
            while True:
                chunk = f.read(1024 * 1024)
                if not chunk:
                    break
                hash_.update(chunk)
        extracted_metadata = get_font_metadata(
            font_path,
            working_dir=args.working_dir,
        )

        if re.search(serif_regex, extracted_metadata.name, re.IGNORECASE):
            serif = 1
        else:
            serif = 0

        metadata = {
            "file_name": font_path.name,
            "font_name": extracted_metadata.name,
            "encoding_length": extracted_metadata.encoding_length,
            "bold": extracted_metadata.bold,
            "italic": extracted_metadata.italic,
            "monospace": extracted_metadata.monospace,
            "serif": serif,
            "ascent": _stable_json_number(extracted_metadata.ascent),
            "descent": _stable_json_number(extracted_metadata.descent),
            "sha3_256": hash_.hexdigest(),
            "size": font_path.stat().st_size,
        }
        metadatas[font_path.name] = metadata
    metadatas = orjson.dumps(
        metadatas,
        option=orjson.OPT_APPEND_NEWLINE | orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS,
    ).decode()
    print(f"FONT METADATA: {metadatas}")
    with (repo_path / "font_metadata.json").open("w") as f:
        f.write(metadatas)


if __name__ == "__main__":
    main()
