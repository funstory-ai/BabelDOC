# Identify non-formula italic fonts that were incorrectly classified as formulas in BableDOC translation results (intermediate)

import json

import babeldoc.tools.italic_assistance as italic_assistance
from babeldoc.format.pdf.document_il.utils.formular_helper import is_formulas_font
from rich.console import Console
from rich.table import Table

console = Console()

json_path = italic_assistance.find_latest_il_json()

fonts = []

# Read intermediate representation
with json_path.open(encoding="utf-8") as f:
    pdf_data = json.load(f)

for page_index, page in enumerate(pdf_data["page"]):
    for paragraph_index, paragraph_content in enumerate(page["pdf_paragraph"]):
        font_debug_id = paragraph_content["debug_id"]
        if font_debug_id:
            # Create page font mapping
            page_font_map = {}
            for font in page["pdf_font"]:
                if "font_id" in font and "name" in font:
                    page_font_map[font["font_id"]] = (font["font_id"], font["name"])

            # Extract fonts from paragraph
            name_list = []
            paragraph_fonts = italic_assistance.extract_fonts_from_paragraph(
                paragraph_content, page_font_map
            )
            for _font_id, font_name in paragraph_fonts:
                name_list.append(font_name)

            font_list = []
            for each in fonts:
                font_list.append(each[1])

            for each_name in name_list:
                if each_name not in font_list:
                    fonts.append(
                        (page_index, each_name, paragraph_index, font_debug_id)
                    )

# Create table
table = Table(title="Font Recognition Results")
table.add_column("Page #", justify="center", style="cyan")
table.add_column("Paragraph #", justify="center", style="cyan")
table.add_column("DEBUG_ID", justify="center", style="cyan")
table.add_column("Font Name", style="magenta")
table.add_column("Recognition Result", justify="center")

# Output results
for each_font in fonts:
    page_index, font_name, paragraph_index, font_debug_id = each_font

    if is_formulas_font(font_name, None):
        table.add_row(
            str(page_index),
            str(paragraph_index),
            str(font_debug_id),
            font_name,
            "[bold red]Formula Font[/bold red]",
        )
    else:
        table.add_row(
            str(page_index),
            str(paragraph_index),
            str(font_debug_id),
            font_name,
            "[bold blue]Non-Formula Font[/bold blue]",
        )

# Print table
console.print(table)
