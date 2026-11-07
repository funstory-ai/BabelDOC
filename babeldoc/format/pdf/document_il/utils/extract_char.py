import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pymupdf
from rich.logging import RichHandler
from sklearn.cluster import DBSCAN

from babeldoc.const import get_process_pool
from babeldoc.format.pdf.document_il import il_version_1

logger = logging.getLogger(__name__)

# --- Algorithm Tuning Parameters ---

# --- Band Creation ---
# Minimum vertical overlap ratio for a character to be added to an existing band.
BAND_CREATION_OVERLAP_THRESHOLD = 0.5

# --- Line Clustering (within a band) ---
# Epsilon for DBSCAN, as a multiplier of the average character width/height.
LINE_CLUSTERING_EPS_MULTIPLIER = 3.5

# --- Line Splitting (for tall/wide lines) ---
# A line is considered for splitting if its height/width is > X times the max char size.
LINE_SPLIT_SIZE_RATIO_THRESHOLD = 1.5
# Epsilon for DBSCAN when splitting lines, as a multiplier of the max char size.
LINE_SPLIT_DBSCAN_EPS_MULTIPLIER = 0.5

# --- Space Insertion (in a finalized line) ---
# A space is inserted if the gap between chars is > X times the average char width.
SPACE_INSERTION_GAP_MULTIPLIER = 0.45

# --- Line Merging (across the page) ---
# --- Optimization ---
# Maximum vertical gap to search for potential merges, as a multiplier of avg char height.
MERGE_VERTICAL_GAP_MULTIPLIER = 1.5
# --- Containment Merge ---
# Intersection-over-area threshold to consider one line as contained within another.
MERGE_CONTAINMENT_IOU_THRESHOLD = 0.6
# --- Adjacency Merge ---
# Minimum vertical/horizontal overlap for adjacent lines to be considered for merging.
MERGE_ADJACENCY_OVERLAP_THRESHOLD = 0.7
# Maximum gap between adjacent lines to merge, as a multiplier of avg char size.
MERGE_ADJACENCY_GAP_MULTIPLIER = 1.5


# --- End of Parameters ---


def parse_pdf(
    pdf_path,
    page_ranges=None,
    *,
    working_dir: str | Path | None = None,
    debug: bool = False,
) -> il_version_1.Document:
    from babeldoc.rust_parser_service import parse_pdf_to_il

    return parse_pdf_to_il(
        pdf_path,
        pages=page_ranges,
        working_dir=working_dir,
        debug=debug,
    )


class Line:
    def __init__(self, chars: list[tuple[il_version_1.Box, str, bool]]):
        self.chars = chars
        self.text = "".join([c[1] for c in chars])


def _recalculate_line_text_with_spacing(line, orientation):
    if not line.chars:
        line.text = ""
        return

    if orientation == "horizontal":

        def get_main_start(c):
            return c[0].x

        def get_main_end(c):
            return c[0].x2

        def get_main_size(c):
            return c[0].x2 - c[0].x

    else:  # vertical

        def get_main_start(c):
            return c[0].y

        def get_main_end(c):
            return c[0].y2

        def get_main_size(c):
            return c[0].y2 - c[0].y

    line_text = ""
    avg_width = np.mean(
        [get_main_size(c) for c in line.chars if get_main_size(c) > 0] or [0]
    )

    if len(line.chars) > 1 and avg_width > 0:
        for i in range(len(line.chars) - 1):
            c1, c2 = line.chars[i], line.chars[i + 1]
            gap = get_main_start(c2) - get_main_end(c1)

            if gap > avg_width * SPACE_INSERTION_GAP_MULTIPLIER:
                line_text += c1[1] + " "
            else:
                line_text += c1[1]

    if line.chars:
        line_text += line.chars[-1][1]

    line.text = line_text


# [box, char_unicode, vertical]
# vertical: True if the char is vertical, False if the char is horizontal
def extract_paragraph_line(
    pdf_path,
    page_ranges=None,
    *,
    working_dir: str | Path | None = None,
    debug: bool = False,
) -> dict[int, list[tuple[il_version_1.Box, str, bool]]] | None:
    il = parse_pdf(
        pdf_path,
        page_ranges=page_ranges,
        working_dir=working_dir,
        debug=debug,
    )
    if il is None:
        return None
    line_boxes = {}
    for page in il.page:
        line_boxes[page.page_number] = convert_page_to_char_boxes(page)
    return line_boxes


def convert_page_to_char_boxes(
    page: il_version_1.Page,
) -> list[tuple[il_version_1.Box, str, bool]]:
    return [
        (char.visual_bbox.box, char.char_unicode, char.vertical)
        for char in page.pdf_character
    ]


def _cluster_by_axis(chars: list[tuple[il_version_1.Box, str, bool]], orientation: str):
    """
    A generalized function to cluster characters into lines based on main and secondary axes.
    """
    if not chars:
        return []

    # Define main and secondary axes based on orientation
    if orientation == "horizontal":

        def get_secondary_start(c):
            return c[0].y

        def get_secondary_end(c):
            return c[0].y2

        def get_main_start(c):
            return c[0].x

        def get_main_end(c):
            return c[0].x2

        def get_main_size(c):
            return c[0].x2 - c[0].x

    else:  # vertical

        def get_secondary_start(c):
            return c[0].x

        def get_secondary_end(c):
            return c[0].x2

        def get_main_start(c):
            return c[0].y

        def get_main_end(c):
            return c[0].y2

        def get_main_size(c):
            return c[0].y2 - c[0].y

    # Step 1: Group chars into bands along the secondary axis based on overlap.
    # This is an optimized version of the band clustering algorithm.
    # It avoids the O(N^2) complexity of the naive approach by making
    # assumptions based on the sorted order of characters.
    chars.sort(key=get_secondary_start)

    # Each band is a tuple: (list_of_chars, min_secondary_coord, max_secondary_coord)
    bands_data: list[tuple[list, float, float]] = []

    for char in chars:
        char_secondary_start = get_secondary_start(char)
        char_secondary_end = get_secondary_end(char)
        char_secondary_size = char_secondary_end - char_secondary_start

        best_band_index = -1
        max_overlap_ratio = (
            BAND_CREATION_OVERLAP_THRESHOLD  # Minimum overlap ratio to be considered
        )

        # Iterate backwards over bands, as recent bands are more likely to overlap.
        for i in range(len(bands_data) - 1, -1, -1):
            band_chars, band_secondary_start, band_secondary_end = bands_data[i]

            # Optimization: If the band is already far above the current char,
            # and since chars are sorted by start, no further bands will match.
            if band_secondary_end < char_secondary_start:
                break

            overlap = max(
                0,
                min(char_secondary_end, band_secondary_end)
                - max(char_secondary_start, band_secondary_start),
            )

            if char_secondary_size > 0:
                overlap_ratio = overlap / char_secondary_size
                if overlap_ratio > max_overlap_ratio:
                    max_overlap_ratio = overlap_ratio
                    best_band_index = i

        if best_band_index != -1:
            # Add char to the best matching band and update its boundaries
            band_chars, band_start, band_end = bands_data[best_band_index]
            band_chars.append(char)
            updated_band = (
                band_chars,
                min(band_start, char_secondary_start),
                max(band_end, char_secondary_end),
            )
            bands_data[best_band_index] = updated_band
            # Move the updated band to the end to maintain rough locality
            bands_data.append(bands_data.pop(best_band_index))
        else:
            # No suitable band found, create a new one
            bands_data.append(([char], char_secondary_start, char_secondary_end))

    # Extract final bands from the data structure
    bands = [b[0] for b in bands_data]

    # Step 2: For each band, cluster along the main axis using DBSCAN
    final_lines = []
    for band in bands:
        if len(band) < 1:
            continue

        main_axis_sizes = [get_main_size(c) for c in band if get_main_size(c) > 0]
        avg_main_size = np.mean(main_axis_sizes) if main_axis_sizes else 10

        # Epsilon for main-axis clustering is twice the average character size in that dimension
        eps = avg_main_size * LINE_CLUSTERING_EPS_MULTIPLIER

        centroids = np.array(
            [((c[0].x + c[0].x2) / 2, (c[0].y + c[0].y2) / 2) for c in band]
        )

        if centroids.size > 0:
            db = DBSCAN(eps=eps, min_samples=1, metric="manhattan").fit(centroids)

            line_groups = defaultdict(list)
            for i, label in enumerate(db.labels_):
                if label != -1:
                    line_groups[label].append(band[i])

            for _, line in line_groups.items():
                line.sort(key=get_main_start)
                final_lines.append(Line(line))

    # Step 3: Split lines that are too tall/wide, which likely contain multiple distinct lines from different columns
    processed_lines = []
    for line in final_lines:
        if not line.chars:
            continue

        line_secondary_start = min(get_secondary_start(c) for c in line.chars)
        line_secondary_end = max(get_secondary_end(c) for c in line.chars)
        line_secondary_size = line_secondary_end - line_secondary_start

        char_secondary_sizes = [
            get_secondary_end(c) - get_secondary_start(c)
            for c in line.chars
            if get_secondary_end(c) - get_secondary_start(c) > 0
        ]
        if not char_secondary_sizes:
            processed_lines.append(line)
            continue

        max_char_secondary_size = np.max(char_secondary_sizes)

        if (
            line_secondary_size
            > max_char_secondary_size * LINE_SPLIT_SIZE_RATIO_THRESHOLD
            and len(line.chars) > 1
        ):
            # logger.debug(
            #     f"Splitting line '{line.text}' which seems to contain multiple lines."
            # )

            # Use DBSCAN on the secondary axis centers to split the line
            centers = np.array(
                [
                    [(get_secondary_start(c) + get_secondary_end(c)) / 2]
                    for c in line.chars
                ]
            )
            db = DBSCAN(
                eps=max_char_secondary_size * LINE_SPLIT_DBSCAN_EPS_MULTIPLIER,
                min_samples=1,
            ).fit(centers)

            sub_lines = defaultdict(list)
            for i, label in enumerate(db.labels_):
                sub_lines[label].append(line.chars[i])

            for _, sub_line_chars in sub_lines.items():
                sub_line_chars.sort(key=get_main_start)
                processed_lines.append(Line(sub_line_chars))
        else:
            processed_lines.append(line)
    final_lines = processed_lines

    for line in final_lines:
        _recalculate_line_text_with_spacing(line, orientation)

    return final_lines


def _merge_lines_on_page(page_lines: list[Line]) -> list[Line]:
    """
    Merge lines on a page that are either contained within or adjacent to each other.
    This function contains both containment and adjacency merge logic.
    """
    if not page_lines:
        return []

    merged_lines = []
    lines_to_skip = set()

    for i in range(len(page_lines)):
        if i in lines_to_skip:
            continue

        line1 = page_lines[i]
        if not line1.chars:
            merged_lines.append(line1)
            continue

        bbox1 = (
            min(c[0].x for c in line1.chars),
            min(c[0].y for c in line1.chars),
            max(c[0].x2 for c in line1.chars),
            max(c[0].y2 for c in line1.chars),
        )

        # Optimization: Calculate a vertical gap threshold to prune the search space.
        # Based on the vertical adjacency merge condition.
        line1_avg_char_height = np.mean(
            [c[0].y2 - c[0].y for c in line1.chars if c[0].y2 > c[0].y] or [0]
        )
        max_v_gap = line1_avg_char_height * MERGE_VERTICAL_GAP_MULTIPLIER

        merged = False
        for j in range(i + 1, len(page_lines)):
            if j in lines_to_skip:
                continue

            line2 = page_lines[j]
            if not line2.chars:
                continue

            bbox2 = (
                min(c[0].x for c in line2.chars),
                min(c[0].y for c in line2.chars),
                max(c[0].x2 for c in line2.chars),
                max(c[0].y2 for c in line2.chars),
            )

            # Optimization: if line2 is too far below line1, no more merges with line1 are possible.
            # The list is sorted top-to-bottom, so we can break early.
            v_gap = bbox1[1] - bbox2[3]  # y_min_1 - y_max_2
            if v_gap > max_v_gap:
                break

            # Check for "mostly contained" by checking intersection over area
            inter_x0 = max(bbox1[0], bbox2[0])
            inter_y0 = max(bbox1[1], bbox2[1])
            inter_x1 = min(bbox1[2], bbox2[2])
            inter_y1 = min(bbox1[3], bbox2[3])

            inter_area = max(0, inter_x1 - inter_x0) * max(0, inter_y1 - inter_y0)

            area1 = (
                (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
                if (bbox1[2] > bbox1[0] and bbox1[3] > bbox1[1])
                else 0
            )
            area2 = (
                (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
                if (bbox2[2] > bbox2[0] and bbox2[3] > bbox2[1])
                else 0
            )

            # Heuristic for merging:
            # 1. By containment: if one line is mostly inside another.
            # 2. By adjacency: if two lines are close and aligned.
            if (
                area2 > 0
                and area1 >= area2
                and (inter_area / area2) > MERGE_CONTAINMENT_IOU_THRESHOLD
            ):
                # Case 1: Merge line2 (smaller) into line1 (larger) by containment
                # logger.debug(
                #     f"Merging line '{line2.text}' into '{line1.text}' (mostly contained)"
                # )
                line1.chars.extend(line2.chars)
                lines_to_skip.add(j)
                merged = True
                bbox1 = (
                    min(bbox1[0], bbox2[0]),
                    min(bbox1[1], bbox2[1]),
                    max(bbox1[2], bbox2[2]),
                    max(bbox1[3], bbox2[3]),
                )

            elif (
                area1 > 0
                and area2 > area1
                and (inter_area / area1) > MERGE_CONTAINMENT_IOU_THRESHOLD
            ):
                # Case 2: Merge line1 (smaller) into line2 (larger) by containment
                # logger.debug(
                #     f"Merging line '{line1.text}' into '{line2.text}' (mostly contained)"
                # )
                line2.chars.extend(line1.chars)
                page_lines[i], page_lines[j] = page_lines[j], page_lines[i]
                line1 = page_lines[i]
                lines_to_skip.add(j)
                merged = True
                bbox1 = (
                    min(bbox1[0], bbox2[0]),
                    min(bbox1[1], bbox2[1]),
                    max(bbox1[2], bbox2[2]),
                    max(bbox1[3], bbox2[3]),
                )

            else:
                # Case 3: Merge by adjacency for lines that are close to each other
                orientation = "horizontal" if not line1.chars[0][2] else "vertical"
                if orientation == "horizontal":
                    height1 = bbox1[3] - bbox1[1]
                    height2 = bbox2[3] - bbox2[1]
                    if height1 > 0 and height2 > 0:
                        v_overlap = max(
                            0,
                            min(bbox1[3], bbox2[3]) - max(bbox1[1], bbox2[1]),
                        )
                        if (
                            v_overlap / height1
                        ) > MERGE_ADJACENCY_OVERLAP_THRESHOLD and (
                            v_overlap / height2
                        ) > MERGE_ADJACENCY_OVERLAP_THRESHOLD:
                            h_gap = max(bbox1[0], bbox2[0]) - min(bbox1[2], bbox2[2])
                            if h_gap >= 0:
                                avg_char_width = np.mean(
                                    [
                                        c[0].x2 - c[0].x
                                        for c in (line1.chars + line2.chars)
                                        if c[0].x2 > c[0].x
                                    ]
                                    or [0]
                                )
                                if (
                                    avg_char_width > 0
                                    and h_gap
                                    < avg_char_width * MERGE_ADJACENCY_GAP_MULTIPLIER
                                ):
                                    # logger.debug(
                                    #     f"Merging adjacent lines '{line1.text}' and '{line2.text}'"
                                    # )
                                    line1.chars.extend(line2.chars)
                                    lines_to_skip.add(j)
                                    merged = True
                                    bbox1 = (
                                        min(bbox1[0], bbox2[0]),
                                        min(bbox1[1], bbox2[1]),
                                        max(bbox1[2], bbox2[2]),
                                        max(bbox1[3], bbox2[3]),
                                    )
                else:  # Vertical
                    width1 = bbox1[2] - bbox1[0]
                    width2 = bbox2[2] - bbox2[0]
                    if width1 > 0 and width2 > 0:
                        h_overlap = max(
                            0,
                            min(bbox1[2], bbox2[2]) - max(bbox1[0], bbox2[0]),
                        )
                        if (
                            h_overlap / width1
                        ) > MERGE_ADJACENCY_OVERLAP_THRESHOLD and (
                            h_overlap / width2
                        ) > MERGE_ADJACENCY_OVERLAP_THRESHOLD:
                            v_gap = max(bbox1[1], bbox2[1]) - min(bbox1[3], bbox2[3])
                            if v_gap >= 0:
                                avg_char_height = np.mean(
                                    [
                                        c[0].y2 - c[0].y
                                        for c in (line1.chars + line2.chars)
                                        if c[0].y2 > c[0].y
                                    ]
                                    or [0]
                                )
                                if (
                                    avg_char_height > 0
                                    and v_gap
                                    < avg_char_height * MERGE_ADJACENCY_GAP_MULTIPLIER
                                ):
                                    # logger.debug(
                                    #     f"Merging adjacent vertical lines '{line1.text}' and '{line2.text}'"
                                    # )
                                    line1.chars.extend(line2.chars)
                                    lines_to_skip.add(j)
                                    merged = True
                                    bbox1 = (
                                        min(bbox1[0], bbox2[0]),
                                        min(bbox1[1], bbox2[1]),
                                        max(bbox1[2], bbox2[2]),
                                        max(bbox1[3], bbox2[3]),
                                    )

        if merged:
            # Re-sort and recalculate text for the merged line
            orientation = (
                "horizontal" if not line1.chars[0][2] else "vertical"
            )  # Guess orientation from first char
            if orientation == "horizontal":
                line1.chars.sort(key=lambda c: c[0].x)
            else:  # vertical
                line1.chars.sort(key=lambda c: c[0].y)
            _recalculate_line_text_with_spacing(line1, orientation)

        merged_lines.append(line1)

    return merged_lines


def process_page_chars_to_lines(
    chars: list[tuple[il_version_1.Box, str, bool]],
) -> list[Line]:
    pool = get_process_pool()
    if pool is None:
        return process_page_chars_to_lines_internal(chars)
    return pool.apply(process_page_chars_to_lines_internal, (chars,))


def process_page_chars_to_lines_internal(
    chars: list[tuple[il_version_1.Box, str, bool]],
) -> list[Line]:
    """
    Process characters on a single page to cluster them into lines.

    Args:
        chars: List of character tuples (box, char_unicode, is_vertical)

    Returns:
        List of Line objects representing clustered and merged lines
    """
    if not chars:
        return []

    horizontal_chars = [c for c in chars if not c[2]]
    vertical_chars = [c for c in chars if c[2]]

    horizontal_lines = _cluster_by_axis(horizontal_chars, "horizontal")
    vertical_lines = _cluster_by_axis(vertical_chars, "vertical")

    page_lines = horizontal_lines + vertical_lines

    # Sort all found lines by their position on the page (top-to-bottom, left-to-right)
    def get_line_position(line):
        if not line:
            return (0, 0)
        # PDF coordinate system: Y increases upwards. We negate it for top-to-bottom sort.
        avg_y = np.mean([(c[0].y + c[0].y2) / 2 for c in line])
        avg_x = np.mean([(c[0].x + c[0].x2) / 2 for c in line])
        return (-avg_y, avg_x)

    page_lines.sort(key=lambda line: get_line_position(line.chars))

    # Merge lines on the page
    merged_page_lines = _merge_lines_on_page(page_lines)
    return merged_page_lines


def cluster_chars_to_lines(
    char_boxes: dict[int, list[tuple[il_version_1.Box, str, bool]]],
) -> dict[int, list[Line]]:
    clustered_lines = {}
    if not char_boxes:
        return clustered_lines

    for page_num, chars in char_boxes.items():
        merged_page_lines = process_page_chars_to_lines(chars)
        clustered_lines[page_num] = merged_page_lines

    return clustered_lines


def draw_clustered_lines_to_image(
    pdf_path,
    clustered_lines: dict[int, list[Line]],
    *,
    output_root: str | Path = "ocr-box-image-clustered",
) -> Path:
    doc = pymupdf.open(pdf_path)
    debug_dir = Path(output_root) / Path(pdf_path).stem
    debug_dir.mkdir(parents=True, exist_ok=True)

    for page_number, lines in clustered_lines.items():
        if not lines:
            continue

        page = doc[page_number]
        pixmap = page.get_pixmap(dpi=300)
        image_height = pixmap.height
        image_width = pixmap.width

        samples = bytearray(pixmap.samples)
        image_array = np.frombuffer(samples, dtype=np.uint8).reshape(
            image_height, image_width, pixmap.n
        )

        if pixmap.n in [3, 4]:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

        # cv2.imwrite(str(debug_dir / f"{page_number}.png"), image_array)

        annotated_image = image_array.copy()

        page_rect = page.rect
        x_scale = image_width / page_rect.width
        y_scale = image_height / page_rect.height

        for i, line in enumerate(lines):
            if not line:
                continue

            # Draw the encompassing line box first (red)
            char_boxes_in_line = [item[0] for item in line.chars]
            min_x = min(b.x for b in char_boxes_in_line)
            min_y = min(b.y for b in char_boxes_in_line)
            max_x2 = max(b.x2 for b in char_boxes_in_line)
            max_y2 = max(b.y2 for b in char_boxes_in_line)

            img_x0_line = int(min_x * x_scale)
            img_y1_line = int(image_height - (max_y2 * y_scale))
            img_x1_line = int(max_x2 * x_scale)
            img_y0_line = int(image_height - (min_y * y_scale))

            cv2.rectangle(
                annotated_image,
                (img_x0_line, img_y1_line),
                (img_x1_line, img_y0_line),
                (0, 0, 255),  # Red for lines
                2,
            )

            cv2.putText(
                annotated_image,
                f"line {i}: {line.text}",
                (img_x0_line, img_y1_line - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

            # Then, draw the individual character boxes on top (green)
            for char_box, _, _ in line.chars:
                pdf_x0, pdf_y0, pdf_x1, pdf_y1 = (
                    char_box.x,
                    char_box.y,
                    char_box.x2,
                    char_box.y2,
                )

                img_x0_char = int(pdf_x0 * x_scale)
                img_y0_char_pdf = int(pdf_y0 * y_scale)
                img_x1_char = int(pdf_x1 * x_scale)
                img_y1_char_pdf = int(pdf_y1 * y_scale)

                img_y0_char = image_height - img_y0_char_pdf
                img_y1_char = image_height - img_y1_char_pdf

                cv2.rectangle(
                    annotated_image,
                    (img_x0_char, img_y1_char),
                    (img_x1_char, img_y0_char),
                    (0, 255, 0),  # Green for characters
                    1,  # Thinner line
                )

        cv2.imwrite(str(debug_dir / f"{page_number}_annotated.png"), annotated_image)

    doc.close()
    return debug_dir


def main(argv: list[str] | None = None):
    logging.basicConfig(level=logging.INFO, handlers=[RichHandler()])
    parser = argparse.ArgumentParser(
        description="Agent-only debug helper: extract native-parser character boxes and clustered line images."
    )
    parser.add_argument("pdf", nargs="+", help="PDF path(s) to inspect.")
    parser.add_argument("--pages", help="Optional BabelDOC page range, e.g. 1-3.")
    parser.add_argument(
        "--working-dir",
        help="Optional working directory for parser temporary files.",
    )
    parser.add_argument(
        "--output-dir",
        default="ocr-box-image-clustered",
        help="Directory for annotated image artifacts.",
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--no-images",
        action="store_true",
        help="Skip annotated image generation and only emit JSON summary.",
    )
    args = parser.parse_args(argv)

    summaries = []
    for pdf_path in args.pdf:
        logger.info(f"Processing {pdf_path}")
        char_boxes = extract_paragraph_line(
            pdf_path,
            page_ranges=args.pages,
            working_dir=args.working_dir,
            debug=args.debug,
        )
        if not char_boxes:
            logger.warning(f"No character boxes extracted from {pdf_path}")
            summaries.append(
                {
                    "char_count": 0,
                    "line_count": 0,
                    "pdf": pdf_path,
                    "status": "no_chars",
                }
            )
            continue

        char_count = sum(len(chars) for chars in char_boxes.values())
        logger.info(f"Extracted {char_count} characters. Clustering them into lines...")
        lines = cluster_chars_to_lines(char_boxes)

        total_lines = sum(len(l) for l in lines.values())
        logger.info(f"Clustered into {total_lines} lines. Drawing boxes...")

        # logger.info("--- Clustered Lines Text ---")
        # for page_num, page_lines in lines.items():
        #     logger.info(f"Page {page_num}:")
        #     for i, line in enumerate(page_lines):
        #         logger.info(f"  Line {i}: {line.text}")
        # logger.info("----------------------------")

        output_dir = None
        if not args.no_images:
            output_dir = draw_clustered_lines_to_image(
                pdf_path,
                lines,
                output_root=args.output_dir,
            )
            logger.info(f"Annotated images saved in {output_dir}.")
        summaries.append(
            {
                "char_count": char_count,
                "line_count": total_lines,
                "output_dir": str(output_dir) if output_dir is not None else None,
                "page_count": len(char_boxes),
                "pages": args.pages,
                "pdf": pdf_path,
                "status": "ok",
            }
        )

    print(json.dumps({"results": summaries}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
