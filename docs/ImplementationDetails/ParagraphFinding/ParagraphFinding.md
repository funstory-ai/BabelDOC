# Paragraph Finding

> [!NOTE]
> This documentation may contain AI-generated content. While we strive for accuracy, there might be inaccuracies. Please report any issues via:
>
> - [GitHub Issues](https://github.com/funstory-ai/yadt/issues)
> - Community contribution (PRs welcome!)

## Background

After PDF analysis, we need to identify paragraphs from individual characters. This is a crucial step before translation and typesetting, as it helps maintain the logical structure of the document.

## Goal

1. Group characters into meaningful paragraphs while preserving the document's logical structure
2. Handle special cases like table of contents, short lines, and multi-line paragraphs
3. Maintain layout information for later typesetting

## Specific Implementation

The paragraph finding process consists of four main steps:

### Step 1: Create Initial Paragraphs

1. Group characters into lines based on their spatial relationships
2. Create paragraphs based on layout information and XObject IDs
3. Characters that don't belong to text layouts are skipped

### Step 2: Process Paragraph Spacing

1. Remove completely empty lines
2. Handle trailing spaces within lines
3. Update paragraph boundary boxes and metadata

### Step 3: Calculate Line Width Statistics

1. Calculate the median width of all lines
2. This information is used for identifying potential paragraph breaks

### Step 4: Process Independent Paragraphs

1. Analyze paragraphs with multiple lines
2. Split paragraphs in two cases:
   - When encountering table of contents entries: dot leaders require at least 10 dots if the run contains internal horizontal whitespace (spaces, tabs, or non-breaking spaces), or 20 dots if all dots are contiguous. Whitespace before or after the run does not lower the threshold, and line separators cannot join two runs.
   - When finding lines significantly shorter than the median width (configurable via `short_line_split_factor`)
3. Keep dot-leader paragraphs separate when merging body text around alternating line numbers, so independent directory page numbers do not cause entries to be joined again.

## Additional Features

1. Layout-aware processing:
   - Respects different layout types (plain text, title, figure caption, etc.)
   - Maintains layout priority order for overlapping regions

2. First line indent detection:
   - Automatically detects and marks paragraphs with first line indentation

3. Flexible character position detection:
   - Uses multiple position detection modes (middle, topleft, bottomright)
   - Special handling for characters with unreliable height information

## Limitations

1. The current implementation assumes left-to-right text direction

2. May not perfectly handle complex layouts with overlapping regions

3. Table of contents detection relies on dot leaders meeting the 10-dot spaced or 20-dot contiguous threshold; shorter leaders and entries without dot leaders are not recognized by this rule.

4. Short line splitting might occasionally create incorrect paragraph breaks

## Configuration Options

The paragraph finding behavior can be customized through `TranslationConfig`:

1. `split_short_lines`: Enable/disable splitting paragraphs at short lines
2. `short_line_split_factor`: Threshold factor for short line detection (relative to median width) 