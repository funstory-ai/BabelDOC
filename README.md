<!-- # Yet Another Document Translator -->

<div align="center">
<!-- <img src="https://s.immersivetranslate.com/assets/r2-uploads/images/babeldoc-banner.png" width="320px"  alt="YADT"/> -->

<br/>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://s.immersivetranslate.com/assets/uploads/babeldoc-big-logo-darkmode-with-transparent-background-IKuNO1.svg" width="320px" alt="BabelDOC"/>
  <img src="https://s.immersivetranslate.com/assets/uploads/babeldoc-big-logo-with-transparent-background-2xweBr.svg" width="320px" alt="BabelDOC"/>
</picture>

<!-- <h2 id="title">BabelDOC</h2> -->

<p>
  <!-- PyPI -->
  <a href="https://pypi.org/project/BabelDOC/">
    <img src="https://img.shields.io/pypi/v/BabelDOC"></a>
  <a href="https://pepy.tech/projects/BabelDOC">
    <img src="https://static.pepy.tech/badge/BabelDOC"></a>
  <!-- <a href="https://github.com/funstory-ai/BabelDOC/pulls">
    <img src="https://img.shields.io/badge/contributions-welcome-green"></a> -->
  <!-- License -->
  <a href="./LICENSE">
    <img src="https://img.shields.io/github/license/funstory-ai/BabelDOC"></a>
  <a href="https://t.me/+Z9_SgnxmsmA5NzBl">
    <img src="https://img.shields.io/badge/Telegram-2CA5E0?style=flat-squeare&logo=telegram&logoColor=white"></a>
  <a href="https://deepwiki.com/funstory-ai/BabelDOC"><img src="https://deepwiki.com/badge.svg" alt="Ask DeepWiki"></a>
</p>

<a href="https://trendshift.io/repositories/13358" target="_blank"><img src="https://trendshift.io/api/badge/repositories/13358" alt="funstory-ai%2FBabelDOC | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>

</div>

PDF scientific paper translation and bilingual comparison library.

- **Online Service**: Beta version launched [Immersive Translate - BabelDOC](https://app.immersivetranslate.com/babel-doc/) Free usage quota is available; please refer to the FAQ section on the page for details.
- **Self-deployment**: [PDFMathTranslate-next](https://github.com/PDFMathTranslate-next/PDFMathTranslate-next) support for BabelDOC, available for self-deployment + WebUI with more translation services.
- Provides a simple [command line interface](#getting-started).
- Provides a [Python API](#python-api).
- Mainly designed to be embedded into other programs, but can also be used directly for simple translation tasks.

> [!TIP]
>
> How to use BabelDOC in Zotero
>
> 1. Immersive Translate Pro members can use the [immersive-translate/zotero-immersivetranslate](https://github.com/immersive-translate/zotero-immersivetranslate) plugin
>
> 2. PDFMathTranslate self-deployed users can use the [guaguastandup/zotero-pdf2zh](https://github.com/guaguastandup/zotero-pdf2zh) plugin

[Supported Language](https://funstory-ai.github.io/BabelDOC/supported_languages/)

## Preview

<div align="center">
<img src="https://s.immersivetranslate.com/assets/r2-uploads/images/babeldoc-preview.png" width="80%"/>
</div>

## We are hiring

See details: [EN](https://github.com/funstory-ai/jobs) | [ZH](https://github.com/funstory-ai/jobs/blob/main/README_ZH.md)

## Getting Started

### Install from PyPI

We recommend using the Tool feature of [uv](https://github.com/astral-sh/uv) to install yadt.

1. First, you need to refer to [uv installation](https://github.com/astral-sh/uv#installation) to install uv and set up the `PATH` environment variable as prompted.

2. Use the following command to install yadt:

```bash
uv tool install --python 3.12 BabelDOC

babeldoc --help
```

3. Use the `babeldoc` command. For example:

```bash
babeldoc --openai --openai-model "gpt-4o-mini" --openai-base-url "https://api.openai.com/v1" --openai-api-key "your-api-key-here"  --files example.pdf

# multiple files
babeldoc --openai --openai-model "gpt-4o-mini" --openai-base-url "https://api.openai.com/v1" --openai-api-key "your-api-key-here"  --files example1.pdf --files example2.pdf
```

### Install from Source

We still recommend using [uv](https://github.com/astral-sh/uv) to manage virtual environments.

1. First, you need to refer to [uv installation](https://github.com/astral-sh/uv#installation) to install uv and set up the `PATH` environment variable as prompted.

2. Use the following command to install yadt:

```bash
# clone the project
git clone https://github.com/funstory-ai/BabelDOC

# enter the project directory
cd BabelDOC

# install dependencies and run babeldoc
uv run babeldoc --help
```

3. Use the `uv run babeldoc` command. For example:

```bash
uv run babeldoc --files example.pdf --openai --openai-model "gpt-4o-mini" --openai-base-url "https://api.openai.com/v1" --openai-api-key "your-api-key-here"

# multiple files
uv run babeldoc --files example.pdf --files example2.pdf --openai --openai-model "gpt-4o-mini" --openai-base-url "https://api.openai.com/v1" --openai-api-key "your-api-key-here"
```

> [!TIP]
> The absolute path is recommended.

## Advanced Options

> [!NOTE]
> This CLI is mainly for debugging purposes. Although end users can use this CLI to translate files, we do not provide any technical support for this purpose.
>
> End users should directly use **Online Service**: Beta version launched [Immersive Translate - BabelDOC](https://app.immersivetranslate.com/babel-doc/) 1000 free pages per month.
>
> End users who need self-deployment should use [PDFMathTranslate 2.0](https://github.com/PDFMathTranslate/PDFMathTranslate-next)
> 
> If you find that an option is not listed below, it means that this option is a debugging option for maintainers. Please do not use these options.


### Language Options

- `--lang-in`, `-li`: Source language code (default: en)
- `--lang-out`, `-lo`: Target language code (default: zh)

> [!TIP]
> Currently, this project mainly focuses on English-to-Chinese translation, and other scenarios have not been tested yet.
> 
> (2025.3.1 update): Basic English target language support has been added, primarily to minimize line breaks within words([0-9A-Za-z]+).
> 
> [HELP WANTED: Collecting word regular expressions for more languages](https://github.com/funstory-ai/BabelDOC/issues/129)

### PDF Processing Options

- `--files`: One or more file paths to input PDF documents.
- `--pages`, `-p`: Specify pages to translate (e.g., "1,2,1-,-3,3-5"). If not set, translate all pages
- `--split-short-lines`: Force split short lines into different paragraphs (may cause poor typesetting & bugs)
- `--short-line-split-factor`: Split threshold factor (default: 0.8). The actual threshold is the median length of all lines on the current page \* this factor
- `--skip-clean`: Skip PDF cleaning step
- `--dual-translate-first`: Put translated pages first in dual PDF mode (default: original pages first)
- `--disable-rich-text-translate`: Disable rich text translation (may help improve compatibility with some PDFs)
- `--enhance-compatibility`: Enable all compatibility enhancement options (equivalent to --skip-clean --dual-translate-first --disable-rich-text-translate)
- `--use-alternating-pages-dual`: Use alternating pages mode for dual PDF. When enabled, original and translated pages are arranged in alternate order. When disabled (default), original and translated pages are shown side by side on the same page.
- `--watermark-output-mode`: Control watermark output mode: 'watermarked' (default) adds watermark to translated PDF, 'no_watermark' doesn't add watermark, 'both' outputs both versions.
- `--max-pages-per-part`: Maximum number of pages per part for split translation. If not set, no splitting will be performed.
- `--no-watermark`: [DEPRECATED] Use --watermark-output-mode=no_watermark instead.
- `--translate-table-text`: Translate table text (experimental, default: False)
- `--formular-font-pattern`: Font pattern to identify formula text (default: None)
- `--formular-char-pattern`: Character pattern to identify formula text (default: None)
- `--show-char-box`: Show character bounding boxes (debug only, default: False)
- `--skip-scanned-detection`: Skip scanned document detection (default: False). When using split translation, only the first part performs detection if not skipped.
- `--ocr-workaround`: Use OCR workaround (default: False). Only suitable for documents with black text on white background. When enabled, white rectangular blocks will be added below the translation to cover the original text content, and all text will be forced to black color.
- `--auto-enable-ocr-workaround`: Enable automatic OCR workaround (default: False). If a document is detected as heavily scanned, this will attempt to enable OCR processing and skip further scan detection. See "Important Interaction Note" below for crucial details on how this interacts with `--ocr-workaround` and `--skip-scanned-detection`.
- `--primary-font-family`: Override primary font family for translated text. Choices: 'serif' for serif fonts, 'sans-serif' for sans-serif fonts, 'script' for script/italic fonts. If not specified, uses automatic font selection based on original text properties.
- `--only-include-translated-page`: Only include translated pages in the output PDF. This option is only effective when `--pages` is used. (default: False)
- `--merge-alternating-line-numbers`: Enable post-processing to merge alternating line-number layouts (keep the number paragraph as an independent paragraph b; merge adjacent text paragraphs a and c across it when `layout_id` and `xobj_id` match, digits are ASCII and spaces only). Default: off.
- `--skip-form-render`: Skip form rendering (default: False). When enabled, PDF forms will not be rendered in the output.
- `--skip-curve-render`: Skip curve rendering (default: False). When enabled, PDF curves will not be rendered in the output.
- `--only-parse-generate-pdf`: Only parse PDF and generate output PDF without translation (default: False). This skips all translation-related processing including layout analysis, paragraph finding, style processing, and translation itself. Useful for testing PDF parsing and reconstruction functionality.
- `--remove-non-formula-lines`: Remove non-formula lines from paragraph areas (default: False). This removes decorative lines that are not part of formulas, while protecting lines in figure/table areas. Useful for cleaning up documents with decorative elements that interfere with text flow.
- `--non-formula-line-iou-threshold`: IoU threshold for detecting paragraph overlap when removing non-formula lines (default: 0.9). Higher values are more conservative and will remove fewer lines.
- `--figure-table-protection-threshold`: IoU threshold for protecting lines in figure/table areas when removing non-formula lines (default: 0.9). Higher values provide more protection for structural elements in figures and tables.

- `--rpc-doclayout`: RPC service host address for document layout analysis (default: None)
- `--working-dir`: Working directory for translation. If not set, use temp directory.
- `--no-auto-extract-glossary`: Disable automatic term extraction. If this flag is present, the step is skipped. Defaults to enabled.
- `--save-auto-extracted-glossary`: Save automatically extracted glossary to the specified file. If not set, the glossary will not be saved.

> [!TIP]
> - Both `--skip-clean` and `--dual-translate-first` may help improve compatibility with some PDF readers
> - `--disable-rich-text-translate` can also help with compatibility by simplifying translation input
> - However, using `--skip-clean` will result in larger file sizes
> - If you encounter any compatibility issues, try using `--enhance-compatibility` first
> - Use `--max-pages-per-part` for large documents to split them into smaller parts for translation and automatically merge them back.
> - Use `--skip-scanned-detection` to speed up processing when you know your document is not a scanned PDF.
> - Use `--ocr-workaround` to fill background for scanned PDF. (Current assumption: background is pure white, text is pure black, this option will also auto enable `--skip-scanned-detection`)

### Translation Service Options

- `--qps`: QPS (Queries Per Second) limit for translation service (default: 4)
- `--ignore-cache`: Ignore translation cache and force retranslation
- `--no-dual`: Do not output bilingual PDF files
- `--no-mono`: Do not output monolingual PDF files
- `--min-text-length`: Minimum text length to translate (default: 5)
- `--openai`: Use OpenAI for translation (default: False)
- `--custom-system-prompt`: Custom system prompt for translation.
- `--add-formula-placehold-hint`: Add formula placeholder hint for translation. (Currently not recommended, it may affect translation quality, default: False)
- `--disable-same-text-fallback`: Disable fallback translation when LLM output matches input text. (default: False)
- `--pool-max-workers`: Maximum number of worker threads for internal task processing pools. If not specified, defaults to QPS value. This parameter directly sets the worker count, replacing previous QPS-based dynamic calculations.
- `--no-auto-extract-glossary`: Disable automatic term extraction. If this flag is present, the step is skipped. Defaults to enabled.

> [!TIP]
>
> 1. Currently, only OpenAI-compatible LLM is supported. For more translator support, please use [PDFMathTranslate 2.0](https://github.com/PDFMathTranslate/PDFMathTranslate-next).
> 2. It is recommended to use models with strong compatibility with OpenAI, such as: `glm-4-flash`, `deepseek-chat`, etc.
> 3. Currently, it has not been optimized for traditional translation engines like Bing/Google, it is recommended to use LLMs.
> 4. You can use [litellm](https://github.com/BerriAI/litellm) to access multiple models.
> 5. `--custom-system-prompt`: It is mainly used to add the `/no_think` instruction of Qwen 3 in the prompt. For example: `--custom-system-prompt "/no_think You are a professional, authentic machine translation engine."`

### OpenAI Specific Options

- `--openai-model`: OpenAI model to use (default: gpt-4o-mini)
- `--openai-base-url`: Base URL for OpenAI API
- `--openai-api-key`: API key for OpenAI service
- `--enable-json-mode-if-requested`: Enable JSON mode for OpenAI requests (default: False)
- `--term-pool-max-workers`: Maximum number of worker threads dedicated to automatic term extraction. If not specified, this defaults to the value of `--pool-max-workers`, which itself defaults to the QPS value when unset.

> [!TIP]
>
> 1. This tool supports any OpenAI-compatible API endpoints. Just set the correct base URL and API key. (e.g. `https://xxx.custom.xxx/v1`)
> 2. For local models like Ollama, you can use any value as the API key (e.g. `--openai-api-key a`).

### Glossary Options

- `--glossary-files`: Comma-separated paths to glossary CSV files.
  - Each CSV file should have the columns: `source`, `target`, and an optional `tgt_lng`.
  - The `source` column contains the term in the original language.
  - The `target` column contains the term in the target language.
  - The `tgt_lng` column (optional) specifies the target language for that specific entry (e.g., "zh-CN", "en-US").
    - If `tgt_lng` is provided for an entry, that entry will only be loaded and used if its (normalized) `tgt_lng` matches the (normalized) overall target language specified by `--lang-out`. Normalization involves lowercasing and replacing hyphens (`-`) with underscores (`_`).
    - If `tgt_lng` is omitted for an entry, that entry is considered applicable for any `--lang-out`.
  - The name of each glossary (used in LLM prompts) is derived from its filename (without the .csv extension).
  - During translation, the system will check the input text against the loaded glossaries. If terms from a glossary are found in the current text segment, that glossary (with the relevant terms) will be included in the prompt to the language model, along with an instruction to adhere to it.

### Output Control

- `--output`, `-o`: Output directory for translated files. If not set, use current working directory.
- `--debug`: Enable debug logging level and export detailed intermediate results in `~/.cache/yadt/working`.
- `--report-interval`: Progress report interval in seconds (default: 0.1).

### General Options

- `--warmup`: Only download and verify required assets then exit (default: False)

### Offline Assets Management

- `--generate-offline-assets`: Generate an offline assets package in the specified directory. This creates a zip file containing all required models and fonts.
- `--restore-offline-assets`: Restore an offline assets package from the specified file. This extracts models and fonts from a previously generated package.

> [!TIP]
> 
> 1. Offline assets packages are useful for environments without internet access or to speed up installation on multiple machines.
> 2. Generate a package once with `babeldoc --generate-offline-assets /path/to/output/dir` and then distribute it.
> 3. Restore the package on target machines with `babeldoc --restore-offline-assets /path/to/offline_assets_*.zip`.
> 4. The offline assets package name cannot be modified because the file list hash is encoded in the name.
> 5. If you provide a directory path to `--restore-offline-assets`, the tool will automatically look for the correct offline assets package file in that directory.
> 6. The package contains all necessary fonts and models required for document processing, ensuring consistent results across different environments.
> 7. The integrity of all assets is verified using SHA3-256 hashes during both packaging and restoration.
> 8. If you're deploying in an air-gapped environment, make sure to generate the package on a machine with internet access first.

### Configuration File

- `--config`, `-c`: Configuration file path. Use the TOML format.

Example Configuration:

```toml
[babeldoc]
# Basic settings
debug = true
lang-in = "en-US"
lang-out = "zh-CN"
qps = 10
output = "/path/to/output/dir"

# PDF processing options
split-short-lines = false
short-line-split-factor = 0.8
skip-clean = false
dual-translate-first = false
disable-rich-text-translate = false
use-alternating-pages-dual = false
watermark-output-mode = "watermarked"  # Choices: "watermarked", "no_watermark", "both"
max-pages-per-part = 50  # Automatically split the document for translation and merge it back.
only_include_translated_page = false # Only include translated pages in the output PDF. Effective only when `pages` is used.
# no-watermark = false  # DEPRECATED: Use watermark-output-mode instead
skip-scanned-detection = false  # Skip scanned document detection for faster processing
auto_extract_glossary = true # Set to false to disable automatic term extraction
formular_font_pattern = "" # Font pattern for formula text
formular_char_pattern = "" # Character pattern for formula text
show_char_box = false # Show character bounding boxes (debug)
ocr_workaround = false # Use OCR workaround for scanned PDFs
rpc_doclayout = "" # RPC service host for document layout analysis
working_dir = "" # Working directory for translation
auto_enable_ocr_workaround = false # Enable automatic OCR workaround for scanned PDFs. See docs for interaction with ocr_workaround and skip_scanned_detection.
skip_form_render = false # Skip form rendering (default: False)
skip_curve_render = false # Skip curve rendering (default: False)
only_parse_generate_pdf = false # Only parse PDF and generate output PDF without translation (default: False)
remove_non_formula_lines = false # Remove non-formula lines from paragraph areas (default: False)
non_formula_line_iou_threshold = 0.2 # IoU threshold for paragraph overlap detection (default: 0.2)
figure_table_protection_threshold = 0.3 # IoU threshold for figure/table protection (default: 0.3)

# Translation service
openai = true
openai-model = "gpt-4o-mini"
openai-base-url = "https://api.openai.com/v1"
openai-api-key = "your-api-key-here"
enable-json-mode-if-requested = false  # Enable JSON mode when requested (default: false)
disable_same_text_fallback = false # Disable fallback translation when LLM output matches input text (default: false)
pool-max-workers = 8  # Maximum worker threads for task processing (defaults to QPS value if not set)

# Glossary Options (Optional)
# glossary-files = "/path/to/glossary1.csv,/path/to/glossary2.csv"

# Output control
no-dual = false
no-mono = false
min-text-length = 5
report-interval = 0.5

# Offline assets management
# Uncomment one of these options as needed:
# generate-offline-assets = "/path/to/output/dir"
# restore-offline-assets = "/path/to/offline_assets_package.zip"
```

## Python API

The current recommended way to call BabelDOC in Python is to call the `high_level.do_translate_async_stream` function of [pdf2zh next](https://github.com/PDFMathTranslate/PDFMathTranslate-next).

> [!WARNING]
> **All APIs of BabelDOC should be considered as internal APIs, and any direct use of BabelDOC is not supported.**

## Background

There are a lot projects and teams working on to make document editing and translating easier like:

- [mathpix](https://mathpix.com/)
- [Doc2X](https://doc2x.noedgeai.com/)
- [minerU](https://github.com/opendatalab/MinerU)
- [PDFMathTranslate](https://github.com/funstory-ai/yadt)

There are also some solutions to solve specific parts of the problem like:

- [layoutreader](https://github.com/microsoft/unilm/tree/master/layoutreader): the read order of the text block in a pdf
- [Surya](https://github.com/surya-is/surya): the structure of the pdf

This project hopes to promote a standard pipeline and interface to solve the problem.

In fact, there are two main stages of a PDF parser or translator:

- **Parsing**: A stage of parsing means to get the structure of the pdf such as text blocks, images, tables, etc.
- **Rendering**: A stage of rendering means to render the structure into a new pdf or other format.

For a service like mathpix, it will parse the pdf into a structure may be in a XML format, and then render them using a single column reader order as [layoutreader](https://github.com/microsoft/unilm/tree/master/layoutreader) does. The bad news is that the original structure lost.

Some people will use Adobe PDF Parser because it will generate a Word document and it keeps the original structure. But it is somewhat expensive.
And you know, a pdf or word document is not a good format for reading in mobile devices.

We offer an intermediate representation of the results from parser and can be rendered into a new pdf or other format. The pipeline is also a plugin-based system which everybody can add their new model, ocr, renderer, etc.

## Roadmap

- [ ] Add line support
- [ ] Add table support
- [ ] Add cross-page/cross-column paragraph support
- [ ] More advanced typesetting features
- [ ] Outline support
- [ ] ...

Our first 1.0 version goal is to finish a translation from [PDF Reference, Version 1.7](https://opensource.adobe.com/dc-acrobat-sdk-docs/pdfstandards/pdfreference1.7old.pdf) to the following language version:

- Simplified Chinese
- Traditional Chinese
- Japanese
- Spanish

And meet the following requirements:

- layout error less than 1%
- content loss less than 1%

## Version Number Explanation

This project uses a combination of [Semantic Versioning](https://semver.org/) and [Pride Versioning](https://pridever.org/). The version number format is: "0.MAJOR.MINOR".

> [!NOTE]
>
> The API compatibility here mainly refers to the compatibility with [pdf2zh_next](https://github.com/PDFMathTranslate/PDFMathTranslate-next).


- MAJOR: Incremented by 1 when API incompatible changes are made or when proud improvements are implemented.

- MINOR: Incremented by 1 when any API compatible changes are made.

## Known Issues

1. Parsing errors in the author and reference sections; they get merged into one paragraph after translation.
2. Lines are not supported.
3. Does not support drop caps.
4. Large pages will be skipped.

## How to Contribute

We encourage you to contribute to YADT! Please check out the [CONTRIBUTING](https://github.com/funstory-ai/yadt/blob/main/docs/CONTRIBUTING.md) guide.

Everyone interacting in YADT and its sub-projects' codebases, issue trackers, chat rooms, and mailing lists is expected to follow the YADT [Code of Conduct](https://github.com/funstory-ai/yadt/blob/main/docs/CODE_OF_CONDUCT.md).

[Immersive Translation](https://immersivetranslate.com) sponsors monthly Pro membership redemption codes for active contributors to this project, see details at: [CONTRIBUTOR_REWARD.md](https://github.com/funstory-ai/BabelDOC/blob/main/docs/CONTRIBUTOR_REWARD.md)

## Acknowledgements

- [PDFMathTranslate](https://github.com/Byaidu/PDFMathTranslate)
- [DocLayout-YOLO](https://github.com/opendatalab/DocLayout-YOLO)
- [pdfminer](https://github.com/pdfminer/pdfminer.six)
- [PyMuPDF](https://github.com/pymupdf/PyMuPDF)
- [Asynchronize](https://github.com/multimeric/Asynchronize/tree/master?tab=readme-ov-file)
- [PriorityThreadPoolExecutor](https://github.com/oleglpts/PriorityThreadPoolExecutor)

<h2 id="star_hist">Star History</h2>

<a href="https://star-history.com/#funstory-ai/babeldoc&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=funstory-ai/babeldoc&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=funstory-ai/babeldoc&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=funstory-ai/babeldoc&type=Date"/>
 </picture>
</a>

> [!WARNING]
> **Important Interaction Note for `--auto-enable-ocr-workaround`:**
>
> When `--auto-enable-ocr-workaround` is set to `true` (either via command line or config file):
>
> 1.  During the initial setup, the values for `ocr_workaround` and `skip_scanned_detection` will be forced to `false` by `TranslationConfig`, regardless of whether you also set `--ocr-workaround` or `--skip-scanned-detection` flags.
> 2.  Then, during the scanned document detection phase (`DetectScannedFile` stage):
>     *   If the document is identified as heavily scanned (e.g., >80% scanned pages) AND `auto_enable_ocr_workaround` is `true` (i.e., `translation_config.auto_enable_ocr_workaround` is true), the system will then attempt to set both `ocr_workaround` to `true` and `skip_scanned_detection` to `true`.
>
> This means that `--auto-enable-ocr-workaround` effectively gives the system control to enable OCR processing for scanned documents, potentially overriding manual settings for `--ocr-workaround` and `--skip_scanned_detection` based on its detection results. If the document is *not* detected as heavily scanned, then the initial `false` values for `ocr_workaround` and `skip_scanned_detection` (forced by `--auto-enable-ocr-workaround` at the `TranslationConfig` initialization stage) will remain in effect unless changed by other logic.
