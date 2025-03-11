import asyncio
import logging
from pathlib import Path
from typing import Any

import configargparse
import tqdm
from rich.progress import BarColumn
from rich.progress import MofNCompleteColumn
from rich.progress import Progress
from rich.progress import TextColumn
from rich.progress import TimeElapsedColumn
from rich.progress import TimeRemainingColumn

import babeldoc.assets.assets
import babeldoc.high_level
from babeldoc.document_il.translator.translator import BingTranslator
from babeldoc.document_il.translator.translator import GoogleTranslator
from babeldoc.document_il.translator.translator import OpenAITranslator
from babeldoc.document_il.translator.translator import TranslateTranslator
from babeldoc.document_il.translator.translator import set_translate_rate_limiter
from babeldoc.docvision.doclayout import DocLayoutModel
from babeldoc.docvision.rpc_doclayout import RpcDocLayoutModel
from babeldoc.translation_config import TranslationConfig
from babeldoc.translation_config import WatermarkOutputMode

logger = logging.getLogger(__name__)
__version__ = "0.1.31"


def create_parser():
    parser = configargparse.ArgParser(
        config_file_parser_class=configargparse.TomlConfigParser(["babeldoc"]),
    )
    parser.add_argument(
        "-c",
        "--config",
        is_config_file=True,
        help="config file path",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    parser.add_argument(
        "--files",
        action="append",
        help="One or more paths to PDF files.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Use debug logging level.",
    )
    parser.add_argument(
        "--warmup",
        action="store_true",
        help="Only download and verify required assets then exit.",
    )
    parser.add_argument(
        "--rpc-doclayout",
        help="RPC service host address for document layout analysis",
    )
    parser.add_argument(
        "--generate-offline-assets",
        default=None,
        help="Generate offline assets package in the specified directory",
    )
    parser.add_argument(
        "--restore-offline-assets",
        default=None,
        help="Restore offline assets package from the specified file",
    )
    # translation option argument group
    translation_group = parser.add_argument_group(
        "Translation",
        description="Used during translation",
    )
    translation_group.add_argument(
        "--pages",
        "-p",
        help="Pages to translate. If not set, translate all pages. like: 1,2,1-,-3,3-5",
    )
    translation_group.add_argument(
        "--min-text-length",
        type=int,
        default=5,
        help="Minimum text length to translate (default: 5)",
    )
    translation_group.add_argument(
        "--lang-in",
        "-li",
        default="en",
        help="The code of source language.",
    )
    translation_group.add_argument(
        "--lang-out",
        "-lo",
        default="zh",
        help="The code of target language.",
    )
    translation_group.add_argument(
        "--output",
        "-o",
        help="Output directory for files. if not set, use same as input.",
    )
    translation_group.add_argument(
        "--qps",
        "-q",
        type=int,
        default=4,
        help="QPS limit of translation service",
    )
    translation_group.add_argument(
        "--ignore-cache",
        action="store_true",
        help="Ignore translation cache.",
    )
    translation_group.add_argument(
        "--no-dual",
        action="store_true",
        help="Do not output bilingual PDF files",
    )
    translation_group.add_argument(
        "--no-mono",
        action="store_true",
        help="Do not output monolingual PDF files",
    )
    translation_group.add_argument(
        "--formular-font-pattern",
        help="Font pattern to identify formula text",
    )
    translation_group.add_argument(
        "--formular-char-pattern",
        help="Character pattern to identify formula text",
    )
    translation_group.add_argument(
        "--split-short-lines",
        action="store_true",
        help="Force split short lines into different paragraphs (may cause poor typesetting & bugs)",
    )
    translation_group.add_argument(
        "--short-line-split-factor",
        type=float,
        default=0.8,
        help="Split threshold factor. The actual threshold is the median length of all lines on the current page * this factor",
    )
    translation_group.add_argument(
        "--skip-clean",
        action="store_true",
        help="Skip PDF cleaning step",
    )
    translation_group.add_argument(
        "--dual-translate-first",
        action="store_true",
        help="Put translated pages first in dual PDF mode",
    )
    translation_group.add_argument(
        "--disable-rich-text-translate",
        action="store_true",
        help="Disable rich text translation (may help improve compatibility with some PDFs)",
    )
    translation_group.add_argument(
        "--enhance-compatibility",
        action="store_true",
        help="Enable all compatibility enhancement options (equivalent to --skip-clean --dual-translate-first --disable-rich-text-translate)",
    )
    translation_group.add_argument(
        "--use-alternating-pages-dual",
        action="store_true",
        help="Use alternating pages mode for dual PDF. When enabled, original and translated pages are arranged in alternate order.",
    )
    translation_group.add_argument(
        "--watermark-output-mode",
        type=str,
        choices=["watermarked", "no_watermark", "both"],
        default="watermarked",
        help="Control watermark output mode: 'watermarked' (default) adds watermark to translated PDF, 'no_watermark' doesn't add watermark, 'both' outputs both versions.",
    )
    translation_group.add_argument(
        "--no-watermark",
        action="store_true",
        help="[DEPRECATED] Use --watermark-output-mode=no_watermark instead. Do not add watermark to the translated PDF.",
    )
    translation_group.add_argument(
        "--report-interval",
        type=float,
        default=0.1,
        help="Progress report interval in seconds (default: 0.1)",
    )
    translation_group.add_argument(
        "--paper",
        action="store_true",
        help="Extract and use the paper title in translation prompts to improve context.",
    )
    # service option argument group
    service_group = translation_group.add_mutually_exclusive_group()
    service_group.add_argument(
        "--openai",
        action="store_true",
        help="Use OpenAI translator.",
    )
    service_group.add_argument(
        "--google",
        action="store_true",
        help="Use Google translator.",
    )
    service_group.add_argument(
        "--translate",
        action="store_true",
        help="Use translate translator.",
    )
    service_group.add_argument(
        "--bing",
        action="store_true",
        help="Use Bing translator.",
    )
    service_group = parser.add_argument_group(
        "Translation - OpenAI Options",
        description="OpenAI specific options",
    )
    service_group.add_argument(
        "--openai-model",
        default="gpt-4o-mini",
        help="The OpenAI model to use for translation.",
    )
    service_group.add_argument(
        "--openai-base-url",
        help="The base URL for the OpenAI API.",
    )
    service_group.add_argument(
        "--openai-api-key",
        "-k",
        help="The API key for the OpenAI API.",
    )
    service_group = parser.add_argument_group(
        "Translation - Translate Options",
        description="Translate specific options",
    )
    service_group.add_argument(
        "--translate-url",
        help="The base URL for the Translation API.",
    )

    return parser


async def main():
    parser = create_parser()
    args: Any = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.generate_offline_assets:
        babeldoc.assets.assets.generate_offline_assets_package(
            Path(args.generate_offline_assets)
        )
        logger.info("Offline assets package generated, exiting...")
        return

    if args.restore_offline_assets:
        babeldoc.assets.assets.restore_offline_assets_package(
            Path(args.restore_offline_assets)
        )
        logger.info("Offline assets package restored, exiting...")
        return

    if args.warmup:
        babeldoc.assets.assets.warmup()
        logger.info("Warmup completed, exiting...")
        return

    # 验证翻译服务选择
    if not (args.openai or args.google or args.bing or args.translate):
        parser.error("必须选择一个翻译服务：--openai、--google、--bing 或 --translate")

    # 验证 OpenAI 参数
    if args.openai and not args.openai_api_key:
        parser.error("使用 OpenAI 服务时必须提供 API key")

    # 实例化翻译器
    if args.openai:
        translator = OpenAITranslator(
            lang_in=args.lang_in,
            lang_out=args.lang_out,
            model=args.openai_model,
            base_url=args.openai_base_url,
            api_key=args.openai_api_key,
            ignore_cache=args.ignore_cache,
        )
    elif args.bing:
        translator = BingTranslator(
            lang_in=args.lang_in,
            lang_out=args.lang_out,
            ignore_cache=args.ignore_cache,
        )
    elif args.translate:
        translator = TranslateTranslator(
            lang_in=args.lang_in,
            lang_out=args.lang_out,
            ignore_cache=args.ignore_cache,
            url=args.translate_url,
        )
    else:
        translator = GoogleTranslator(
            lang_in=args.lang_in,
            lang_out=args.lang_out,
            ignore_cache=args.ignore_cache,
        )

    # 设置翻译速率限制
    set_translate_rate_limiter(args.qps)

    # 初始化文档布局模型
    if args.rpc_doclayout:
        doc_layout_model = RpcDocLayoutModel(host=args.rpc_doclayout)
    else:
        doc_layout_model = DocLayoutModel.load_onnx()

    pending_files = []
    for file in args.files:
        # 清理文件路径，去除两端的引号
        if file.startswith("--files="):
            file = file[len("--files=") :]
        file = file.lstrip("-").strip("\"'")
        if not Path(file).exists():
            logger.error(f"文件不存在：{file}")
            exit(1)
        if not file.endswith(".pdf"):
            logger.error(f"文件不是 PDF 文件：{file}")
            exit(1)
        pending_files.append(file)

    if args.output:
        if not Path(args.output).exists():
            logger.info(f"输出目录不存在，创建：{args.output}")
            try:
                Path(args.output).mkdir(parents=True, exist_ok=True)
            except OSError:
                logger.critical(
                    f"Failed to create output folder at {args.output}",
                    exc_info=True,
                )
                exit(1)
    else:
        args.output = None

    watermark_output_mode = WatermarkOutputMode.Watermarked
    if args.no_watermark:
        watermark_output_mode = WatermarkOutputMode.NoWatermark
    elif args.watermark_output_mode == "both":
        watermark_output_mode = WatermarkOutputMode.Both
    elif args.watermark_output_mode == "watermarked":
        watermark_output_mode = WatermarkOutputMode.Watermarked
    elif args.watermark_output_mode == "no_watermark":
        watermark_output_mode = WatermarkOutputMode.NoWatermark

    for file in pending_files:
        # 清理文件路径，去除两端的引号
        file = file.strip("\"'")
        # 创建配置对象
        config = TranslationConfig(
            input_file=file,
            font=None,
            pages=args.pages,
            output_dir=args.output,
            translator=translator,
            debug=args.debug,
            lang_in=args.lang_in,
            lang_out=args.lang_out,
            no_dual=args.no_dual,
            no_mono=args.no_mono,
            qps=args.qps,
            formular_font_pattern=args.formular_font_pattern,
            formular_char_pattern=args.formular_char_pattern,
            split_short_lines=args.split_short_lines,
            short_line_split_factor=args.short_line_split_factor,
            doc_layout_model=doc_layout_model,
            skip_clean=args.skip_clean,
            dual_translate_first=args.dual_translate_first,
            disable_rich_text_translate=args.disable_rich_text_translate,
            enhance_compatibility=args.enhance_compatibility,
            use_alternating_pages_dual=args.use_alternating_pages_dual,
            report_interval=args.report_interval,
            min_text_length=args.min_text_length,
            watermark_output_mode=watermark_output_mode,
            use_paper_title=args.paper,
        )

        # Create progress handler
        progress_context, progress_handler = create_progress_handler(config)

        # 开始翻译
        with progress_context:
            async for event in babeldoc.high_level.async_translate(config):
                progress_handler(event)
                if config.debug:
                    logger.debug(event)
                if event["type"] == "finish":
                    result = event["translate_result"]
                    logger.info(str(result))
                    break


def create_progress_handler(translation_config: TranslationConfig):
    """Create a progress handler function based on the configuration.

    Args:
        translation_config: The translation configuration.

    Returns:
        A tuple of (progress_context, progress_handler), where progress_context is a context
        manager that should be used to wrap the translation process, and progress_handler
        is a function that will be called with progress events.
    """
    if translation_config.use_rich_pbar:
        progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        )
        translate_task_id = progress.add_task("translate", total=100)
        stage_tasks = {}

        def progress_handler(event):
            if event["type"] == "progress_start":
                stage_tasks[event["stage"]] = progress.add_task(
                    f"{event['stage']}",
                    total=event.get("stage_total", 100),
                )
            elif event["type"] == "progress_update":
                stage = event["stage"]
                if stage in stage_tasks:
                    progress.update(
                        stage_tasks[stage],
                        completed=event["stage_current"],
                        total=event["stage_total"],
                        description=f"{event['stage']}",
                        refresh=True,
                    )
                progress.update(
                    translate_task_id,
                    completed=event["overall_progress"],
                    refresh=True,
                )
            elif event["type"] == "progress_end":
                stage = event["stage"]
                if stage in stage_tasks:
                    progress.update(
                        stage_tasks[stage],
                        completed=event["stage_total"],
                        total=event["stage_total"],
                        description=f"{event['stage']}",
                        refresh=True,
                    )
                    progress.update(
                        translate_task_id,
                        completed=event["overall_progress"],
                        refresh=True,
                    )
                progress.refresh()

        return progress, progress_handler
    else:
        pbar = tqdm.tqdm(total=100, desc="translate")

        def progress_handler(event):
            if event["type"] == "progress_update":
                pbar.update(event["overall_progress"] - pbar.n)
                pbar.set_description(
                    f"{event['stage']} ({event['stage_current']}/{event['stage_total']})",
                )
            elif event["type"] == "progress_end":
                pbar.set_description(f"{event['stage']} (Complete)")
                pbar.refresh()

        return pbar, progress_handler


# for backward compatibility
def create_cache_folder():
    return babeldoc.high_level.create_cache_folder()


# for backward compatibility
def download_font_assets():
    return babeldoc.high_level.download_font_assets()


def cli():
    """Command line interface entry point."""
    from rich.logging import RichHandler

    logging.basicConfig(level=logging.INFO, handlers=[RichHandler()])

    logging.getLogger("httpx").setLevel("CRITICAL")
    logging.getLogger("httpx").propagate = False
    logging.getLogger("openai").setLevel("CRITICAL")
    logging.getLogger("openai").propagate = False
    logging.getLogger("httpcore").setLevel("CRITICAL")
    logging.getLogger("httpcore").propagate = False
    logging.getLogger("http11").setLevel("CRITICAL")
    logging.getLogger("http11").propagate = False
    for v in logging.Logger.manager.loggerDict.values():
        if getattr(v, "name", None) is None:
            continue
        if (
            v.name.startswith("pdfminer")
            or v.name.startswith("peewee")
            or v.name.startswith("httpx")
            or "http11" in v.name
            or "openai" in v.name
        ):
            v.disabled = True
            v.propagate = False

    babeldoc.high_level.init()
    asyncio.run(main())


if __name__ == "__main__":
    cli()
