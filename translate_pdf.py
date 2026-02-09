import asyncio
import logging

import babeldoc.format.pdf.high_level
from babeldoc.docvision.doclayout import DocLayoutModel
from babeldoc.format.pdf.translation_config import TranslationConfig, WatermarkOutputMode
from babeldoc.translator.translator import OpenAITranslator, set_translate_rate_limiter

# ============================================================
# 请在这里填写你的配置
# ============================================================

# 输入 PDF 文件路径
INPUT_PDF = "/path/to/your/input.pdf"

# 输出目录（翻译后的 PDF 会保存到这里）
OUTPUT_DIR = "/path/to/your/output"

# 模型 API 配置
OPENAI_BASE_URL = "<YOUR_API_BASE_URL>"  # 例如 "https://api.openai.com/v1"
OPENAI_API_KEY = "<YOUR_API_KEY>"
OPENAI_MODEL = "<YOUR_MODEL_NAME>"       # 例如 "gpt-4o-mini"

# 翻译语言配置
LANG_IN = "en"   # 源语言
LANG_OUT = "zh"  # 目标语言

# QPS 限制（每秒请求数）
QPS = 4

# ============================================================


async def translate_pdf():
    # 1. 初始化（创建缓存目录等）
    babeldoc.format.pdf.high_level.init()

    # 2. 创建翻译器
    translator = OpenAITranslator(
        lang_in=LANG_IN,
        lang_out=LANG_OUT,
        model=OPENAI_MODEL,
        base_url=OPENAI_BASE_URL,
        api_key=OPENAI_API_KEY,
    )

    # 3. 设置翻译速率限制
    set_translate_rate_limiter(QPS)

    # 4. 加载文档版面分析模型（首次运行会自动下载）
    doc_layout_model = DocLayoutModel.load_onnx()

    # 5. 创建翻译配置
    config = TranslationConfig(
        input_file=INPUT_PDF,
        translator=translator,
        lang_in=LANG_IN,
        lang_out=LANG_OUT,
        doc_layout_model=doc_layout_model,
        output_dir=OUTPUT_DIR,
        qps=QPS,
        watermark_output_mode=WatermarkOutputMode.NoWatermark,
    )

    # 6. 异步翻译，实时打印进度
    async for event in babeldoc.format.pdf.high_level.async_translate(config):
        if event["type"] == "progress_update":
            stage = event["stage"]
            progress = event["overall_progress"]
            print(f"[{progress:5.1f}%] {stage}")
        elif event["type"] == "finish":
            result = event["translate_result"]
            print("\n翻译完成！")
            if result.mono_pdf_path:
                print(f"  单语 PDF: {result.mono_pdf_path}")
            if result.dual_pdf_path:
                print(f"  双语 PDF: {result.dual_pdf_path}")
            break
        elif event["type"] == "error":
            print(f"\n翻译出错: {event['error']}")
            break


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(translate_pdf())
