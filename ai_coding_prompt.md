# 任务说明

你收到了一个名为 **BabelDOC** 的 Python 项目压缩包。这是一个 PDF 文档翻译库，能够在保留原始排版的前提下将 PDF 翻译成其他语言。

你的任务是：**在这个仓库中新建一个 `web_translate/` 子目录，在其中实现一个基于 FastAPI 的 Web 应用**，让用户可以通过浏览器上传 PDF、翻译并下载结果。

详细的功能规格文档已经写好，位于仓库根目录：**`web_translate_prompt.md`**。请先完整阅读该文件，它是你的主要需求文档。

本文件是对 `web_translate_prompt.md` 的补充，提供仓库代码层面的精确信息，帮助你正确调用 BabelDOC 的 API。

---

## 第一步：了解仓库结构

在开始编码之前，请先阅读以下关键文件，理解 BabelDOC 的内部实现：

| 文件路径 | 作用 |
|---|---|
| `babeldoc/format/pdf/high_level.py` | 翻译流水线入口，包含 `init()`、`async_translate()` 函数 |
| `babeldoc/format/pdf/translation_config.py` | `TranslationConfig` 类和 `TranslateResult` 类的定义 |
| `babeldoc/translator/translator.py` | `OpenAITranslator` 和 `set_translate_rate_limiter` 的定义 |
| `babeldoc/docvision/base_doclayout.py` | `DocLayoutModel.load_onnx()` 的定义 |
| `babeldoc/docvision/doclayout.py` | DocLayoutModel 的具体实现 |
| `babeldoc/main.py` | CLI 的完整调用示例（可作为参考） |
| `web_translate_prompt.md` | **主需求文档，必须完整阅读** |

---

## 第二步：BabelDOC API 精确调用方式

以下是从仓库源码中提取的精确 API，请严格按此调用，不要猜测或改动。

### 2.1 正确的 import 路径

```python
import babeldoc.format.pdf.high_level
from babeldoc.docvision.doclayout import DocLayoutModel
from babeldoc.format.pdf.translation_config import TranslationConfig, WatermarkOutputMode
from babeldoc.translator.translator import OpenAITranslator, set_translate_rate_limiter
```

### 2.2 应用启动时（只执行一次）

```python
# 初始化缓存目录（来自 high_level.py:1248 的 init() 函数）
babeldoc.format.pdf.high_level.init()

# 加载版面分析 ONNX 模型（来自 base_doclayout.py:42 的 load_onnx()）
# 此操作极耗时（10~30 秒）且占内存，全局只能执行一次
doc_layout_model = DocLayoutModel.load_onnx()
```

### 2.3 OpenAITranslator 构造函数签名（来自 translator.py:207）

```python
translator = OpenAITranslator(
    lang_in=lang_in,        # str，如 "en"
    lang_out=lang_out,      # str，如 "zh"
    model=model_name,       # str，如 "gpt-4o-mini"
    base_url=api_base_url,  # str，如 "https://api.openai.com/v1"，可为 None（默认 OpenAI）
    api_key=api_key,        # str
    ignore_cache=False,     # bool，默认 False
)
```

### 2.4 TranslationConfig 构造函数关键参数（来自 translation_config.py:143）

```python
config = TranslationConfig(
    translator=translator,              # OpenAITranslator 实例（必填）
    input_file=input_pdf_path,          # str 或 pathlib.Path（必填）
    lang_in=lang_in,                    # str（必填）
    lang_out=lang_out,                  # str（必填）
    doc_layout_model=doc_layout_model,  # 全局共享的模型实例（必填）
    output_dir=output_dir,              # str 或 pathlib.Path，输出目录（可选，默认当前目录）
    qps=4,                              # int，每秒 API 调用次数（可选）
    watermark_output_mode=WatermarkOutputMode.NoWatermark,  # 不加水印
    no_dual=False,                      # False = 同时生成双语 PDF
    no_mono=False,                      # False = 同时生成单语 PDF
)
```

### 2.5 async_translate 事件结构（来自 high_level.py:366）

`babeldoc.format.pdf.high_level.async_translate(config)` 是一个 **async generator**，yield 的事件类型如下：

```python
async for event in babeldoc.format.pdf.high_level.async_translate(config):
    event_type = event["type"]  # str

    if event_type == "progress_update":
        progress = event["overall_progress"]  # float，0.0 ~ 100.0
        stage    = event["stage"]             # str，当前阶段名称，如 "Translate Paragraphs"

    elif event_type == "progress_start":
        stage       = event["stage"]        # str
        stage_total = event["stage_total"]  # int，该阶段总步数

    elif event_type == "progress_end":
        stage = event["stage"]  # str

    elif event_type == "finish":
        result = event["translate_result"]  # TranslateResult 对象
        # result.mono_pdf_path  → pathlib.Path 或 None
        # result.dual_pdf_path  → pathlib.Path 或 None
        break

    elif event_type == "error":
        error_msg = event["error"]  # str
        break
```

### 2.6 TranslateResult 对象的属性（来自 translation_config.py:489）

```python
result.mono_pdf_path             # pathlib.Path | None，单语翻译 PDF
result.dual_pdf_path             # pathlib.Path | None，双语对照 PDF
result.no_watermark_mono_pdf_path # pathlib.Path | None（与 mono_pdf_path 相同，当 NoWatermark 模式时）
result.no_watermark_dual_pdf_path # pathlib.Path | None（与 dual_pdf_path 相同，当 NoWatermark 模式时）
result.total_seconds             # float，翻译耗时秒数
```

---

## 第三步：关键注意事项

1. **模型只加载一次**：`DocLayoutModel.load_onnx()` 在 FastAPI 的 `lifespan` 事件中调用，结果保存为模块级全局变量，所有请求共享。绝对不能在每次翻译请求时重新调用。

2. **翻译必须在后台运行**：翻译一篇论文通常需要 5~15 分钟。必须用 `asyncio.create_task()` 在后台启动翻译协程，HTTP 接口立即返回 `task_id`，前端再通过 SSE 订阅进度。

3. **SSE 依赖额外包**：进度推送接口使用 SSE，需要安装 `sse-starlette` 包（加入 requirements.txt），然后：
   ```python
   from sse_starlette.sse import EventSourceResponse
   ```

4. **async_translate 运行环境**：`async_translate` 内部通过 `loop.run_in_executor(None, do_translate, ...)` 在线程池中执行翻译（因为翻译是 CPU/IO 密集型同步代码），外部 async for 循环可以安全地在 FastAPI 的 asyncio 事件循环中运行，不会阻塞。

5. **输出目录**：每个任务建议使用独立的输出子目录 `outputs/{task_id}/`，避免不同任务的文件互相覆盖。

6. **不要修改仓库原有文件**：所有新建文件放在 `web_translate/` 目录下，不要改动 `babeldoc/` 目录下的任何文件。

7. **requirements.txt 必须包含**：
   ```
   BabelDOC
   fastapi
   uvicorn[standard]
   python-multipart
   sse-starlette
   ```

---

## 第四步：需要创建的文件清单

在仓库根目录下新建以下文件，**不要修改其他任何已有文件**：

```
web_translate/
├── app.py              # FastAPI 后端主文件
├── templates/
│   └── index.html      # 前端单页面
└── requirements.txt    # Python 依赖
```

---

## 第五步：完成标准

完成后，用户只需执行以下命令即可运行：

```bash
cd web_translate
pip install -r requirements.txt
python app.py
```

然后打开浏览器访问 `http://localhost:8000`，应能看到完整可用的翻译网页。

功能验收点：
- [ ] 能填写并保存 API 配置（持久化到 localStorage）
- [ ] 能上传 PDF 文件（支持拖拽）
- [ ] 点击翻译后进度条实时更新
- [ ] 翻译完成后显示下载按钮
- [ ] 能成功下载单语 PDF 和双语 PDF
- [ ] 翻译出错时页面显示错误信息
- [ ] 翻译中途刷新页面后，进度条能重新连接并继续显示（因为后台任务仍在运行）
