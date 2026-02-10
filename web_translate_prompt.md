# Web PDF 翻译应用 - AI 编程指令

## 项目目标

基于 BabelDOC 这个 Python PDF 翻译库，创建一个网页应用。用户可以上传 PDF 文件，选择源语言和目标语言，提交翻译任务后实时查看进度，翻译完成后下载翻译后的 PDF 文件。

## 技术栈要求

- **后端**: Python + FastAPI
- **前端**: 纯 HTML + CSS + JavaScript（单页面，不用前端框架）
- **异步任务**: FastAPI 后台任务（BackgroundTasks）
- **实时进度**: SSE（Server-Sent Events）
- **依赖**: BabelDOC（`pip install BabelDOC`）

## 项目结构

```
web_translate/
├── app.py              # FastAPI 后端主文件
├── templates/
│   └── index.html      # 前端页面
├── uploads/            # 用户上传的 PDF 临时存放
├── outputs/            # 翻译完成的 PDF 存放
└── requirements.txt    # Python 依赖
```

## BabelDOC 核心 API 调用方式

这是调用 BabelDOC 进行翻译的核心代码，请严格按照此方式调用：

```python
import asyncio
import babeldoc.format.pdf.high_level
from babeldoc.docvision.doclayout import DocLayoutModel
from babeldoc.format.pdf.translation_config import TranslationConfig, WatermarkOutputMode
from babeldoc.translator.translator import OpenAITranslator, set_translate_rate_limiter

# 初始化（应用启动时调用一次）
babeldoc.format.pdf.high_level.init()

# 文档版面分析模型（应用启动时加载一次，全局共享）
doc_layout_model = DocLayoutModel.load_onnx()

# 创建翻译器（每次翻译任务创建一个）
translator = OpenAITranslator(
    lang_in="en",
    lang_out="zh",
    model="gpt-4o-mini",
    base_url="https://api.openai.com/v1",
    api_key="your-api-key",
)

# 设置 QPS 限制
set_translate_rate_limiter(4)

# 创建翻译配置
config = TranslationConfig(
    input_file="/path/to/input.pdf",    # 输入文件路径（str 或 Path）
    translator=translator,
    lang_in="en",
    lang_out="zh",
    doc_layout_model=doc_layout_model,  # 共享模型实例
    output_dir="/path/to/output",       # 输出目录（str 或 Path）
    qps=4,
    watermark_output_mode=WatermarkOutputMode.NoWatermark,
)

# 异步翻译（推荐，可获取实时进度）
async for event in babeldoc.format.pdf.high_level.async_translate(config):
    if event["type"] == "progress_update":
        # event["overall_progress"] 是 0-100 的浮点数
        # event["stage"] 是当前阶段名称字符串
        pass
    elif event["type"] == "finish":
        result = event["translate_result"]
        # result.mono_pdf_path  → 单语翻译 PDF 的 Path 对象（可能为 None）
        # result.dual_pdf_path  → 双语对照 PDF 的 Path 对象（可能为 None）
        break
    elif event["type"] == "error":
        error = event["error"]
        break
```

### 重要注意事项

1. `DocLayoutModel.load_onnx()` 加载 ONNX 模型很慢且占内存，必须在应用启动时加载一次，所有任务共享这一个实例。
2. `babeldoc.format.pdf.high_level.init()` 只需调用一次。
3. `TranslationConfig` 的 `input_file` 参数接受文件路径字符串或 Path 对象。
4. 翻译完成后 `result.mono_pdf_path` 和 `result.dual_pdf_path` 是 `pathlib.Path` 对象，指向输出目录中生成的文件。
5. 单次翻译可能耗时数分钟，绝不能在普通 HTTP 请求中同步等待。

## 功能需求

### 1. 前端页面（index.html）

页面包含以下内容：

- **配置区域**（首次使用时填写，保存到 localStorage）：
  - API Base URL 输入框
  - API Key 输入框
  - 模型名称输入框（默认值 "gpt-4o-mini"）

- **翻译区域**：
  - PDF 文件上传（拖拽 + 点击，限制只接受 .pdf 文件）
  - 源语言下拉框（默认 English）
  - 目标语言下拉框（默认 中文）
  - "开始翻译" 按钮

- **进度区域**（翻译开始后显示）：
  - 进度条（百分比）
  - 当前阶段名称文本
  - 翻译耗时计时器

- **结果区域**（翻译完成后显示）：
  - "下载单语 PDF" 按钮
  - "下载双语 PDF" 按钮

- **常用语言列表**（至少包含以下选项）：
  ```
  en - English
  zh - 中文
  ja - 日本語
  ko - 한국어
  fr - Français
  de - Deutsch
  es - Español
  ru - Русский
  pt - Português
  it - Italiano
  ```

### 2. 后端 API（app.py）

需要实现以下接口：

#### POST /api/translate
- 接收：上传的 PDF 文件 + API 配置（base_url, api_key, model）+ 语言配置（lang_in, lang_out）
- 处理：保存文件到 uploads/ 目录，创建翻译任务，返回 task_id
- 响应：`{"task_id": "xxx"}`

#### GET /api/progress/{task_id}
- SSE 接口，持续推送翻译进度
- 事件格式：
  ```
  data: {"progress": 45.2, "stage": "Translate Paragraphs"}
  data: {"progress": 100, "stage": "done", "mono_pdf": "xxx.pdf", "dual_pdf": "xxx.pdf"}
  data: {"progress": -1, "stage": "error", "message": "错误信息"}
  ```

#### GET /api/download/{task_id}/{file_type}
- file_type 为 "mono" 或 "dual"
- 返回翻译后的 PDF 文件供下载

### 3. 任务管理

- 用一个全局字典 `tasks = {}` 存储任务状态，key 为 task_id（用 uuid4 生成）
- 每个任务存储：状态（pending/running/done/error）、进度、阶段名、输出文件路径、错误信息
- 翻译任务在后台协程中执行，通过更新任务字典来传递进度
- 任务完成 30 分钟后自动清理上传文件和输出文件

### 4. 错误处理

- 上传文件非 PDF 时返回错误
- API 配置为空时前端提示
- 翻译过程中出错时通过 SSE 推送错误信息
- 后端全局异常捕获，避免崩溃

## 界面风格

- 简洁现代，白色背景，卡片式布局
- 响应式设计，移动端可用
- 进度条使用蓝色渐变动画
- 按钮使用圆角，有 hover 效果

## requirements.txt 内容

```
BabelDOC
fastapi
uvicorn[standard]
python-multipart
```

## 启动方式

```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
```

浏览器打开 http://localhost:8000 即可使用。
