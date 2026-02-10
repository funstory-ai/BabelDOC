# AI 编程指令：基于 BabelDOC 的 Web PDF 翻译应用

## 一、项目目标

请基于 BabelDOC（一个 Python PDF 翻译库）创建一个完整的网页应用。用户流程如下：

1. 打开网页，填写自己的 LLM API 配置（地址、密钥、模型名）
2. 上传一个 PDF 文件，选择源语言和目标语言
3. 点击"开始翻译"，页面实时显示翻译进度
4. 翻译完成后，点击按钮下载翻译后的 PDF（单语版 / 双语对照版）

## 二、技术栈

- **后端**: Python 3.12 + FastAPI
- **前端**: 单个 HTML 文件（内联 CSS + JavaScript，不使用任何前端框架）
- **实时进度推送**: SSE（Server-Sent Events）
- **核心依赖**: BabelDOC（通过 `pip install BabelDOC` 安装）

## 三、项目文件结构

```
web_translate/
├── app.py              # FastAPI 后端（唯一的后端文件）
├── templates/
│   └── index.html      # 前端页面（唯一的前端文件）
├── uploads/            # 运行时自动创建，存放用户上传的 PDF
├── outputs/            # 运行时自动创建，存放翻译后的 PDF
└── requirements.txt
```

requirements.txt 内容：
```
BabelDOC
fastapi
uvicorn[standard]
python-multipart
```

## 四、BabelDOC API 调用规范（极其重要，请严格遵守）

### 4.1 所需的 import

```python
import babeldoc.format.pdf.high_level
from babeldoc.docvision.doclayout import DocLayoutModel
from babeldoc.format.pdf.translation_config import TranslationConfig, WatermarkOutputMode
from babeldoc.translator.translator import OpenAITranslator, set_translate_rate_limiter
```

### 4.2 应用启动时执行一次的初始化

```python
# 创建缓存目录
babeldoc.format.pdf.high_level.init()

# 加载文档版面分析模型（ONNX 模型，加载较慢，约需 10-30 秒）
# 必须全局加载一次，所有翻译任务共享同一个实例，绝对不要每次任务都重新加载
doc_layout_model = DocLayoutModel.load_onnx()
```

### 4.3 每次翻译任务的完整流程

```python
import asyncio

async def run_translate(
    input_pdf_path: str,     # 用户上传的 PDF 文件路径
    output_dir: str,         # 翻译结果输出目录
    api_base_url: str,       # 用户填写的 API 地址，如 "https://api.openai.com/v1"
    api_key: str,            # 用户填写的 API 密钥
    model_name: str,         # 用户填写的模型名，如 "gpt-4o-mini"
    lang_in: str,            # 源语言代码，如 "en"
    lang_out: str,           # 目标语言代码，如 "zh"
):
    # 第 1 步：创建翻译器实例（每次任务创建新的，因为参数可能不同）
    translator = OpenAITranslator(
        lang_in=lang_in,
        lang_out=lang_out,
        model=model_name,
        base_url=api_base_url,
        api_key=api_key,
    )

    # 第 2 步：设置翻译 API 调用速率限制（每秒请求数）
    set_translate_rate_limiter(4)

    # 第 3 步：创建翻译配置
    config = TranslationConfig(
        input_file=input_pdf_path,          # str 或 pathlib.Path 均可
        translator=translator,
        lang_in=lang_in,
        lang_out=lang_out,
        doc_layout_model=doc_layout_model,  # 用启动时加载的全局模型
        output_dir=output_dir,              # str 或 pathlib.Path 均可
        qps=4,
        watermark_output_mode=WatermarkOutputMode.NoWatermark,
    )

    # 第 4 步：执行异步翻译，遍历进度事件
    async for event in babeldoc.format.pdf.high_level.async_translate(config):
        event_type = event["type"]

        if event_type == "progress_update":
            progress = event["overall_progress"]  # float, 0.0 ~ 100.0
            stage = event["stage"]                # str, 当前阶段名称
            # → 在这里更新任务进度

        elif event_type == "finish":
            result = event["translate_result"]
            # result 是 TranslateResult 对象，包含以下属性：
            #   result.mono_pdf_path  → pathlib.Path 或 None（单语翻译 PDF）
            #   result.dual_pdf_path  → pathlib.Path 或 None（双语对照 PDF）
            # → 在这里记录输出文件路径
            break

        elif event_type == "error":
            error_message = event["error"]  # str
            # → 在这里记录错误信息
            break
```

### 4.4 必须遵守的规则

1. **`DocLayoutModel.load_onnx()` 只能在应用启动时调用一次**，保存为全局变量，所有任务共享。该方法加载一个 ONNX 机器学习模型，耗时长、占内存大，重复加载会导致内存溢出。
2. **`babeldoc.format.pdf.high_level.init()` 只需调用一次**。
3. **`OpenAITranslator` 每次任务创建新实例**，因为不同任务的 API 配置和语言可能不同。
4. **翻译是耗时操作**（一篇论文可能几分钟到十几分钟），绝不能在 HTTP 请求处理函数中同步等待，必须在后台协程中执行。
5. **`async_translate()` 是一个 async generator**，必须在 async 函数中用 `async for` 遍历。
6. **不要自己猜测或编造 BabelDOC 的 API**，上面给出的就是全部需要用到的接口，请严格按照上面的代码调用。

## 五、后端 API 设计（app.py）

### 5.1 应用生命周期

使用 FastAPI 的 lifespan 机制，在启动时完成 BabelDOC 初始化和模型加载：

```python
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时
    babeldoc.format.pdf.high_level.init()
    global doc_layout_model
    doc_layout_model = DocLayoutModel.load_onnx()
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    yield
    # 关闭时（可选清理）

app = FastAPI(lifespan=lifespan)
```

### 5.2 任务状态管理

使用一个全局字典管理所有翻译任务：

```python
import uuid

# key: task_id (str), value: dict
tasks: dict[str, dict] = {}

# 每个任务的结构：
# {
#     "status": "pending" | "running" | "done" | "error",
#     "progress": 0.0,          # 0.0 ~ 100.0
#     "stage": "",               # 当前阶段名称
#     "mono_pdf_path": None,     # pathlib.Path or None
#     "dual_pdf_path": None,     # pathlib.Path or None
#     "error": None,             # str or None
#     "created_at": time.time(),
# }
```

### 5.3 需要实现的 4 个接口

**接口 1：`GET /`**
- 返回 index.html 页面

**接口 2：`POST /api/translate`**
- 请求体：multipart/form-data，包含以下字段：
  - `file`: 上传的 PDF 文件
  - `api_base_url`: str
  - `api_key`: str
  - `model`: str
  - `lang_in`: str
  - `lang_out`: str
- 处理逻辑：
  1. 验证上传文件是 PDF（检查文件名后缀 .pdf）
  2. 生成 task_id（uuid4）
  3. 将文件保存到 `uploads/{task_id}.pdf`
  4. 在 tasks 字典中创建任务记录
  5. 用 `asyncio.create_task()` 启动后台翻译协程
  6. 立即返回 `{"task_id": task_id}`
- 后台翻译协程：
  1. 将任务状态更新为 "running"
  2. 调用第四节中的 `run_translate` 流程
  3. 在 `progress_update` 事件中更新 tasks 字典的 progress 和 stage
  4. 在 `finish` 事件中将状态更新为 "done"，记录输出文件路径
  5. 在 `error` 事件或异常中将状态更新为 "error"，记录错误信息
  6. 整个翻译协程必须用 try/except 包裹，确保任何异常都被捕获并记录到任务中

**接口 3：`GET /api/progress/{task_id}`**
- SSE（Server-Sent Events）接口
- 返回 `EventSourceResponse`（从 `sse_starlette` 导入，需要额外安装 `sse-starlette` 包）
- 每 0.5 秒读取 tasks 字典中该任务的状态，推送给前端：
  - 正在运行时：`data: {"status": "running", "progress": 45.2, "stage": "Translate Paragraphs"}`
  - 翻译完成时：`data: {"status": "done", "progress": 100}`，然后关闭连接
  - 翻译出错时：`data: {"status": "error", "message": "错误信息"}`，然后关闭连接
- 如果 task_id 不存在，立即返回错误事件并关闭

**接口 4：`GET /api/download/{task_id}/{file_type}`**
- `file_type` 取值 `"mono"` 或 `"dual"`
- 从 tasks 字典中找到对应的输出文件路径
- 用 `FileResponse` 返回文件，设置正确的文件名和 Content-Type
- 如果任务不存在、未完成、或对应文件为 None，返回 404

### 5.4 自动清理

- 任务完成或失败 30 分钟后，自动删除 uploads 和 outputs 中的相关文件，并从 tasks 字典中移除
- 可以用一个后台定时任务实现，每 5 分钟扫描一次

## 六、前端页面设计（index.html）

### 6.1 整体布局

页面分为四个区域，纵向排列，居中显示（最大宽度 700px）：

```
┌─────────────────────────────────┐
│  标题: BabelDOC PDF 翻译工具      │
├─────────────────────────────────┤
│  区域 1: API 配置（可折叠）        │
│  ┌─ API Base URL 输入框 ────────┐│
│  ┌─ API Key 输入框（密码类型）──┐│
│  ┌─ 模型名称输入框 ────────────┐│
│  [保存配置] 按钮                  │
├─────────────────────────────────┤
│  区域 2: 文件上传与翻译           │
│  ┌──────────────────────────┐   │
│  │  拖拽 PDF 到此处 / 点击选择 │   │
│  └──────────────────────────┘   │
│  源语言 [下拉框]  目标语言 [下拉框]│
│  [开始翻译] 按钮                  │
├─────────────────────────────────┤
│  区域 3: 进度显示（翻译时显示）    │
│  ██████████░░░░░  45.2%          │
│  当前阶段: Translate Paragraphs  │
│  已用时间: 01:23                  │
├─────────────────────────────────┤
│  区域 4: 下载结果（完成后显示）    │
│  [下载单语 PDF]  [下载双语 PDF]   │
└─────────────────────────────────┘
```

### 6.2 API 配置区域

- 三个输入框：API Base URL、API Key（`type="password"`）、模型名称
- 默认值：模型名称填 `gpt-4o-mini`，其余为空
- 点击"保存配置"时将三个值存入 `localStorage`
- 页面加载时从 `localStorage` 读取并填充
- 该区域默认折叠（如果 localStorage 中已有配置），点击标题可展开/收起

### 6.3 文件上传区域

- 一个拖拽区域（虚线边框），支持拖拽上传和点击选择文件
- 仅接受 `.pdf` 文件（`accept=".pdf"`）
- 上传后显示文件名
- 两个并排的 `<select>` 下拉框用于选择语言，选项列表：
  ```
  en - English
  zh - 中文（简体）
  zh-TW - 中文（繁体）
  ja - 日本語
  ko - 한국어
  fr - Français
  de - Deutsch
  es - Español
  ru - Русский
  pt - Português
  it - Italiano
  ar - العربية
  th - ไทย
  vi - Tiếng Việt
  ```
- 源语言默认选 `en`，目标语言默认选 `zh`
- "开始翻译"按钮：点击后验证 API 配置和文件是否已选择，然后提交

### 6.4 进度区域

- 初始隐藏，点击"开始翻译"后显示
- 进度条：带百分比数字显示
- 当前阶段：显示来自 SSE 的 stage 字符串
- 计时器：从翻译开始后每秒更新，格式 `MM:SS`

### 6.5 结果区域

- 初始隐藏，翻译完成后显示
- 两个下载按钮，点击后触发浏览器下载
- 如果某种 PDF 不存在（后端返回 404），则隐藏对应按钮

### 6.6 前端交互流程

```
用户点击"开始翻译"
  → 前端用 FormData 发送 POST /api/translate（包含文件和配置）
  → 收到 task_id
  → 立即用 EventSource 连接 GET /api/progress/{task_id}
  → 收到 SSE 事件，实时更新进度条和阶段文字
  → 收到 status="done" 时，关闭 SSE，显示下载按钮
  → 收到 status="error" 时，关闭 SSE，显示错误信息
  → 用户点击下载按钮，浏览器打开 /api/download/{task_id}/mono 或 /api/download/{task_id}/dual
```

### 6.7 界面风格

- 简洁现代风格，白色背景（#ffffff），浅灰色卡片背景（#f8f9fa）
- 卡片式布局，圆角（border-radius: 12px），轻微阴影
- 主色调蓝色（#4A90D9），按钮圆角带 hover 变色效果
- 进度条蓝色渐变动画
- 响应式设计，移动端时全宽显示
- 字体使用系统默认字体栈

## 七、错误处理要求

1. 前端：API 配置未填写时，点击翻译要弹出提示
2. 前端：未选择文件时，点击翻译要弹出提示
3. 后端：上传非 PDF 文件时返回 400 错误
4. 后端：翻译过程中 BabelDOC 抛出任何异常，都要捕获并记录到任务中，通过 SSE 推送 error 状态
5. 后端：task_id 不存在时返回 404
6. 翻译进行中时，"开始翻译"按钮禁用，防止重复提交

## 八、启动与运行

```bash
cd web_translate
pip install -r requirements.txt
python app.py
# 或者: uvicorn app:app --host 0.0.0.0 --port 8000
```

在 app.py 最底部添加：
```python
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

浏览器打开 http://localhost:8000 即可使用。
