@echo off
cd /d %~dp0

if exist ".venv\" (
    echo [BabelDOC] 正在激活Python虚拟环境...
    call .venv\Scripts\activate.bat
    echo 虚拟环境已激活，输入命令开始使用
    echo 输入 exit 退出
    cmd /k "echo 当前工作目录: %cd% && title BabelDOC环境"
) else (
    echo [BabelDOC] 正在初始化环境...
    uv run babeldoc --help
    python -m venv .venv
    echo [BabelDOC] 正在激活新建的虚拟环境...
    call .venv\Scripts\activate.bat
    cmd /k "echo 环境已准备就绪！输入命令开始使用 && title BabelDOC环境"
)
