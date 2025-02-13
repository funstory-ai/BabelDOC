@echo off
cd /d %~dp0

if exist ".venv\" (
    echo [BabelDOC] 正在激活Python虚拟环境...
    call .venv\Scripts\activate.bat
    echo 虚拟环境已激活，输入命令开始使用
    echo 输入 exit 退出
    cmd /k "echo 当前工作目录: %cd% && title BabelDOC环境"
) else (
    echo 错误: 虚拟环境不存在于：
    echo %cd%\.venv
    echo 请先执行以下命令创建：
    echo python -m venv .venv
    pause
)