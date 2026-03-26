import itertools
import multiprocessing as mp
import os
import shutil
import subprocess
import threading
from pathlib import Path

__version__ = "0.5.24"

CACHE_FOLDER = Path.home() / ".cache" / "babeldoc"


def get_cache_file_path(filename: str, sub_folder: str | None = None) -> Path:
    if sub_folder is not None:
        sub_folder = sub_folder.strip("/")
        sub_folder_path = CACHE_FOLDER / sub_folder
        sub_folder_path.mkdir(parents=True, exist_ok=True)
        return sub_folder_path / filename
    return CACHE_FOLDER / filename


try:
    git_path = shutil.which("git")
    if git_path is None:
        raise FileNotFoundError("git executable not found")
    two_parent = Path(__file__).resolve().parent.parent
    md_ = two_parent / "docs" / "README.md"
    if two_parent.name == "site-packages" or not md_.exists():
        raise FileNotFoundError("not in git repo")
    WATERMARK_VERSION = (
        subprocess.check_output(  # noqa: S603
            [git_path, "describe", "--always"],
            cwd=Path(__file__).resolve().parent,
        )
        .strip()
        .decode()
    )
except (OSError, FileNotFoundError, subprocess.CalledProcessError):
    WATERMARK_VERSION = f"v{__version__}"

TIKTOKEN_CACHE_FOLDER = CACHE_FOLDER / "tiktoken"
TIKTOKEN_CACHE_FOLDER.mkdir(parents=True, exist_ok=True)
os.environ["TIKTOKEN_CACHE_DIR"] = str(TIKTOKEN_CACHE_FOLDER)


_process_pool = None
_process_pool_lock = threading.Lock()
_ENABLE_PROCESS_POOL = False


def enable_process_pool():
    # Development and Testing ONLY API
    global _ENABLE_PROCESS_POOL
    _ENABLE_PROCESS_POOL = True


# macos & windows use spawn mode
# linux use forkserver mode


def get_process_pool():
    if not _ENABLE_PROCESS_POOL:
        return None
    global _process_pool
    with _process_pool_lock:
        if _process_pool is None:
            # Create pool only in main process
            if mp.current_process().name != "MainProcess":
                return None

            _process_pool = mp.Pool()
        return _process_pool


def close_process_pool():
    if not _ENABLE_PROCESS_POOL:
        return None
    global _process_pool
    with _process_pool_lock:
        if _process_pool:
            _process_pool.close()
            _process_pool.join()
            _process_pool = None


def batched(iterable, n, *, strict=False):
    # batched('ABCDEFG', 3) → ABC DEF G
    if n < 1:
        raise ValueError("n must be at least one")
    iterator = iter(iterable)
    while batch := tuple(itertools.islice(iterator, n)):
        if strict and len(batch) != n:
            raise ValueError("batched(): incomplete batch")
        yield batch
