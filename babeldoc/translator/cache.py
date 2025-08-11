import json
import logging
import random
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import peewee
from babeldoc.const import CACHE_FOLDER
from peewee import SQL, AutoField, CharField, Model, SqliteDatabase, TextField, fn

logger = logging.getLogger(__name__)

# SQLite 数据库（多线程模式）
db = SqliteDatabase(
    None,
    check_same_thread=False  # 允许多线程访问
)

# 清理配置
CLEAN_PROBABILITY = 0.0001  # 0.01% 概率触发清理
MAX_CACHE_ROWS = 5_000_000  # 保留最新 5,000,000 条记录

# 进程级互斥锁，防止多个线程同时执行清理
_cleanup_lock = threading.Lock()


def retry_on_locked(func):
    """重试装饰器：遇到 database is locked 时指数退避"""

    def wrapper(*args, **kwargs):
        delay = 0.05
        for attempt in range(6):
            try:
                return func(*args, **kwargs)
            except peewee.OperationalError as e:
                if "database is locked" in str(e).lower():
                    logger.debug(f"DB locked, retry {attempt + 1}, delay {delay:.2f}s")
                    time.sleep(delay)
                    delay *= 2
                else:
                    raise
        logger.error("DB locked too long, give up")
        return None

    return wrapper


@contextmanager
def get_db_connection():
    """确保当前线程有独立连接"""
    with db.connection_context():
        yield


class _TranslationCache(Model):
    id = AutoField()
    translate_engine = CharField(max_length=20)
    translate_engine_params = TextField()
    original_text = TextField()
    translation = TextField()

    class Meta:
        database = db
        constraints = [
            SQL(
                """
                UNIQUE (
                    translate_engine,
                    translate_engine_params,
                    original_text
                )
                ON CONFLICT REPLACE
                """,
            ),
        ]


class TranslationCache:
    @staticmethod
    def _sort_dict_recursively(obj):
        if isinstance(obj, dict):
            return {
                k: TranslationCache._sort_dict_recursively(v)
                for k in sorted(obj.keys())
                for v in [obj[k]]
            }
        elif isinstance(obj, list):
            return [TranslationCache._sort_dict_recursively(item) for item in obj]
        return obj

    def __init__(self, translate_engine: str, translate_engine_params: dict = None):
        self.translate_engine = translate_engine
        self.replace_params(translate_engine_params)

    def replace_params(self, params: dict = None):
        if params is None:
            params = {}
        self.params = params
        params = self._sort_dict_recursively(params)
        self.translate_engine_params = json.dumps(params)

    def update_params(self, params: dict = None):
        if params is None:
            params = {}
        self.params.update(params)
        self.replace_params(self.params)

    def add_params(self, k: str, v):
        self.params[k] = v
        self.replace_params(self.params)

    @retry_on_locked
    def get(self, original_text: str) -> str | None:
        with get_db_connection():
            result = _TranslationCache.get_or_none(
                translate_engine=self.translate_engine,
                translate_engine_params=self.translate_engine_params,
                original_text=original_text,
            )
            if result and random.random() < CLEAN_PROBABILITY:
                self._cleanup()
            return result.translation if result else None

    @retry_on_locked
    def set(self, original_text: str, translation: str):
        with get_db_connection():
            _TranslationCache.create(
                translate_engine=self.translate_engine,
                translate_engine_params=self.translate_engine_params,
                original_text=original_text,
                translation=translation,
            )
            if random.random() < CLEAN_PROBABILITY:
                self._cleanup()

    def _cleanup(self) -> None:
        if not _cleanup_lock.acquire(blocking=False):
            return
        try:
            logger.info("Cleaning up translation cache...")
            with get_db_connection():
                max_id = _TranslationCache.select(fn.MAX(_TranslationCache.id)).scalar()
                if not max_id or max_id <= MAX_CACHE_ROWS:
                    return
                threshold = max_id - MAX_CACHE_ROWS
                _TranslationCache.delete().where(
                    _TranslationCache.id <= threshold
                ).execute()
        finally:
            _cleanup_lock.release()


def init_db(remove_exists=False):
    CACHE_FOLDER.mkdir(parents=True, exist_ok=True)
    cache_db_path = CACHE_FOLDER / "cache.v1.db"
    logger.info(f"Initializing cache database at {cache_db_path}")
    if remove_exists and cache_db_path.exists():
        cache_db_path.unlink()
    db.init(
        cache_db_path,
        pragmas={
            "journal_mode": "wal",  # 并发读写
            "synchronous": "normal",  # 提高写性能
            "temp_store": "memory",
            "cache_size": -6400000,  # ~64MB
            "busy_timeout": 10000,  # 提高锁等待时间
        },
        check_same_thread=False
    )
    with get_db_connection():
        db.create_tables([_TranslationCache], safe=True)


def init_test_db():
    import tempfile
    temp_file = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    cache_db_path = temp_file.name
    temp_file.close()

    test_db = SqliteDatabase(
        cache_db_path,
        pragmas={
            "journal_mode": "wal",
            "synchronous": "normal",
            "temp_store": "memory",
            "cache_size": -6400000,
            "busy_timeout": 10000,
        },
        check_same_thread=False
    )
    test_db.bind([_TranslationCache], bind_refs=False, bind_backrefs=False)
    test_db.connect()
    test_db.create_tables([_TranslationCache], safe=True)
    return test_db


def clean_test_db(test_db):
    test_db.drop_tables([_TranslationCache])
    test_db.close()
    db_path = Path(test_db.database)
    if db_path.exists():
        db_path.unlink()
    wal_path = Path(str(db_path) + "-wal")
    if wal_path.exists():
        wal_path.unlink()
    shm_path = Path(str(db_path) + "-shm")
    if shm_path.exists():
        shm_path.unlink()


# 初始化数据库
init_db()
