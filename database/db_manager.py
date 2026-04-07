import sqlite3
import os
import time
import threading
import cv2
from datetime import datetime


class DBManager:
    """
    告警事件持久化管理模块（Data Persistence Adapter）。

    负责将多模态融合证据图像写入文件系统，并在 SQLite3 关系型数据库中
    维护对应的元数据索引记录。所有文件 I/O 操作均限定于本模块所在目录
    （``database/``）内，确保路径可移植性。

    线程安全设计（修正自 v1.0）：
      ① WAL 模式（Write-Ahead Logging）：允许写操作与读操作真正并发，
         消除后台 hub 线程写入 与 Qt 主线程读取 之间的 "database is locked"。
      ② threading.Lock：串行化写序列（LRU 剪枝 + INSERT），保证原子性。
      ③ 读操作（get_all_alerts）无锁：WAL 模式已保证读写并发安全。
      ④ cv2.imwrite 返回值校验：写入失败时路径标记为 <MISSING>，不存破损记录。
      ⑤ 全方法异常兜底：任何 DB 错误均被捕获回滚，不影响 hub 主循环。

    单实例原则：
      系统中应仅存在一个 DBManager 实例（在 system_hub.CentralHubEngine 中创建）。
      GUI 层通过 hub.db_engine 引用该实例，避免多实例并发冲突。

    存储容量约束：数据库记录上限 MAX_RECORDS=1000 条；超出时触发 LRU 淘汰，
    自动删除最早的 PRUNE_COUNT=100 条记录及其关联图像文件。
    """

    MAX_RECORDS = 1000
    PRUNE_COUNT = 100

    def __init__(self, db_filename: str = "rf_alert_history.db",
                 img_dirname: str = "alert_images"):
        """
        Parameters
        ----------
        db_filename : str — SQLite3 数据库文件名，默认 rf_alert_history.db
        img_dirname : str — 告警图像存储目录名，默认 alert_images
        """
        self.module_dir = os.path.dirname(os.path.abspath(__file__))
        self.db_path    = os.path.join(self.module_dir, db_filename)
        self.img_dir    = os.path.join(self.module_dir, img_dirname)

        os.makedirs(self.img_dir, exist_ok=True)

        # 写操作互斥锁：保护 LRU 剪枝 + INSERT 的原子性
        self._write_lock = threading.Lock()

        self._init_tables()

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        """
        创建并配置数据库连接。

        - check_same_thread=False：允许跨线程复用（写操作由 _write_lock 互斥，
          读操作由 WAL 并发保护）。
        - WAL 模式：读写并发，解决写锁阻塞读事务的问题。
        - synchronous=NORMAL：WAL 下 NORMAL 已足够安全，性能优于 FULL。
        - timeout=10：等待最多 10 秒再抛出 OperationalError。
        """
        conn = sqlite3.connect(self.db_path,
                               check_same_thread=False,
                               timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA cache_size=-8192")   # 8 MiB 页缓存
        return conn

    def _init_tables(self):
        """创建 alerts 数据表（若已存在则跳过），并激活 WAL 模式。"""
        conn = self._connect()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp  TEXT    NOT NULL,
                freq_mhz   REAL    NOT NULL,
                score      REAL    NOT NULL,
                image_path TEXT    NOT NULL
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_alerts_id ON alerts(id)"
        )
        conn.commit()
        conn.close()

    # ------------------------------------------------------------------
    # 公开写接口
    # ------------------------------------------------------------------

    def log_alert(self, freq_mhz: float, score: float, bgr_image) -> int:
        """
        将一次告警事件的融合证据图像持久化至磁盘，并在数据库中注册元数据。

        设计要点（v2.0 修正）：
          1. cv2.imwrite 在锁外执行（不占用 DB 事务时间）；返回值严格校验。
          2. 持有 _write_lock 后，在 **单条数据库连接** 中完成：
               ① LRU 剪枝（超限时批量删除旧记录及文件）
               ② INSERT 新记录
             整个操作在同一事务中提交，保证原子性，消除 TOCTOU 竞争。
          3. 全程 try-except：任何异常捕获后回滚，返回 -1，不影响 hub 循环。

        Returns
        -------
        int — 写入记录的自增 ID；失败时返回 -1
        """
        # ── Step 1: 图像写入磁盘（锁外，避免长 I/O 占用 DB 事务）──────────
        now            = datetime.now()
        timestamp_str  = now.strftime("%Y-%m-%d %H:%M:%S")
        timestamp_file = now.strftime("%Y%m%d_%H%M%S")
        ms             = int((time.time() % 1) * 1000)
        filename          = (f"UAV_Intercept_{freq_mhz:.0f}MHz"
                             f"_{timestamp_file}_{ms:03d}.jpg")
        absolute_img_path = os.path.join(self.img_dir, filename)

        ok = cv2.imwrite(absolute_img_path, bgr_image,
                         [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not ok:
            # imwrite 失败时仍入库，但路径标记提示缺失
            print(f"[DBManager] 警告: cv2.imwrite 写入失败 -> {filename}")
            absolute_img_path = "<MISSING>"

        # ── Step 2: 加锁 → LRU 剪枝 + INSERT（单连接单事务）───────────────
        with self._write_lock:
            conn = None
            try:
                conn   = self._connect()
                cursor = conn.cursor()

                # LRU 剪枝：计数后若超限则批量删除最旧记录
                cursor.execute("SELECT COUNT(*) FROM alerts")
                count = cursor.fetchone()[0]
                if count >= self.MAX_RECORDS:
                    cursor.execute(
                        "SELECT id, image_path FROM alerts "
                        "ORDER BY id ASC LIMIT ?",
                        (self.PRUNE_COUNT,)
                    )
                    old_rows  = cursor.fetchall()
                    ids_to_del = [r[0] for r in old_rows]

                    # 先删文件（文件删除失败不影响 DB 事务）
                    for _, img_path in old_rows:
                        if img_path and img_path != "<MISSING>":
                            try:
                                if os.path.exists(img_path):
                                    os.remove(img_path)
                            except OSError:
                                pass

                    # 批量删除 DB 记录（单条 SQL，比逐行 DELETE 效率高 10x）
                    placeholders = ",".join("?" * len(ids_to_del))
                    cursor.execute(
                        f"DELETE FROM alerts WHERE id IN ({placeholders})",
                        ids_to_del
                    )
                    print(f"[DBManager] LRU 淘汰：删除最早 {len(ids_to_del)} 条记录。")

                # INSERT 新记录
                cursor.execute(
                    "INSERT INTO alerts "
                    "(timestamp, freq_mhz, score, image_path) "
                    "VALUES (?, ?, ?, ?)",
                    (timestamp_str, freq_mhz, score, absolute_img_path)
                )
                new_id = cursor.lastrowid
                conn.commit()
                conn.close()
                return new_id

            except sqlite3.OperationalError as e:
                print(f"[DBManager] DB 操作异常（已回滚）: {e}")
            except Exception as e:
                print(f"[DBManager] 未知异常（已回滚）: {e}")
            finally:
                # 确保连接在异常路径下也被关闭
                if conn:
                    try:
                        conn.rollback()
                        conn.close()
                    except Exception:
                        pass

            return -1

    # ------------------------------------------------------------------
    # 公开读接口（无锁，WAL 保证读写并发）
    # ------------------------------------------------------------------

    def get_all_alerts(self) -> list:
        """
        按时间逆序检索所有告警记录，供 GUI 表现层渲染历史日志列表。

        WAL 模式下，此方法可与 log_alert 真正并发执行（无需等待写锁）。

        Returns
        -------
        list of tuple : [(id, timestamp, freq_mhz, score, image_path), ...]
                        按 id 降序（最新记录在前）
        """
        try:
            conn   = self._connect()
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, timestamp, freq_mhz, score, image_path "
                "FROM alerts ORDER BY id DESC"
            )
            rows = cursor.fetchall()
            conn.close()
            return rows
        except Exception as e:
            print(f"[DBManager] get_all_alerts 异常: {e}")
            return []

    def clear_all(self) -> int:
        """
        清除数据库中全部告警记录及其关联图像文件。

        需持有 _write_lock，防止与 log_alert 并发冲突。

        Returns
        -------
        int — 本次操作删除的记录总条数；失败时返回 0
        """
        with self._write_lock:
            conn = None
            try:
                conn   = self._connect()
                cursor = conn.cursor()

                cursor.execute("SELECT image_path FROM alerts")
                paths         = [row[0] for row in cursor.fetchall()]
                deleted_count = len(paths)

                for img_path in paths:
                    if img_path and img_path != "<MISSING>":
                        try:
                            if os.path.exists(img_path):
                                os.remove(img_path)
                        except OSError:
                            pass

                cursor.execute("DELETE FROM alerts")
                # 重置自增 ID 计数器（sqlite_sequence 可能不存在，忽略错误）
                cursor.execute(
                    "DELETE FROM sqlite_sequence WHERE name='alerts'"
                )
                conn.commit()
                conn.close()
                return deleted_count

            except Exception as e:
                print(f"[DBManager] clear_all 异常: {e}")
                if conn:
                    try:
                        conn.rollback()
                        conn.close()
                    except Exception:
                        pass
                return 0
