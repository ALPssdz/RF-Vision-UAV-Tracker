import sqlite3
import os
import time
import cv2
from datetime import datetime


class DBManager:
    """
    告警事件持久化管理模块（Data Persistence Adapter）。

    负责将多模态融合证据图像写入文件系统，并在 SQLite3 关系型数据库中
    维护对应的元数据索引记录。所有文件 I/O 操作均限定于本模块所在目录
    （``database/``）内，确保路径可移植性。

    存储容量约束：数据库记录上限为 1000 条；超出时触发 LRU 淘汰策略，
    自动删除最早的 100 条记录及其关联图像文件。
    """

    def __init__(self, db_filename: str = "rf_alert_history.db",
                 img_dirname: str = "alert_images"):
        """
        Parameters
        ----------
        db_filename : str
            SQLite3 数据库文件名，默认为 ``rf_alert_history.db``。
        img_dirname : str
            告警图像存储目录名，默认为 ``alert_images``。
        """
        self.module_dir = os.path.dirname(os.path.abspath(__file__))
        self.db_path    = os.path.join(self.module_dir, db_filename)
        self.img_dir    = os.path.join(self.module_dir, img_dirname)

        os.makedirs(self.img_dir, exist_ok=True)
        self._init_tables()

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _init_tables(self):
        """创建 alerts 数据表（若已存在则跳过）。"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS alerts (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp  TEXT,
                freq_mhz   REAL,
                score      REAL,
                image_path TEXT
            )
        ''')
        conn.commit()
        conn.close()

    def _manage_storage(self):
        """
        LRU 存储容量管理：确保数据库记录不超过预设上限。

        当记录总数超过 ``max_records``（1000 条）时，按插入顺序删除最旧的
        ``prune_count``（100 条）记录及其关联的图像文件。
        """
        max_records = 1000
        prune_count = 100

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM alerts")
        count = cursor.fetchone()[0]

        if count > max_records:
            cursor.execute(
                "SELECT id, image_path FROM alerts ORDER BY id ASC LIMIT ?",
                (prune_count,)
            )
            old_rows = cursor.fetchall()

            for row_id, img_path in old_rows:
                if os.path.exists(img_path):
                    try:
                        os.remove(img_path)
                    except Exception:
                        pass
                cursor.execute("DELETE FROM alerts WHERE id=?", (row_id,))

            conn.commit()
            print(f"[DBManager] LRU 淘汰触发：已删除最早的 {prune_count} 条记录及关联图像。")

        conn.close()

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def log_alert(self, freq_mhz: float, score: float,
                  bgr_image: "np.ndarray") -> int:
        """
        将一次告警事件的融合证据图像持久化至磁盘，并在数据库中注册元数据。

        Parameters
        ----------
        freq_mhz : float
            触发告警的频率（MHz）。
        score : float
            综合置信度评分（归一化至 [0, 1]）。
        bgr_image : numpy.ndarray
            BGR 格式的融合证据图像矩阵，由 ``system_hub`` 拼接生成。

        Returns
        -------
        int
            本次写入记录在数据库中自增的唯一 ID。
        """
        self._manage_storage()

        now            = datetime.now()
        timestamp_str  = now.strftime("%Y-%m-%d %H:%M:%S")
        timestamp_file = now.strftime("%Y%m%d_%H%M%S")
        ms             = int((time.time() % 1) * 1000)

        filename          = f"UAV_Intercept_{freq_mhz}MHz_{timestamp_file}_{ms}.jpg"
        absolute_img_path = os.path.join(self.img_dir, filename)

        # 将融合证据图像以 JPEG 格式编码写入文件系统
        cv2.imwrite(absolute_img_path, bgr_image)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO alerts (timestamp, freq_mhz, score, image_path) VALUES (?, ?, ?, ?)",
            (timestamp_str, freq_mhz, score, absolute_img_path)
        )
        new_id = cursor.lastrowid
        conn.commit()
        conn.close()

        return new_id

    def get_all_alerts(self) -> list:
        """
        按时间逆序检索所有告警记录，供 GUI 表现层渲染历史日志列表。

        Returns
        -------
        list of tuple
            格式为 [(id, timestamp, freq_mhz, score, image_path), ...]，
            按 id 降序排列（最新记录在前）。
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, timestamp, freq_mhz, score, image_path FROM alerts ORDER BY id DESC"
        )
        rows = cursor.fetchall()
        conn.close()
        return rows

    def clear_all(self) -> int:
        """
        清除数据库中全部告警记录及其关联图像文件。

        操作步骤：
          1. 查询所有记录的图像路径并逐一删除对应文件；
          2. 清空 alerts 数据表（保留表结构）；
          3. 重置自增 ID 计数器，使后续记录从 1 开始编号。

        Returns
        -------
        int
            本次操作删除的记录总条数。
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT image_path FROM alerts")
        paths         = [row[0] for row in cursor.fetchall()]
        deleted_count = len(paths)

        for img_path in paths:
            try:
                if os.path.exists(img_path):
                    os.remove(img_path)
            except Exception:
                pass

        cursor.execute("DELETE FROM alerts")
        cursor.execute("DELETE FROM sqlite_sequence WHERE name='alerts'")
        conn.commit()
        conn.close()

        return deleted_count
