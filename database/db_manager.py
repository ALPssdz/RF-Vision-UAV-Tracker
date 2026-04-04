import sqlite3
import os
import time
import cv2
from datetime import datetime

class DBManager:
    """
    底层数据封存封装层接口（Data Persistence Adapter）。
    对内处理跨物理文件系统图像矩阵存储和多模态异构融合数据的 SQLite 表单事务。
    该模块严格遵循内聚原则，使得日志、缓存皆固定落位于模块局域存储目录内。
    """
    def __init__(self, db_filename="rf_alert_history.db", img_dirname="alert_images"):
        # 限定文件I/O活动半径在 database 的单一实体语义文件夹内
        self.module_dir = os.path.dirname(os.path.abspath(__file__))
        
        self.db_path = os.path.join(self.module_dir, db_filename)
        self.img_dir = os.path.join(self.module_dir, img_dirname)
        
        if not os.path.exists(self.img_dir):
            os.makedirs(self.img_dir)
            
        self._init_tables()

    def _init_tables(self):
        """ 执行初始化的事件总账数据库物理建表操作。 """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                freq_mhz REAL,
                score REAL,
                image_path TEXT
            )
        ''')
        conn.commit()
        conn.close()

    def log_alert(self, freq_mhz, score, bgr_image):
        """
        向磁盘映射生成的联合模态监控事件序列图，并在关系型数据库内顺次进行实体注册。
        返回本地日志事务流中自增衍生的唯一索引 ID 号。
        """
        now = datetime.now()
        timestamp_str = now.strftime("%Y-%m-%d %H:%M:%S")
        timestamp_file = now.strftime("%Y%m%d_%H%M%S")
        ms = int((time.time() % 1) * 1000)
        
        filename = f"UAV_Intercept_{freq_mhz}MHz_{timestamp_file}_{ms}.jpg"
        absolute_img_path = os.path.join(self.img_dir, filename)
        
        # 使用下层调用栈实施硬编码字节级序列化矩阵
        cv2.imwrite(absolute_img_path, bgr_image)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO alerts (timestamp, freq_mhz, score, image_path)
            VALUES (?, ?, ?, ?)
        ''', (timestamp_str, freq_mhz, score, absolute_img_path))
        
        new_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return new_id

    def get_all_alerts(self):
        """
        提供给前端 View 表现层执行历史日志抽取的访问方法。
        按反时间轴提取记录，返回游标数据的只读链表形式：[(id, timestamp, freq, score, image_path), ...]
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT id, timestamp, freq_mhz, score, image_path FROM alerts ORDER BY id DESC")
        rows = cursor.fetchall()
        conn.close()
        return rows
