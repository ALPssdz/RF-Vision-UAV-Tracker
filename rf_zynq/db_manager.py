import sqlite3
import os
import time
import cv2
from datetime import datetime

class DBManager:
    def __init__(self, db_filename="rf_alert_history.db", img_dirname="alert_images"):
        # 获取当前模块同一级物理路径
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.db_path = os.path.join(self.base_dir, db_filename)
        self.img_dir = os.path.join(self.base_dir, img_dirname)
        
        # 不存在照片文件夹则创建
        if not os.path.exists(self.img_dir):
            os.makedirs(self.img_dir)
            
        self._init_tables()

    def _init_tables(self):
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
        物理落盘：将传来的彩图硬写到硬盘，并把记录推入数据库。
        返回生成的那条库记录的自增 ID。
        """
        now = datetime.now()
        # 给人看的时间
        timestamp_str = now.strftime("%Y-%m-%d %H:%M:%S")
        # 拼接文件名里的时间
        timestamp_file = now.strftime("%Y%m%d_%H%M%S")
        ms = int((time.time() % 1) * 1000)
        
        filename = f"UAV_Intercept_{freq_mhz}MHz_{timestamp_file}_{ms}.jpg"
        absolute_img_path = os.path.join(self.img_dir, filename)
        
        # 用 OpenCV 暴力写入 JPG 存根
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
        供 GUI 无脑获取所有历史数据的接口。按时间倒序（最新的在上头）。
        返回列表[(id, timestamp, freq, score, image_path), ...]
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT id, timestamp, freq_mhz, score, image_path FROM alerts ORDER BY id DESC")
        rows = cursor.fetchall()
        conn.close()
        return rows
