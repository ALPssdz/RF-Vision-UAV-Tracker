import sqlite3
import os
import time
import cv2
from datetime import datetime

class DBManager:
    """
    Data Persistence Adapter.
    Handles physical file I/O operations and SQLite table management for the synthesized multi-modal evidential records.
    All runtime databases and image caches are maintained within this localized module directory to ensure encapsulation.
    """
    def __init__(self, db_filename="rf_alert_history.db", img_dirname="alert_images"):
        # Localize the storage paths strictly within the 'database' semantic folder
        self.module_dir = os.path.dirname(os.path.abspath(__file__))
        
        self.db_path = os.path.join(self.module_dir, db_filename)
        self.img_dir = os.path.join(self.module_dir, img_dirname)
        
        if not os.path.exists(self.img_dir):
            os.makedirs(self.img_dir)
            
        self._init_tables()

    def _init_tables(self):
        """ Initializes the database relations and schemas for sequential event logs. """
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
        Commits a generated multi-modal event snapshot to the file system and registers its metadata sequentially.
        Returns the auto-incremented primary key of the new transaction log.
        """
        now = datetime.now()
        timestamp_str = now.strftime("%Y-%m-%d %H:%M:%S")
        timestamp_file = now.strftime("%Y%m%d_%H%M%S")
        ms = int((time.time() % 1) * 1000)
        
        filename = f"UAV_Intercept_{freq_mhz}MHz_{timestamp_file}_{ms}.jpg"
        absolute_img_path = os.path.join(self.img_dir, filename)
        
        # Serialize the matrix to disk
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
        Extracts all historical alert rows ordered in descending chronology.
        Provides a data population feed for decoupled presentation entities (View Layer).
        Returns a list of tuples: [(id, timestamp, freq, score, image_path), ...]
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT id, timestamp, freq_mhz, score, image_path FROM alerts ORDER BY id DESC")
        rows = cursor.fetchall()
        conn.close()
        return rows
