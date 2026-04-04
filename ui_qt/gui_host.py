import torch 
import sys
import os

# -------- [环境路径设置] --------
PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

import time
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QPlainTextEdit,
                             QTabWidget, QTableWidget, QTableWidgetItem, QHeaderView, QFrame)
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QFont
from backend_rk3588.main_rf_pipeline import RFToolchain
from database.db_manager import DBManager

# ==========================================
# Worker 1: 射频底盘雷达 (ZYNQ/SDR)
# ==========================================
class RFPipelineWorker(QThread):
    frame_ready = pyqtSignal(np.ndarray)
    log_ready = pyqtSignal(str)
    alert_triggered = pyqtSignal(dict, np.ndarray)  

    def __init__(self):
        super().__init__()
        self.toolchain = RFToolchain()
        self.running = False
        self.step_once = False
        self._killed = False
        
    def run(self):
        self.log_ready.emit("✅ 初始化完成：软件定义无线电 (SDR) 数据链路与存储模块已就绪。")
        while not self._killed:
            if self.running or self.step_once:
                try:
                    frame, log_str, alert_flag, alert_info = self.toolchain.tick()
                    self.frame_ready.emit(frame)
                    self.log_ready.emit(log_str)
                    
                    if alert_flag:
                        self.alert_triggered.emit(alert_info, frame)
                        
                except Exception as e:
                    import traceback
                    self.log_ready.emit(f"❌ 运行时错误 (Runtime Exception): {e}\n{traceback.format_exc()}")
                    self.running = False
                    
                if self.step_once:
                    self.step_once = False
                    self.running = False
            else:
                time.sleep(0.05)

    def kill(self):
        self._killed = True
        self.running = False
        self.wait(1000)
        if self.isRunning():
            self.terminate()

# ==========================================
# Worker 2: K230 边缘视觉检测系统
# ==========================================
class K230VisionWorker(QThread):
    frame_ready = pyqtSignal(np.ndarray)
    log_ready = pyqtSignal(str)
    alert_triggered = pyqtSignal(np.ndarray)

    def __init__(self, stream_url="rtsp://192.168.31.250/stream"):
        super().__init__()
        self.stream_url = stream_url
        self.running = False
        self._killed = False
        self.mock_drone_detected = False 

    def run(self):
        self.log_ready.emit("✅ 视觉模块监听已就绪。正在尝试链接 K230 视频推流...")
        cap = cv2.VideoCapture(self.stream_url)
        
        while not self._killed:
            if self.running:
                ret, frame = cap.read()
                if not ret:
                    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
                    cv2.putText(frame, "K230 RTSP STREAM STANDBY", (460, 540), cv2.FONT_HERSHEY_SIMPLEX, 2, (100, 100, 200), 4)
                    time.sleep(0.1) 
                
                # ------ K230 光电检测演示逻辑 ------
                if self.mock_drone_detected:
                    cv2.rectangle(frame, (800, 400), (1100, 600), (0, 0, 255), 8)
                    cv2.putText(frame, "TARGET LOCKED: UAV", (800, 380), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
                    self.alert_triggered.emit(frame)
                else:
                    cv2.putText(frame, "OPTICAL SENSING...", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                self.frame_ready.emit(frame)
            else:
                time.sleep(0.05)
                
        cap.release()

    def kill(self):
        self._killed = True
        self.running = False
        self.wait(1000)
        if self.isRunning():
            self.terminate()

# ==========================================
# 上位机主窗口
# ==========================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RF-Vision: 多模态无人机监测系统 (1080P 自适应)")
        self.resize(1600, 900)
        
        self.db_engine = DBManager()
        
        self.last_rf_frame = np.zeros((640, 640, 3), dtype=np.uint8)
        self.last_k230_frame = np.zeros((640, 1137, 3), dtype=np.uint8) 
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        top_layout = QVBoxLayout(central_widget)
        top_layout.setContentsMargins(5, 5, 5, 5)
        
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("QTabBar::tab { height: 45px; font-size: 16px; padding: 0 30px; }")
        top_layout.addWidget(self.tabs)
        
        self.tab1 = QWidget()
        self.setup_live_dashboard()
        self.tabs.addTab(self.tab1, "📡 实时监控台")
        
        self.tab2 = QWidget()
        self.setup_evidence_database()
        self.tabs.addTab(self.tab2, "🗄️ 历史告警数据库")
        
        self.tabs.currentChanged.connect(self.on_tab_changed)
        
        self.rf_worker = RFPipelineWorker()
        self.rf_worker.frame_ready.connect(self.update_rf_frame)
        self.rf_worker.log_ready.connect(self.append_log)
        self.rf_worker.alert_triggered.connect(self.on_rf_alert)
        
        self.k230_worker = K230VisionWorker()
        self.k230_worker.frame_ready.connect(self.update_k230_frame)
        self.k230_worker.log_ready.connect(self.append_log)
        self.k230_worker.alert_triggered.connect(self.on_optical_alert)
        
        self.rf_worker.start()
        self.k230_worker.start()

    def setup_live_dashboard(self):
        main_layout = QVBoxLayout(self.tab1)
        
        status_banner = QHBoxLayout()
        self.lbl_rf_status = QLabel("射频单元 (SDR): 🟢 就绪")
        self.lbl_k230_status = QLabel("视觉单元 (K230): 🟢 就绪")
        self.lbl_fusion_status = QLabel("系统状态: 🟢 未发现异常信号")
        for lbl in [self.lbl_rf_status, self.lbl_k230_status, self.lbl_fusion_status]:
            lbl.setFont(QFont("Microsoft YaHei", 12, QFont.Bold))
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet("background-color: #2c3e50; color: white; padding: 10px; border-radius: 5px;")
            status_banner.addWidget(lbl)
            
        main_layout.addLayout(status_banner)
        
        battle_layout = QHBoxLayout()
        
        rf_group = QVBoxLayout()
        rf_title = QLabel("SDR 射频多流向瀑布图")
        rf_title.setFont(QFont("Microsoft YaHei", 14, QFont.Bold))
        rf_title.setStyleSheet("color: #3498db;")
        rf_title.setAlignment(Qt.AlignCenter)
        
        self.img_rf = QLabel()
        self.img_rf.setFixedSize(640, 640)
        self.img_rf.setStyleSheet("background-color: #000; border: 2px solid #ccc;")
        self.img_rf.setAlignment(Qt.AlignCenter)
        rf_group.addWidget(rf_title)
        rf_group.addWidget(self.img_rf)
        
        k230_group = QVBoxLayout()
        k230_title = QLabel("K230 光学补偿成像")
        k230_title.setFont(QFont("Microsoft YaHei", 14, QFont.Bold))
        k230_title.setStyleSheet("color: #e67e22;")
        k230_title.setAlignment(Qt.AlignCenter)
        
        self.img_k230 = QLabel()
        self.img_k230.setFixedSize(1137, 640)
        self.img_k230.setStyleSheet("background-color: #000; border: 2px solid #ccc;")
        self.img_k230.setAlignment(Qt.AlignCenter)
        k230_group.addWidget(k230_title)
        k230_group.addWidget(self.img_k230)
        
        battle_layout.addLayout(rf_group)
        battle_layout.addLayout(k230_group)
        
        main_layout.addLayout(battle_layout, stretch=1)
        
        bottom_layout = QHBoxLayout()
        
        btn_layout = QVBoxLayout()
        self.btn_play = QPushButton("▶ 启动多模态联合监测")
        self.btn_play.setMinimumHeight(60)
        self.btn_play.setStyleSheet("background-color: #27ae60; color: white; border-radius: 5px; font-weight: bold; font-size: 14px;")
        self.btn_play.clicked.connect(self.toggle_play)
        
        self.btn_mock_optical = QPushButton("💻 (逻辑验证) 拟合生成摄像头视觉告警")
        self.btn_mock_optical.setMinimumHeight(40)
        self.btn_mock_optical.setStyleSheet("background-color: #f39c12; color: white; border-radius: 5px;")
        self.btn_mock_optical.pressed.connect(lambda: setattr(self.k230_worker, 'mock_drone_detected', True))
        self.btn_mock_optical.released.connect(lambda: setattr(self.k230_worker, 'mock_drone_detected', False))
        
        self.btn_exit = QPushButton("⏏ 安全退出系统")
        self.btn_exit.setMinimumHeight(40)
        self.btn_exit.setStyleSheet("background-color: #34495e; color: white; border-radius: 5px;")
        self.btn_exit.clicked.connect(self.close)
        
        btn_layout.addWidget(self.btn_play)
        btn_layout.addWidget(self.btn_mock_optical)
        btn_layout.addWidget(self.btn_exit)
        
        self.log_textbox = QPlainTextEdit()
        self.log_textbox.setReadOnly(True)
        self.log_textbox.setStyleSheet("background-color: #1a1a1a; color: #00ff00; font-family: Consolas; font-size: 13px;")
        
        bottom_layout.addLayout(btn_layout, stretch=1)
        bottom_layout.addWidget(self.log_textbox, stretch=3)
        
        main_layout.addLayout(bottom_layout)

    def setup_evidence_database(self):
        layout = QHBoxLayout(self.tab2)
        
        self.db_table = QTableWidget()
        self.db_table.setColumnCount(4)
        self.db_table.setHorizontalHeaderLabels(["记录编号 (ID)", "入库时间", "截获频段", "置信水平"])
        self.db_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.db_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.db_table.setSelectionMode(QTableWidget.SingleSelection)
        self.db_table.itemSelectionChanged.connect(self.on_db_row_selected)
        
        self.current_db_paths = [] 
        
        self.db_img_label = QLabel("正在待命...");
        self.db_img_label.setFixedSize(1400, 600)  
        self.db_img_label.setStyleSheet("background-color: #1e1e1e; color: #888; font-size: 16px; border: 2px dashed #666;")
        self.db_img_label.setAlignment(Qt.AlignCenter)
        
        layout.addWidget(self.db_table, stretch=1)
        layout.addWidget(self.db_img_label, stretch=3)

    def update_rf_frame(self, frame):
        self.last_rf_frame = frame
        self.render_cv2_to_qlabel(frame, self.img_rf)

    def update_k230_frame(self, frame):
        h, w = frame.shape[:2]
        new_w = int((640 / h) * w)
        frame_aligned = cv2.resize(frame, (new_w, 640))
        self.last_k230_frame = frame_aligned
        self.render_cv2_to_qlabel(frame_aligned, self.img_k230)

    def trigger_fused_storage(self, cause_text):
        fused_evidence = np.hstack([self.last_rf_frame, self.last_k230_frame])
        cv2.putText(fused_evidence, f"ALARM REASON: {cause_text}", (20, 600), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        
        new_id = self.db_engine.log_alert(999.0, 1.0, fused_evidence)
        self.append_log(f"💾 数据已归档: 异常快照已分配索引 ID-[{new_id}]。")

    def on_rf_alert(self, alert_info, frame):
        self.lbl_fusion_status.setText("系统告警: 🔴 SDR 拾取到无人机通信特征")
        self.lbl_fusion_status.setStyleSheet("background-color: #e74c3c; color: white;")
        self.trigger_fused_storage("SDR_RF_TRIGGER")
        
    def on_optical_alert(self, frame):
        self.lbl_fusion_status.setText("系统告警: 🔴 光电设备发现上空目标个体")
        self.lbl_fusion_status.setStyleSheet("background-color: #e74c3c; color: white;")
        self.append_log("⚠️ [视觉反馈]: 未授权飞行物已进入视频监控视区。")
        self.trigger_fused_storage("OPTICAL_VIDEO_TRIGGER")

    def append_log(self, text):
        self.log_textbox.appendPlainText(text)
        self.log_textbox.verticalScrollBar().setValue(self.log_textbox.verticalScrollBar().maximum())

    def toggle_play(self):
        if self.rf_worker.running:
            self.rf_worker.running = False
            self.k230_worker.running = False
            self.btn_play.setText("▶ 启动多模态联合监测")
            self.btn_play.setStyleSheet("background-color: #27ae60; color: white;")
            self.lbl_fusion_status.setText("系统状态: 🟡 监测已挂起")
            self.lbl_fusion_status.setStyleSheet("background-color: #f1c40f; color: black;")
        else:
            self.rf_worker.running = True
            self.k230_worker.running = True
            self.btn_play.setText("⏸ 停止采集并挂起系统")
            self.btn_play.setStyleSheet("background-color: #c0392b; color: white;")
            self.lbl_fusion_status.setText("系统状态: 🟢 监测流水线运转中...")
            self.lbl_fusion_status.setStyleSheet("background-color: #27ae60; color: white;")

    def on_tab_changed(self, index):
        if index == 1:
            self.load_db_data()

    def load_db_data(self):
        rows = self.db_engine.get_all_alerts()
        self.db_table.setRowCount(len(rows))
        self.current_db_paths = []
        for row_idx, data in enumerate(rows):
            self.db_table.setItem(row_idx, 0, QTableWidgetItem(f"ID-{data[0]}"))
            self.db_table.setItem(row_idx, 1, QTableWidgetItem(str(data[1])))
            self.db_table.setItem(row_idx, 2, QTableWidgetItem(f"{data[2]} MHz"))
            self.db_table.setItem(row_idx, 3, QTableWidgetItem(f"{data[3] * 100:.2f} %"))
            self.current_db_paths.append(data[4])
        self.db_img_label.clear()
        self.db_img_label.setText("数据加载完毕。请单击左侧单号进行快照回溯。")

    def on_db_row_selected(self):
        selected_items = self.db_table.selectedItems()
        if not selected_items: return
        row = selected_items[0].row()
        img_path = self.current_db_paths[row]
        cv_img = cv2.imread(img_path)
        if cv_img is not None:
            h, w = cv_img.shape[:2]
            scaled = cv2.resize(cv_img, (1400, int(1400/w*h)))
            self.render_cv2_to_qlabel(scaled, self.db_img_label)
        else:
            self.db_img_label.setText("❌ 源图片文件未找到，记录可能已损坏。")

    def render_cv2_to_qlabel(self, cv_img, qlabel):
        qlabel.clear()
        if len(cv_img.shape) == 3:
            h, w, ch = cv_img.shape
            bytes_per_line = ch * w
            rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            qt_img = QImage(rgb_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
        else:
            h, w = cv_img.shape
            bytes_per_line = w
            qt_img = QImage(cv_img.data, w, h, bytes_per_line, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(qt_img)
        qlabel.setPixmap(pixmap)

    def closeEvent(self, event):
        self.rf_worker.kill()
        self.k230_worker.kill()
        super().closeEvent(event)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
