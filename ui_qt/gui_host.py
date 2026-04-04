import os
import cv2
import numpy as np
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QPlainTextEdit,
                             QTabWidget, QTableWidget, QTableWidgetItem, QHeaderView)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap, QFont
from database.db_manager import DBManager

class MainWindow(QMainWindow):
    """
    基于 PyQt5 构建的纯粹前端表现视图类（Strict View Layer）。
    采用读写隔离沙盒法则，屏蔽硬件数据抓取方法及底层 SQLite 数据源写操作，
    专注于单向数据流的可视化解析缓冲工作。
    """
    def __init__(self, hub=None):
        super().__init__()
        self.setWindowTitle("RF-Vision: 无人机多模态探测系统 (Display View)")
        self.resize(1600, 900)
        
        self.hub = hub
        self.db_engine = DBManager() # 仅用作视图层模型表格只读挂载点
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        top_layout = QVBoxLayout(central_widget)
        top_layout.setContentsMargins(5, 5, 5, 5)
        
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("QTabBar::tab { height: 45px; font-size: 16px; padding: 0 30px; }")
        top_layout.addWidget(self.tabs)
        
        self.tab1 = QWidget()
        self.setup_live_dashboard()
        self.tabs.addTab(self.tab1, "监测数据流 (Real-Time)")
        
        self.tab2 = QWidget()
        self.setup_evidence_database()
        self.tabs.addTab(self.tab2, "告警日志库 (Database Records)")
        
        self.tabs.currentChanged.connect(self.on_tab_changed)
        
        if self.hub:
            # 建立系统前端总线事件信号槽联结节点
            self.hub.signal_rf_frame.connect(self.update_rf_frame)
            self.hub.signal_k230_frame.connect(self.update_k230_frame)
            self.hub.signal_log.connect(self.append_log)
            self.hub.signal_system_status.connect(self.update_status_labels)
            self.hub.signal_db_updated.connect(self.load_db_data)  # 异步模型同步触发器

    def setup_live_dashboard(self):
        main_layout = QVBoxLayout(self.tab1)
        
        status_banner = QHBoxLayout()
        self.lbl_rf_status = QLabel("SDR 节点: 🟢 就绪 (休眠)")
        self.lbl_k230_status = QLabel("视频节点: 🟢 就绪 (休眠)")
        self.lbl_fusion_status = QLabel("系统模式: 🟡 待机总控指令")
        for lbl in [self.lbl_rf_status, self.lbl_k230_status, self.lbl_fusion_status]:
            lbl.setFont(QFont("Microsoft YaHei", 12, QFont.Bold))
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet("background-color: #2c3e50; color: white; padding: 10px; border-radius: 5px;")
            status_banner.addWidget(lbl)
            
        main_layout.addLayout(status_banner)
        
        battle_layout = QHBoxLayout()
        
        rf_group = QVBoxLayout()
        rf_title = QLabel("SDR 射频推断反馈 (瀑布图)")
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
        k230_title = QLabel("K230 边缘节点推流图像")
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
        self.btn_play = QPushButton("▶ 启动数据采集进程")
        self.btn_play.setMinimumHeight(60)
        self.btn_play.setStyleSheet("background-color: #27ae60; color: white; border-radius: 5px; font-weight: bold; font-size: 14px;")
        self.btn_play.clicked.connect(self.toggle_play)
        
        self.btn_mock_optical = QPushButton("🔧 [开发工具] 发送系统调试信令 (JSON)")
        self.btn_mock_optical.setMinimumHeight(40)
        self.btn_mock_optical.setStyleSheet("background-color: #f39c12; color: white; border-radius: 5px;")
        self.btn_mock_optical.pressed.connect(lambda: self.hub.mock_k230_trigger(True) if self.hub else None)
        self.btn_mock_optical.released.connect(lambda: self.hub.mock_k230_trigger(False) if self.hub else None)
        
        self.btn_exit = QPushButton("⏏ 安全终止进程组")
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
        self.db_table.setHorizontalHeaderLabels(["ID 号", "系统触发时间", "信号参考量", "置信度"])
        self.db_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.db_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.db_table.setSelectionMode(QTableWidget.SingleSelection)
        self.db_table.itemSelectionChanged.connect(self.on_db_row_selected)
        
        self.current_db_paths = [] 
        
        self.db_img_label = QLabel("待命：正在等待实体行列选择事件...");
        self.db_img_label.setFixedSize(1400, 600)  
        self.db_img_label.setStyleSheet("background-color: #1e1e1e; color: #888; font-size: 16px; border: 2px dashed #666;")
        self.db_img_label.setAlignment(Qt.AlignCenter)
        
        layout.addWidget(self.db_table, stretch=1)
        layout.addWidget(self.db_img_label, stretch=3)

    # =============== 视图渲染回调集合 ===============
    def update_rf_frame(self, frame):
        self.render_cv2_to_qlabel(frame, self.img_rf)

    def update_k230_frame(self, frame):
        self.render_cv2_to_qlabel(frame, self.img_k230)

    def update_status_labels(self, status_dict):
        """ 解析中央控制器下发的指令更新状态信标 """
        if "sdr" in status_dict:
            self.lbl_rf_status.setText(status_dict["sdr"])
        if "vision" in status_dict:
            self.lbl_k230_status.setText(status_dict["vision"])
        if "system" in status_dict:
            self.lbl_fusion_status.setText(status_dict["system"])
        if "color" in status_dict:
            self.lbl_fusion_status.setStyleSheet(f"background-color: {status_dict['color']}; color: white;")

    def append_log(self, text):
        self.log_textbox.appendPlainText(text)
        self.log_textbox.verticalScrollBar().setValue(self.log_textbox.verticalScrollBar().maximum())

    def toggle_play(self):
        if not self.hub: return
        if self.hub.running:
            self.hub.stop_sensing()
            self.btn_play.setText("▶ 呼叫总线唤醒采流任务")
            self.btn_play.setStyleSheet("background-color: #27ae60; color: white;")
        else:
            self.hub.start_sensing()
            self.btn_play.setText("⏸ 安全关停采集管道")
            self.btn_play.setStyleSheet("background-color: #c0392b; color: white;")

    def on_tab_changed(self, index):
        if index == 1:
            self.load_db_data()

    def load_db_data(self):
        # Read-only 模型操作调用：挂载历史证据数据条目。
        rows = self.db_engine.get_all_alerts()
        self.db_table.setRowCount(len(rows))
        self.current_db_paths = []
        for row_idx, data in enumerate(rows):
            self.db_table.setItem(row_idx, 0, QTableWidgetItem(f"REC-{data[0]}"))
            self.db_table.setItem(row_idx, 1, QTableWidgetItem(str(data[1])))
            self.db_table.setItem(row_idx, 2, QTableWidgetItem(f"{data[2]} MHz"))
            self.db_table.setItem(row_idx, 3, QTableWidgetItem(f"{data[3] * 100:.2f} %"))
            self.current_db_paths.append(data[4])
            
        if len(rows) > 0 and self.db_img_label.text() == "待命：正在等待实体行列选择事件...":
            self.db_img_label.setText("数据装载执行完毕。请激活目标记录以调用底层视觉文件。")

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
            self.db_img_label.setText("I/O 系统级错误：本地文件检索寻址失败。")

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
