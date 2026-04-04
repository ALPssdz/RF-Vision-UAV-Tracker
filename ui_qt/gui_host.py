import torch # 在第一行极其强势地霸占 C++ DLL 底层加载入口！
import sys
import os

# -------- [重大环境修补 / 环境路径挂载] --------
# 强行抢占将根目录 rf_zynq 压入系统级搜寻链路，如此才能实现跨包拉取依赖物。
PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)
# ---------------------------------------------

import time
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QPlainTextEdit,
                             QTabWidget, QTableWidget, QTableWidgetItem, QHeaderView)
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QImage, QPixmap
from backend_rk3588.main_rf_pipeline import RFToolchain
from database.db_manager import DBManager

class RFPipelineWorker(QThread):
    frame_ready = pyqtSignal(np.ndarray)
    log_ready = pyqtSignal(str)
    # 当检测到且确认是无人机时发射，夹带极高价值告警字典供 UI 落盘
    alert_triggered = pyqtSignal(dict, np.ndarray) 
    
    def __init__(self, toolchain):
        super().__init__()
        self.toolchain = toolchain
        self.running = False
        self.step_once = False
        self._killed = False
        
    def run(self):
        self.log_ready.emit("✅ 系统初始化完成，软件定义无线电 (SDR) 线程与数据库读写通道已正常待命。")
        while not self._killed:
            if self.running or self.step_once:
                try:
                    frame, log_str, alert_flag, alert_info = self.toolchain.tick()
                    
                    self.frame_ready.emit(frame)
                    self.log_ready.emit(log_str)
                    
                    # 侦破逻辑落盘触发器！一旦 S3 核准通过，立刻丢给硬盘固化！
                    if alert_flag:
                        self.alert_triggered.emit(alert_info, frame)
                        
                except Exception as e:
                    import traceback
                    self.log_ready.emit(f"❌ 运行期线程异常 (Runtime Exception): {e}\n{traceback.format_exc()}")
                    self.running = False
                    
                if self.step_once:
                    self.step_once = False
            else:
                time.sleep(0.05)

    def kill(self):
        self._killed = True

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RF-Vision: 分布式射频态势感知与特征溯源数据库")
        self.resize(1150, 750)
        
        # 挂载极其纯洁轻量的关系型存根数据库引擎
        self.db_engine = DBManager()
        
        # 顶级布局
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        top_layout = QVBoxLayout(central_widget)
        top_layout.setContentsMargins(5, 5, 5, 5)
        
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("QTabBar::tab { height: 40px; font-size: 16px; font-weight: bold; padding: 0 20px; }")
        top_layout.addWidget(self.tabs)
        
        # ==================================
        # -- Tab 1: 实时防空作战大屏 --
        # ==================================
        self.tab1 = QWidget()
        self.setup_tab1()
        self.tabs.addTab(self.tab1, "📡 实时频谱监测模块")
        
        # ==================================
        # -- Tab 2: 历史特征数据库 --
        # ==================================
        self.tab2 = QWidget()
        self.setup_tab2()
        self.tabs.addTab(self.tab2, "🗄️ 异常信号特征数据库")
        
        # 当从前台切回后台表单时，我们让表单即刻刷新！
        self.tabs.currentChanged.connect(self.on_tab_changed)
        
        # ---- 启动强悍的后台多线程挂载操作 ----
        self.toolchain = RFToolchain()
        self.worker = RFPipelineWorker(self.toolchain)
        self.worker.frame_ready.connect(self.update_frame)
        self.worker.log_ready.connect(self.append_log)
        self.worker.alert_triggered.connect(self.on_alert_triggered)
        self.worker.start()

    def setup_tab1(self):
        main_layout = QHBoxLayout(self.tab1)
        
        left_layout = QVBoxLayout()
        left_layout.setContentsMargins(10, 10, 10, 10)
        
        self.btn_play = QPushButton("▶ 连续采集模式 (Continuous)")
        self.btn_play.setMinimumHeight(60)
        self.btn_play.setStyleSheet("background-color: #1b5e20; color: white; border-radius: 5px;")
        self.btn_play.clicked.connect(self.toggle_play)
        
        self.btn_step = QPushButton("⏭ 单帧步进分析 (Step-by-step)")
        self.btn_step.setMinimumHeight(60)
        self.btn_step.setStyleSheet("background-color: #e65100; color: white; border-radius: 5px;")
        self.btn_step.clicked.connect(self.step_once)
        
        self.log_box = QPlainTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setStyleSheet("background-color: #0d1117; color: #58a6ff; font-family: Consolas; font-size:15px;")
        
        left_layout.addWidget(self.btn_play)
        left_layout.addWidget(self.btn_step)
        left_layout.addWidget(self.log_box)
        
        self.img_label = QLabel()
        self.img_label.setFixedSize(640, 640)
        self.img_label.setStyleSheet("background-color: #000; border: 3px solid #666;")
        self.img_label.setAlignment(Qt.AlignCenter)
        self.img_label.setText("建立通信连接以接收实时时频张量流 ...")
        self.img_label.setStyleSheet("color: white; font-size: 18px;")
        
        main_layout.addLayout(left_layout, stretch=1)
        main_layout.addWidget(self.img_label)

    def setup_tab2(self):
        layout = QHBoxLayout(self.tab2)
        
        # >>> 左侧数据表
        self.db_table = QTableWidget()
        self.db_table.setColumnCount(4)
        self.db_table.setHorizontalHeaderLabels(["记录 ID", "时间戳", "中心频点 (MHz)", "AI 置信度"])
        self.db_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.db_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.db_table.setSelectionMode(QTableWidget.SingleSelection)
        self.db_table.setStyleSheet("font-size: 14px;")
        self.db_table.itemSelectionChanged.connect(self.on_db_row_selected)
        
        # 物理隐藏路径锚点
        self.current_db_paths = [] 
        
        # >>> 右侧历史图传投影仪
        self.db_img_label = QLabel("正在等待选中记录以呈现时频热图特征结构...")
        self.db_img_label.setFixedSize(640, 640)
        self.db_img_label.setStyleSheet("background-color: #1a1a1a; color: #aaa; font-size: 18px; border: 3px dashed #444;")
        self.db_img_label.setAlignment(Qt.AlignCenter)
        
        layout.addWidget(self.db_table, stretch=1)
        layout.addWidget(self.db_img_label)

    def toggle_play(self):
        if self.worker.running:
            self.worker.running = False
            self.btn_play.setText("▶ 连续采集模式 (Continuous)")
            self.btn_play.setStyleSheet("background-color: #1b5e20; color: white;")
        else:
            self.worker.running = True
            self.btn_play.setText("⏸ 挂起平台采样 (Suspend)")
            self.btn_play.setStyleSheet("background-color: #b71c1c; color: white;")
            
    def step_once(self):
        if self.worker.running:
            self.toggle_play()
        self.worker.step_once = True
        
    def append_log(self, text):
        self.log_box.appendPlainText(text)
        self.log_box.verticalScrollBar().setValue(self.log_box.verticalScrollBar().maximum())
        
    def render_cv2_to_qlabel(self, cv_img, qlabel):
        h, w, ch = cv_img.shape
        bytes_per_line = ch * w
        rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        qt_img = QImage(rgb_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
        qlabel.setPixmap(QPixmap.fromImage(qt_img))

    def update_frame(self, cv_img):
        self.render_cv2_to_qlabel(cv_img, self.img_label)

    def on_alert_triggered(self, alert_info, frame):
        """核心钩子：当接收到底层彻底抓获无人机的确认信号时，狂傲落盘！"""
        freq = alert_info.get("freq_mhz", 0.0)
        score = alert_info.get("score", 0.0)
        new_id = self.db_engine.log_alert(freq, score, frame)
        self.append_log(f"💾 数据持久化完成: 异常记录 ID: {new_id} 的时频特征矩阵已稳固至本地数据库。")

    def on_tab_changed(self, index):
        # 只要您敢切到案卷管理 Tab (索引 1)，我就敢刷最热库表！
        if index == 1:
            self.load_db_data()

    def load_db_data(self):
        rows = self.db_engine.get_all_alerts()
        self.db_table.setRowCount(len(rows))
        self.current_db_paths = []
        
        for row_idx, data in enumerate(rows):
            # data: (id, timestamp, freq, score, image_path)
            self.db_table.setItem(row_idx, 0, QTableWidgetItem(f"ID-{data[0]}"))
            self.db_table.setItem(row_idx, 1, QTableWidgetItem(str(data[1])))
            self.db_table.setItem(row_idx, 2, QTableWidgetItem(f"{data[2]} MHz"))
            self.db_table.setItem(row_idx, 3, QTableWidgetItem(f"{data[3] * 100:.2f} %"))
            self.current_db_paths.append(data[4])
            
        # 刷完表格顺便清空一下右侧
        self.db_img_label.clear()
        self.db_img_label.setText("数据加载完毕，请点选相关条目检视图像实体。")

    def on_db_row_selected(self):
        selected_items = self.db_table.selectedItems()
        if not selected_items: return
        row = selected_items[0].row()
        img_path = self.current_db_paths[row]
        
        cv_img = cv2.imread(img_path)
        if cv_img is not None:
            self.render_cv2_to_qlabel(cv_img, self.db_img_label)
        else:
            self.db_img_label.setText("❌ 数据读取异常：对应的物理图像实体丢失。")

    def closeEvent(self, event):
        self.worker.kill()
        self.worker.wait()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
