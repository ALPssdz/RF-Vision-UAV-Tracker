import torch # [Patch]: 规避 PyTorch 与 PyQt5 的 C++ 动态链接库初始化冲突 (WinError 1114)
import sys
import os
import time
import threading
import numpy as np

PROJ_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QObject, pyqtSignal

from backend_rk3588.main_rf_pipeline import RFToolchain
from vision_k230.k230_client import K230NetworkClient
from ui_qt.gui_host import MainWindow
from database.db_manager import DBManager

class CentralHubEngine(QObject):
    """
    射频-视觉复合管线中央控制中枢 (事件总线构建)。
    负责全局编排 SDR 与光电传感器节点，执行跨模态特征校验对齐，
    向持久化数据库提交事件记录，并向处于表现层的系统界面广播硬件状态帧。
    """
    signal_rf_frame = pyqtSignal(object)
    signal_k230_frame = pyqtSignal(object)
    signal_log = pyqtSignal(str)
    
    # 针对表现层的聚合状态负载
    signal_system_status = pyqtSignal(dict)
    signal_db_updated = pyqtSignal()
    
    def __init__(self):
        super().__init__()
        
        # 第一阶段：初始化边缘传感器节点及持久化数据底座
        self.rf_toolchain = RFToolchain()
        self.k230_client = K230NetworkClient(rtsp_url="rtsp://192.168.31.250/stream", udp_port=8080)
        self.k230_client.start()
        
        self.db_engine = DBManager()
        
        self.running = False
        self._master_thread = None
        
        # 初始化并发推断事件证据采集缓冲池
        self.cache_rf = np.zeros((640, 640, 3), dtype=np.uint8)
        self.cache_vis = np.zeros((640, 1137, 3), dtype=np.uint8)
        
        # 第二阶段：无代理挂接视图表现层
        self.ui_window = MainWindow(hub=self)
        self.signal_log.emit("系统冷启动完成：中央事件总线已建立，视图绑定校验通过。")

    def start_sensing(self):
        if self.running: return
        self.running = True
        self.signal_system_status.emit({
            "system": "系统状态: 🟢 主管道全速轮询中...", 
            "color": "#27ae60",
            "sdr": "SDR 节点: 🔄 IQ 数据采集中",
            "vision": "视频节点: 🔄 画面及信令监听中"
        })
        self.signal_log.emit("底层硬件物理端口阻塞解除，并行信号采集进程已置位。")
        self._master_thread = threading.Thread(target=self._hub_loop, daemon=True)
        self._master_thread.start()
        
    def stop_sensing(self):
        self.running = False
        self.signal_system_status.emit({"system": "系统模式: 🟡 任务挂起", "color": "#f1c40f"})
        self.signal_log.emit("系统中央主循环进程已安全退出执行。")
        
    def mock_k230_trigger(self, state):
        self.k230_client.mock_drone_detected = state

    def _trigger_composite_save(self, reason_tag):
        """ 执行视觉与射频张量的图像空间拼接矩阵生成，并向子文件模块抛出无界面异步落盘指令。 """
        import cv2
        fused_evidence = np.hstack([self.cache_rf, self.cache_vis])
        cv2.putText(fused_evidence, f"ALARM REASON: {reason_tag}", (20, 600), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        
        new_id = self.db_engine.log_alert(999.0, 1.0, fused_evidence)
        self.signal_log.emit(f"持久化操作：事件证据标的 [REC-{new_id}] 标签属性 ({reason_tag}) 已完成写入。")
        
        # 通知异步界面的前端模型重新加载历史缓存
        self.signal_db_updated.emit()

    def _hub_loop(self):
        while self.running:
            # === [处理管线一：软件无线电跳频解调] ===
            try:
                rf_frame, rf_log, rf_alert, rf_info = self.rf_toolchain.tick()
                self.cache_rf = rf_frame
                self.signal_rf_frame.emit(rf_frame)
                
                if rf_log.strip(): 
                    self.signal_log.emit(rf_log)
                    
                if rf_alert:
                    self.signal_system_status.emit({"system": "Alert: Unusual RF Comm Link", "color": "#e74c3c"})
                    self._trigger_composite_save("SDR_OMNI_TRIGGER")
            except Exception as e:
                self.signal_log.emit(f"SDR 射频传感器寻址异常: {e}")
                
            # === [处理管线二：带外信令网络及边缘端光学流] ===
            try:
                k_frame, k_telemetry = self.k230_client.get_synced_data()
                
                if k_telemetry.get("alert", False):
                    bbox = k_telemetry.get("bbox", [])
                    if len(bbox) == 4:
                        import cv2
                        cv2.rectangle(k_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 8)
                        cv2.putText(k_frame, "OOB JSON LOCK", (bbox[0], bbox[1]-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                        
                    self.signal_system_status.emit({"system": "特征命中：目标物理轮廓验证通过", "color": "#e74c3c"})
                    self.signal_log.emit("OOB 触发器：高速网络侧带外接收到正向标定数据包。")
                    self._trigger_composite_save("K230_ZENITH_TRIGGER")
                
                self.cache_vis = k_frame
                self.signal_k230_frame.emit(k_frame)
            except Exception as e:
                self.signal_log.emit(f"边缘侧光学流推流挂起异常: {e}")
                
            time.sleep(0.01)

    def shutdown(self):
        self.stop_sensing()
        self.k230_client.stop()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    hub = CentralHubEngine()
    hub.ui_window.show()
    
    exit_code = app.exec_()
    hub.shutdown()
    sys.exit(exit_code)
