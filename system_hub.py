import torch # [Patch]: Prevents WinError 1114 caused by pyqt5 and torch C++ DLL initialization conflicts.
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
    Central Controller (Event Bus) for the RF-Vision Pipeline.
    Responsible for orchestrating SDR and Optical sensors, executing cross-check alignments,
    committing evidence to the persistent database, and broadcasting states to the View Layer.
    """
    signal_rf_frame = pyqtSignal(object)
    signal_k230_frame = pyqtSignal(object)
    signal_log = pyqtSignal(str)
    
    # Aggregated state payloads for the Presentation Layer
    signal_system_status = pyqtSignal(dict)
    signal_db_updated = pyqtSignal()
    
    def __init__(self):
        super().__init__()
        
        # Phase 1: Initialize Sensor Peripherals and Data Layers
        self.rf_toolchain = RFToolchain()
        self.k230_client = K230NetworkClient(rtsp_url="rtsp://192.168.31.250/stream", udp_port=8080)
        self.k230_client.start()
        
        self.db_engine = DBManager()
        
        self.running = False
        self._master_thread = None
        
        # Keep track of latest frames for evidence aggregation
        self.cache_rf = np.zeros((640, 640, 3), dtype=np.uint8)
        self.cache_vis = np.zeros((640, 1137, 3), dtype=np.uint8)
        
        # Phase 2: Attach View Plugin
        self.ui_window = MainWindow(hub=self)
        self.signal_log.emit("System Boot Success: Central bus initialized. View bindings established.")

    def start_sensing(self):
        if self.running: return
        self.running = True
        self.signal_system_status.emit({
            "system": "系统状态: 🟢 主管道全速轮询中...", 
            "color": "#27ae60",
            "sdr": "SDR 节点: 🔄 IQ 数据采集中",
            "vision": "视频节点: 🔄 画面及信令监听中"
        })
        self.signal_log.emit("Hardware pipelines unblocked. Sensing thread engaged.")
        self._master_thread = threading.Thread(target=self._hub_loop, daemon=True)
        self._master_thread.start()
        
    def stop_sensing(self):
        self.running = False
        self.signal_system_status.emit({"system": "System Suspended", "color": "#f1c40f"})
        self.signal_log.emit("System execution loop halted successfully.")
        
    def mock_k230_trigger(self, state):
        self.k230_client.mock_drone_detected = state

    def _trigger_composite_save(self, reason_tag):
        """ Composites visual and RF buffers and commits to the database independently of the UI. """
        import cv2
        fused_evidence = np.hstack([self.cache_rf, self.cache_vis])
        cv2.putText(fused_evidence, f"ALARM REASON: {reason_tag}", (20, 600), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        
        new_id = self.db_engine.log_alert(999.0, 1.0, fused_evidence)
        self.signal_log.emit(f"Persisted: Event [REC-{new_id}] tagged with ({reason_tag}).")
        
        # Notify the UI to refresh its tables
        self.signal_db_updated.emit()

    def _hub_loop(self):
        while self.running:
            # === [Pipeline 1: RF Demodulation] ===
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
                self.signal_log.emit(f"SDR Polling Exception: {e}")
                
            # === [Pipeline 2: OOB Json & RTSP Vision] ===
            try:
                k_frame, k_telemetry = self.k230_client.get_synced_data()
                
                if k_telemetry.get("alert", False):
                    bbox = k_telemetry.get("bbox", [])
                    if len(bbox) == 4:
                        import cv2
                        cv2.rectangle(k_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 8)
                        cv2.putText(k_frame, "OOB JSON LOCK", (bbox[0], bbox[1]-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                        
                    self.signal_system_status.emit({"system": "Alert: Visual Object Detected", "color": "#e74c3c"})
                    self.signal_log.emit("OOB Trigger: Target locked via UDP fast-channel.")
                    self._trigger_composite_save("K230_ZENITH_TRIGGER")
                
                self.cache_vis = k_frame
                self.signal_k230_frame.emit(k_frame)
            except Exception as e:
                self.signal_log.emit(f"Vision Stream Exception: {e}")
                
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
