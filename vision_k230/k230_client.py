import cv2
import numpy as np
import threading
import socket
import json
import time

class K230NetworkClient:
    """
    K230 边缘节点网络通信客户端配置实体。
    此适配器主要负责高保真获取非对称的异步带外 UDP (Out-Of-Band) 信令极低延迟遥测包，
    同时在解耦的旁支线程获取高吞吐量的连贯化预解码 RTSP 光流媒体反馈。
    """
    def __init__(self, rtsp_url="rtsp://192.168.31.250/stream", udp_port=8080):
        self.rtsp_url = rtsp_url
        self.udp_port = udp_port
        
        # 建立线程读写的内部本地数据缓冲池
        self.latest_frame = np.zeros((640, 1137, 3), dtype=np.uint8)
        self.latest_telemetry = {"alert": False, "confidence": 0.0, "bbox": []}
        
        self.running = False
        self.mock_drone_detected = False # 经 UI 呈现层内部调试面板下发的人工覆写开关
        
        self._video_thread = None
        self._udp_thread = None
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.settimeout(0.5)

    def start(self):
        self.running = True
        self._sock.bind(("0.0.0.0", self.udp_port))
        
        self._video_thread = threading.Thread(target=self._video_loop, daemon=True)
        self._udp_thread = threading.Thread(target=self._udp_loop, daemon=True)
        
        self._video_thread.start()
        self._udp_thread.start()

    def stop(self):
        self.running = False
        self._sock.close()
        if self._video_thread: self._video_thread.join(timeout=1.0)
        if self._udp_thread: self._udp_thread.join(timeout=1.0)

    def get_synced_data(self):
        """ 
        向上限中央总线透传在物理时空刻表产生交叉时间对齐的张量特征流副本。
        """
        import copy
        telemetry = copy.deepcopy(self.latest_telemetry)
        
        if self.mock_drone_detected:
            telemetry["alert"] = True
            telemetry["confidence"] = 99.9
            telemetry["bbox"] = [800, 400, 1100, 600]
            
        frame = self.latest_frame.copy()
        return frame, telemetry

    def _video_loop(self):
        """ 解耦异步 RTSP H.264 视频推流轮询处理线程 """
        cap = cv2.VideoCapture(self.stream_url) if hasattr(self, 'stream_url') else cv2.VideoCapture(self.rtsp_url)
        while self.running:
            ret, frame = cap.read()
            if not ret:
                blank = np.zeros((640, 1137, 3), dtype=np.uint8)
                cv2.putText(blank, "[K230] RTSP UNAVAILABLE", (150, 320), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (100, 100, 200), 3)
                self.latest_frame = blank
                time.sleep(0.5)
                # 自愈式异常重联执行器机制
                cap = cv2.VideoCapture(self.rtsp_url)
            else:
                h, w = frame.shape[:2]
                new_w = int((640 / h) * w)
                self.latest_frame = cv2.resize(frame, (new_w, 640))
            time.sleep(0.02) # 强制 50FPS 最高帧率阀门管制
            
        cap.release()

    def _udp_loop(self):
        """ 强化硬级别实时优先级的端到端带外 UDP 遥测信函提取环路 """
        while self.running:
            try:
                data, addr = self._sock.recvfrom(1024)
                if data:
                    json_str = data.decode('utf-8')
                    packet = json.loads(json_str)
                    
                    self.latest_telemetry["alert"] = packet.get("alert", False)
                    self.latest_telemetry["bbox"] = packet.get("bbox", [])
                    self.latest_telemetry["confidence"] = packet.get("conf", 0.0)
                    
            except socket.timeout:
                pass
            except Exception as e:
                pass
