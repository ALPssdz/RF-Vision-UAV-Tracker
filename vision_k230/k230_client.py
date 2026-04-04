import cv2
import numpy as np
import threading
import socket
import json
import time

class K230NetworkClient:
    """
    K230 Edge Node Network Client.
    Responsible for receiving asynchronous Out-Of-Band (OOB) UDP telemetry packets
    with minimal latency constraints, alongside the decoupled High-throughput RTSP video stream.
    """
    def __init__(self, rtsp_url="rtsp://192.168.31.250/stream", udp_port=8080):
        self.rtsp_url = rtsp_url
        self.udp_port = udp_port
        
        # Local state buffers
        self.latest_frame = np.zeros((640, 1137, 3), dtype=np.uint8)
        self.latest_telemetry = {"alert": False, "confidence": 0.0, "bbox": []}
        
        self.running = False
        self.mock_drone_detected = False # Toggled via Dev Tools in UI
        
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
        Exposes the temporally aligned memory snapshot of the current video buffer 
        and the OOB JSON event parameters to the central scheduler.
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
        """ Decoupled RTSP H.264 Polling Thread """
        cap = cv2.VideoCapture(self.stream_url) if hasattr(self, 'stream_url') else cv2.VideoCapture(self.rtsp_url)
        while self.running:
            ret, frame = cap.read()
            if not ret:
                blank = np.zeros((640, 1137, 3), dtype=np.uint8)
                cv2.putText(blank, "[K230] RTSP UNAVAILABLE", (150, 320), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (100, 100, 200), 3)
                self.latest_frame = blank
                time.sleep(0.5)
                # Auto-resume mechanism
                cap = cv2.VideoCapture(self.rtsp_url)
            else:
                h, w = frame.shape[:2]
                new_w = int((640 / h) * w)
                self.latest_frame = cv2.resize(frame, (new_w, 640))
            time.sleep(0.02) # Enforce 50FPS max limits
            
        cap.release()

    def _udp_loop(self):
        """ Hard-realtime UDP polling channel for OOB JSON payloads """
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
