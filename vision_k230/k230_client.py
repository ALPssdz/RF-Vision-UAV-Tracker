import cv2
import numpy as np
import threading
import socket
import json
import copy
import time


class K230NetworkClient:
    """
    K230 边缘视觉节点网络通信客户端。

    本模块通过两条并行网络信道与 K230 边缘节点进行通信：
      - **RTSP/TCP 信道**：接收 H.264 编码视频流，用于人工复核与 GUI 展示；
      - **UDP OOB 信道**：接收轻量级目标锁定遥测包（边界框坐标 + 置信度），
        实现与视频流解耦的低延迟告警触发。

    视频轮询与 UDP 监听分别运行于独立的守护线程，通过内部共享缓冲区与主线程
    进行异步数据交换。
    """

    def __init__(self, rtsp_url: str = "rtsp://192.168.31.250/stream",
                 udp_port: int = 8080):
        """
        Parameters
        ----------
        rtsp_url : str
            K230 视频推流地址（RTSP 或 HTTP-MJPEG 均可）。
        udp_port : int
            本机 UDP 监听端口，用于接收 K230 上报的遥测数据包。
        """
        self.rtsp_url = rtsp_url
        self.udp_port = udp_port

        # 线程间共享的最新数据缓冲区（由守护线程写入，主线程只读复制）
        self.latest_frame     = np.zeros((640, 1137, 3), dtype=np.uint8)
        self.latest_telemetry = {"alert": False, "confidence": 0.0, "bbox": []}

        self.running = False

        self._video_thread = None
        self._udp_thread   = None
        self._sock         = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.settimeout(0.5)

    def start(self):
        """启动视频采集线程与 UDP 遥测监听线程。"""
        self.running = True
        self._sock.bind(("0.0.0.0", self.udp_port))

        self._video_thread = threading.Thread(target=self._video_loop, daemon=True)
        self._udp_thread   = threading.Thread(target=self._udp_loop,   daemon=True)

        self._video_thread.start()
        self._udp_thread.start()

    def stop(self):
        """发出停止信号，等待守护线程退出并释放 socket 资源。"""
        self.running = False
        self._sock.close()
        if self._video_thread:
            self._video_thread.join(timeout=1.0)
        if self._udp_thread:
            self._udp_thread.join(timeout=1.0)

    def get_synced_data(self) -> tuple:
        """
        获取最新的视频帧与遥测数据的深拷贝副本，供主线程安全读取。

        Returns
        -------
        tuple of (numpy.ndarray, dict)
            - frame      : 形状为 (640, width, 3) 的 BGR 视频帧；
            - telemetry  : 遥测字典，键包括 ``alert`` (bool)、
                           ``confidence`` (float)、``bbox`` (list)。
        """
        telemetry = copy.deepcopy(self.latest_telemetry)
        frame     = self.latest_frame.copy()
        return frame, telemetry

    # ------------------------------------------------------------------
    # 守护线程内部方法
    # ------------------------------------------------------------------

    def _video_loop(self):
        """
        RTSP 视频流采集循环（守护线程）。

        持续从 RTSP/HTTP 地址读取视频帧，按比例缩放至高度 640 像素后
        写入共享缓冲区。连接中断时自动重新建立连接。
        """
        cap = cv2.VideoCapture(self.rtsp_url)
        while self.running:
            ret, frame = cap.read()
            if not ret:
                # 连接中断：写入占位图像并尝试重连
                blank = np.zeros((640, 1137, 3), dtype=np.uint8)
                cv2.putText(blank, "[K230] RTSP UNAVAILABLE",
                            (150, 320), cv2.FONT_HERSHEY_SIMPLEX,
                            1.5, (100, 100, 200), 3)
                self.latest_frame = blank
                time.sleep(0.5)
                cap = cv2.VideoCapture(self.rtsp_url)
            else:
                h, w   = frame.shape[:2]
                new_w  = int((640 / h) * w)
                self.latest_frame = cv2.resize(frame, (new_w, 640))

            time.sleep(0.02)  # 限制轮询频率上限为 50 Hz

        cap.release()

    def _udp_loop(self):
        """
        UDP OOB 遥测数据包接收循环（守护线程）。

        持续监听指定 UDP 端口，解析 JSON 格式的遥测数据包并更新共享缓冲区。
        超时或格式异常时静默跳过，不中断循环。
        """
        while self.running:
            try:
                data, _ = self._sock.recvfrom(1024)
                if data:
                    packet = json.loads(data.decode("utf-8"))
                    self.latest_telemetry["alert"]      = packet.get("alert", False)
                    self.latest_telemetry["bbox"]       = packet.get("bbox",  [])
                    self.latest_telemetry["confidence"] = packet.get("conf",  0.0)
            except socket.timeout:
                pass
            except Exception:
                pass
