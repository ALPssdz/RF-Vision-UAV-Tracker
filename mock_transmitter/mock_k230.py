# -*- coding: utf-8 -*-
"""
mock_transmitter/mock_k230.py — K230 视觉节点模拟器
=====================================================
在 K230 硬件不在场时，于 Windows PC 上运行本脚本，
向香橙派提供标准接口的视觉数据：

  1. HTTP MJPEG 视频流（端口 8554）
     — 模拟 K230 的 RTSP 输出，OpenCV 可直接读取
     — 画面内容：含时间戳与状态的测试图案，或从 PC 摄像头采集

  2. UDP 遥测包（目标: 香橙派 192.168.31.34:8080）
     — 格式与真实 K230 完全一致
     — 可通过命令行参数模拟无人机检测事件

运行方式：
  python mock_k230.py              # 纯背景模式（无检测）
  python mock_k230.py --alert      # 模拟无人机检测触发告警

依赖：pip install numpy opencv-python
"""

import cv2
import numpy as np
import socket
import json
import time
import threading
import argparse
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

# ──────────────────────────────────────────────────────────────────────────────
# 配置参数
# ──────────────────────────────────────────────────────────────────────────────
ORANGEPI_IP  = "192.168.31.34"   # 香橙派 5 IP
UDP_PORT     = 8080               # 与 config.py K230_UDP_PORT 对应
HTTP_PORT    = 8554               # MJPEG 服务端口

# ──────────────────────────────────────────────────────────────────────────────
# 全局状态（线程间共享）
# ──────────────────────────────────────────────────────────────────────────────
_alert_mode    = False
_alert_conf    = 0.0
_alert_bbox    = []
_frame_counter = 0
_lock          = threading.Lock()


def _generate_test_frame(t: float, alert: bool, conf: float, bbox: list) -> np.ndarray:
    """
    生成测试图案帧。

    Parameters
    ----------
    t     : 运行时间（秒）
    alert : 是否处于告警状态
    conf  : 置信度
    bbox  : 检测框 [x1, y1, x2, y2]
    """
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    # 背景渐变（便于视觉确认帧在更新）
    phase = int((t * 30) % 256)
    frame[:, :, 0] = phase // 4          # B 通道缓慢变化
    frame[:240, :, 1] = 20               # 顶部区域
    frame[480:, :, 2] = 20               # 底部区域

    # 网格线（模拟摄像机场景结构）
    for x in range(0, 1280, 160):
        cv2.line(frame, (x, 0), (x, 720), (40, 40, 40), 1)
    for y in range(0, 720, 90):
        cv2.line(frame, (0, y), (1280, y), (40, 40, 40), 1)

    # 状态标注
    status_color = (0, 80, 255) if alert else (0, 220, 80)
    status_text  = f"[ALERT] UAV conf={conf:.2f}" if alert else "[MOCK K230] Standby"
    cv2.putText(frame, status_text,
                (40, 60), cv2.FONT_HERSHEY_DUPLEX, 1.4, status_color, 2)
    cv2.putText(frame, f"t = {t:7.2f} s",
                (40, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (180, 180, 180), 2)
    cv2.putText(frame, f"MockK230 -> OrangePi {ORANGEPI_IP}:{UDP_PORT}",
                (40, 680), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (120, 120, 120), 1)

    # 检测框
    if alert and len(bbox) == 4:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 80, 255), 3)
        cv2.putText(frame, f"UAV {conf:.2f}",
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 80, 255), 2)

    return frame


class _MJPEGHandler(BaseHTTPRequestHandler):
    """HTTP MJPEG 视频流请求处理器"""

    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type',
                         'multipart/x-mixed-replace; boundary=mjpegframe')
        self.end_headers()
        t_start = time.time()
        try:
            while True:
                t = time.time() - t_start
                with _lock:
                    alert = _alert_mode
                    conf  = _alert_conf
                    bbox  = _alert_bbox[:]
                frame = _generate_test_frame(t, alert, conf, bbox)
                ok, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if not ok:
                    continue
                payload = jpeg.tobytes()
                self.wfile.write(b'--mjpegframe\r\n')
                self.wfile.write(b'Content-Type: image/jpeg\r\n')
                self.wfile.write(f'Content-Length: {len(payload)}\r\n\r\n'.encode())
                self.wfile.write(payload)
                self.wfile.write(b'\r\n')
                time.sleep(1 / 25)   # 25 FPS
        except (BrokenPipeError, ConnectionResetError):
            pass  # 客户端断开，正常退出

    def log_message(self, fmt, *args):
        pass  # 抑制每帧的 HTTP 访问日志


def _udp_sender_loop():
    """持续向香橙派发送 UDP 遥测包"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    print(f"[UDP] 开始向 {ORANGEPI_IP}:{UDP_PORT} 发送遥测包...")
    while True:
        with _lock:
            packet = {
                "alert": _alert_mode,
                "conf":  round(_alert_conf, 3),
                "bbox":  _alert_bbox[:],
            }
        try:
            sock.sendto(json.dumps(packet).encode(), (ORANGEPI_IP, UDP_PORT))
        except Exception as e:
            print(f"[UDP] 发送失败: {e}")
        time.sleep(0.1)


def _alert_toggle_loop(interval: float = 5.0):
    """
    自动周期性切换告警状态（用于功能验证）。
    每隔 interval 秒在告警/静默之间切换。
    """
    while True:
        time.sleep(interval)
        with _lock:
            global _alert_mode, _alert_conf, _alert_bbox
            _alert_mode = not _alert_mode
            _alert_conf = 0.87 if _alert_mode else 0.0
            _alert_bbox = [480, 200, 800, 480] if _alert_mode else []
        state = "ALERT" if _alert_mode else "standby"
        print(f"[MOCK] 状态切换 → {state}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mock K230 视觉节点模拟器")
    parser.add_argument("--alert",  action="store_true", help="启动时即处于告警状态")
    parser.add_argument("--toggle", action="store_true", help="每 5 秒自动切换告警状态（功能测试用）")
    parser.add_argument("--port",   type=int, default=HTTP_PORT, help=f"MJPEG HTTP 端口（默认 {HTTP_PORT}）")
    args = parser.parse_args()

    HTTP_PORT = args.port

    with _lock:
        _alert_mode = args.alert
        _alert_conf = 0.87 if args.alert else 0.0
        _alert_bbox = [480, 200, 800, 480] if args.alert else []

    # 启动 UDP 遥测发送线程
    threading.Thread(target=_udp_sender_loop, daemon=True).start()

    # 启动自动切换线程（调试用）
    if args.toggle:
        print("[MOCK] 自动切换模式：每 5 秒翻转告警状态")
        threading.Thread(target=_alert_toggle_loop, daemon=True).start()

    print("=" * 55)
    print(f"  Mock K230 模拟器已启动")
    print(f"  MJPEG 视频流: http://192.168.31.206:{HTTP_PORT}/")
    print(f"  UDP 遥测目标: {ORANGEPI_IP}:{UDP_PORT}")
    print(f"  初始状态    : {'告警' if args.alert else '静默'}")
    print(f"  修改 config.py 中 K230_RTSP_URL 为上述 HTTP 地址")
    print("=" * 55)

    try:
        HTTPServer(("0.0.0.0", HTTP_PORT), _MJPEGHandler).serve_forever()
    except KeyboardInterrupt:
        print("\n[MOCK] 模拟器已停止。")
