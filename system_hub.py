#!/usr/bin/env python3
import torch # [Patch]: 规避 PyTorch 与 PyQt5 的 C++ 动态链接库初始化冲突 (WinError 1114)
import sys
import os
import time
import threading
import numpy as np

PROJ_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

from PyQt5.QtCore import Qt, QEvent
from PyQt5.QtGui import QImage, QPixmap, QFont, QKeySequence
from PyQt5.QtWidgets import QApplication, QShortcut
from PyQt5.QtCore import QObject, pyqtSignal

from backend_rk3588 import config as cfg
from backend_rk3588.main_rf_pipeline import RFToolchain
from vision_k230.k230_client import K230NetworkClient
from ui_qt.gui_host import MainWindow
from database.db_manager import DBManager


# ==============================================================================
# stdout/stderr 重定向器
# 功能：将系统中所有 print() 输出路由到 GUI 日志框
# 分两阶段工作：
#   ① 预 GUI 阶段（hub 初始化期间）：存入内部缓冲区，并同步写入原 stdout
#   ② GUI 就绪后：flush 缓冲区 + 直接 emit 到 signal_log
# ==============================================================================
class _GuiLogRedirector:
    """
    sys.stdout 重定向器：将 print() 输出路由至 Qt signal_log 信号。

    使用方法：
        redirector = _GuiLogRedirector()
        sys.stdout = redirector
        # ... 创建 hub / GUI ...
        redirector.attach_signal(hub.signal_log.emit)
    """

    def __init__(self):
        self._lock      = threading.Lock()
        self._buf       = []           # 预 GUI 阶段缓冲区
        self._emit_fn   = None         # GUI 就绪后的 signal.emit 函数
        self._orig      = sys.__stdout__   # 保留原 stdout 用于 debug

    def attach_signal(self, emit_fn):
        """
        绑定 Qt 信号 emit 函数，并将缓冲区内容 flush 至 GUI 日志框。
        在 CentralHubEngine.__init__() 完成后（GUI 已构建）立即调用。
        """
        with self._lock:
            self._emit_fn = emit_fn
            for line in self._buf:
                try:
                    emit_fn(line)
                except Exception:
                    pass
            self._buf.clear()

    def write(self, text: str):
        # 始终写入原 stdout（保留终端可见性）
        if self._orig:
            try:
                self._orig.write(text)
            except Exception:
                pass

        with self._lock:
            # 按行切分，过滤纯空行避免日志框刷屏
            for line in text.split('\n'):
                stripped = line.rstrip()
                if not stripped:
                    continue
                if self._emit_fn:
                    try:
                        self._emit_fn(stripped)
                    except Exception:
                        self._buf.append(stripped)
                else:
                    self._buf.append(stripped)

    def flush(self):
        if self._orig:
            try:
                self._orig.flush()
            except Exception:
                pass

    def fileno(self):
        """兼容需要 fd 的库（如 tqdm）"""
        return self._orig.fileno() if self._orig else 1

class CentralHubEngine(QObject):
    """
    系统级中央调度引擎（Central Orchestration Engine）

    基于事件总线（Event Bus）架构，负责协调射频检测子系统与光电视觉子系统
    的并发数据流，执行跨模态特征融合对齐，将告警事件持久化写入数据库，
    并通过 Qt 信号机制向 GUI 表现层广播实时状态。

    子系统组成：
      - RFToolchain   : 三级射频检测流水线（S1-RSSI / S2-YOLO / S3-CycloAudit）
      - K230Client    : K230 边缘端光电视觉流接收与带外信令解析
      - DBManager     : SQLite 告警事件持久化引擎
      - MainWindow    : PyQt5 GUI 表现层
    """
    signal_rf_frame = pyqtSignal(object)
    signal_k230_frame = pyqtSignal(object)
    signal_log = pyqtSignal(str)
    
    # 针对表现层的聚合状态负载
    signal_system_status = pyqtSignal(dict)
    signal_db_updated = pyqtSignal()
    
    def __init__(self):
        super().__init__()
        
        # 初始化各子系统节点
        self.rf_toolchain = RFToolchain()
        self.k230_client  = K230NetworkClient(
            rtsp_url=cfg.K230_RTSP_URL, udp_port=cfg.K230_UDP_PORT
        )
        self.k230_client.start()
        self.db_engine = DBManager()

        self.running        = False
        self._master_thread = None
        self._tick_lock     = threading.Lock()  # 防止 tick() 在 stop/start 切换时重入

        # ── 时序持久化滤波器（Temporal Persistence Filter v2）───────────────────
        # 自适应 bypass 机制：
        #   强信号（score ≥ bypass_th）：单 tick 直通，0 延迟
        #   普通信号（th ≤ score < bypass_th）：2-tick 确认
        #   噪声（score < th）：streak 重置
        #
        # 虚警率模型（弱信号分支）：P_fa_final = P_fa_tick^N
        #   N=2, P_fa_tick=5% → P_fa_final = 0.25%
        # 强信号旁路条件（单 tick）：
        #   score ≥ BYPASS_RATIO × th ≈ 6~12%（典型 OcuSync 近场 NCC 8~25%）
        #   在此 NCC 量级下，PSR+CFS 双重门限已保证 Pfa < 0.1%，TPF 无需介入
        self._rf_confirm_streak   = 0
        self._rf_confirm_info     = {}    # 缓存最近一次有效告警信息
        self.RF_CONFIRM_REQUIRED  = 2     # 弱信号分支：连续 2 tick 才入库报警
        self.RF_STRONG_BYPASS_RATIO = 3.0 # 强信号旁路：score ≥ 3× 阈值时单 tick 直通


        # RF 与视觉通道的最新帧缓存（供多模态证据融合使用）
        self.cache_rf  = np.zeros((640, 640, 3),  dtype=np.uint8)
        self.cache_vis = np.zeros((640, 1137, 3), dtype=np.uint8)

        # 实例化 GUI 表现层并注入中央事件总线引用
        self.ui_window = MainWindow(hub=self)
        self.signal_log.emit("系统初始化完成：各子系统节点已就绪，中央事件路由建立。")

    def start_sensing(self):
        if self.running: return
        self.running = True
        self.signal_system_status.emit({
            "system": "[ACTIVE] 主管道全速轮询中...", 
            "color": "#27ae60",
            "sdr": "SDR 节点: [RX] IQ 数据采集中",
            "vision": "视频节点: [RX] 画面及信令监听中"
        })
        self.signal_log.emit("采集管道启动：SDR 前端及视觉网络客户端已进入工作状态。")
        self._master_thread = threading.Thread(target=self._hub_loop, daemon=True)
        self._master_thread.start()
        
    def _get_bypass_threshold(self, freq_mhz: float) -> float:
        """
        计算强信号旁路阈值（bypass threshold）。

        原理：
          当 S3 NCC 得分超过检测阈值 RF_STRONG_BYPASS_RATIO 倍时，
          统计上 PSR+CFS 双重门限已足够排除误报，TPF 2-tick 等待无需介入。

          bypass_th = max(th_30k, th_15k) × RF_STRONG_BYPASS_RATIO

          优先使用 calibrate_s3 写入的每扇区标定阈值；
          JSON 缺失时回退到类级别硬编码下限 × BYPASS_RATIO。

        Parameters
        ----------
        freq_mhz : float — 当前扇区中心频率（MHz）

        Returns
        -------
        float — bypass NCC 绝对阈值
        """
        try:
            thresholds = self.rf_toolchain.stage3_audit._sector_thresholds
            if thresholds:
                freq_hz = freq_mhz * 1e6
                key = min(thresholds, key=lambda k: abs(k - int(freq_hz)))
                th_30k, th_15k = thresholds[key]
                return max(th_30k, th_15k) * self.RF_STRONG_BYPASS_RATIO
        except Exception:
            pass
        # 回退：使用硬编码下限（1.8% × 3.0 = 5.4%）
        return 0.018 * self.RF_STRONG_BYPASS_RATIO

    def stop_sensing(self):
        self.running = False
        self.signal_system_status.emit({"system": "系统模式: [挂起] 任务挂起", "color": "#f1c40f"})
        self.signal_log.emit("系统中央主循环进程已安全退出执行。")
        

    def _trigger_composite_save(self, reason_tag, freq_mhz, score):
        """
        生成多模态证据融合图像并写入告警数据库。

        将当前射频频谱帧（cache_rf）与视觉图像帧（cache_vis）水平拼接，
        叠加告警标注后，调用 DBManager 完成持久化存储。
        """
        import cv2
        fused_evidence = np.hstack([self.cache_rf, self.cache_vis])
        cv2.putText(fused_evidence, f"ALARM REASON: {reason_tag}", (20, 600), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        
        new_id = self.db_engine.log_alert(freq_mhz, score, fused_evidence)
        self.signal_log.emit(f"持久化操作：事件证据标的 [REC-{new_id}] 标签属性 ({reason_tag}) 已完成写入。")
        
        # 通知异步界面的前端模型重新加载历史缓存
        self.signal_db_updated.emit()

    def _hub_loop(self):
        """
        主监测循环（在独立守护线程中运行）。

        管线一（RF）：调用 RFToolchain.tick() 执行一次三级检测周期，
                      处理告警结果并更新 GUI 状态。
        管线二（光电视觉）：从 K230 客户端获取视频帧与带外信令，
                           若检测到目标则生成告警记录。
        """
        while self.running:
            # === [处理管线一：软件无线电跳频解调] ===
            if not self._tick_lock.acquire(blocking=False):
                time.sleep(0.01)  # 上一次 tick 尚未完成，等待后重试
                continue
            try:
                rf_frame, rf_log, rf_alert, rf_info = self.rf_toolchain.tick()
                self.cache_rf = rf_frame
                self.signal_rf_frame.emit(rf_frame)

                if rf_log.strip():
                    self.signal_log.emit(rf_log)

                # ── 自适应时序持久化滤波器（Adaptive TPF v2）─────────────────────
                # 分支 A — 强信号旁路（bypass）:
                #   score ≥ 3× th_calibrated → 单 tick 直通，无延迟
                #   Pfa 在此 NCC 量级下由 PSR+CFS 保证 < 0.1%，TPF 无需加锁
                #
                # 分支 B — 弱信号确认:
                #   th ≤ score < 3× th → 维持 2-tick 计数确认（P_fa_final ≈ 0.25%）
                #
                # 分支 C — 噪声:
                #   score < th → streak 归零
                # ────────────────────────────────────────────────────────────────
                if rf_alert:
                    freq_mhz = rf_info.get("freq_mhz", 0.0)
                    score    = rf_info.get("score",    0.0)
                    bypass_th = self._get_bypass_threshold(freq_mhz)

                    if score >= bypass_th:
                        # ── 分支 A：强信号直通 ──────────────────────────────────
                        self._rf_confirm_streak = self.RF_CONFIRM_REQUIRED  # 直接置满
                        self._rf_confirm_info   = rf_info
                        self.signal_log.emit(
                            f"[TPF-BYPASS] 强信号直通！"
                            f"NCC={score*100:.2f}% ≥ bypass_th={bypass_th*100:.2f}% "
                            f"({self.RF_STRONG_BYPASS_RATIO:.1f}× th) "
                            f"freq={freq_mhz:.0f}MHz")
                    else:
                        # ── 分支 B：弱信号 2-tick 确认 ──────────────────────────
                        self._rf_confirm_streak += 1
                        self._rf_confirm_info    = rf_info
                        streak = self._rf_confirm_streak
                        self.signal_log.emit(
                            f"[TPF] S3 确认 {streak}/{self.RF_CONFIRM_REQUIRED} tick "
                            f"(freq={freq_mhz:.0f}MHz "
                            f"score={score*100:.1f}% bypass_th={bypass_th*100:.1f}%)")

                    if self._rf_confirm_streak >= self.RF_CONFIRM_REQUIRED:
                        # 达标（bypass 直通 或 累计确认）→ 发出最终告警
                        self.signal_system_status.emit({
                            "system": "[!] Alert: OcuSync RF Detected!",
                            "color":  "#e74c3c",
                            "alert":  True,
                        })
                        freq  = self._rf_confirm_info.get("freq_mhz", 0.0)
                        score_out = self._rf_confirm_info.get("score", 0.0)
                        self._trigger_composite_save("SDR_OMNI_TRIGGER", freq, score_out)
                        # 不重置 streak，维持告警状态直到信号消失
                else:
                    # ── 分支 C：S3 未通过，归零计数器 ──────────────────────────
                    if self._rf_confirm_streak > 0:
                        self.signal_log.emit(
                            f"[TPF] 连续确认中断（streak 重置: "
                            f"{self._rf_confirm_streak} → 0）")
                    self._rf_confirm_streak = 0
                    self.signal_system_status.emit({
                        "system": f"系统状态: [扫描] ({rf_info.get('freq_mhz', 0):.0f}MHz)" if rf_info else "系统状态: 主管道全速扫描中...",
                        "color": "#27ae60"
                    })
            except Exception as e:
                self.signal_log.emit(f"SDR 射频传感器寻址异常: {e}")
            finally:
                self._tick_lock.release()
                
            # === [处理管线二：带外信令网络及边缘端光学流] ===
            try:
                k_frame, k_telemetry = self.k230_client.get_synced_data()
                
                if k_telemetry.get("alert", False):
                    bbox = k_telemetry.get("bbox", [])
                    if len(bbox) == 4:
                        import cv2
                        cv2.rectangle(k_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 8)
                        cv2.putText(k_frame, "OOB JSON LOCK", (bbox[0], bbox[1]-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                        
                    self.signal_system_status.emit({
                        "system": "特征命中：目标物理轮廓验证通过",
                        "color":  "#e74c3c",
                        "alert":  True,   # 语义字段—GUI 告警计数使用
                    })
                    self.signal_log.emit("OOB 触发器：高速网络侧带外接收到正向标定数据包。")
                    self._trigger_composite_save("K230_ZENITH_TRIGGER", 0.0, 1.0)
                
                self.cache_vis = k_frame
                self.signal_k230_frame.emit(k_frame)
            except Exception as e:
                self.signal_log.emit(f"边缘侧光学流推流挂起异常: {e}")
                
            time.sleep(0.001)  # 1ms 让出 CPU 给 Qt 事件循环，避免 UI 卡顿

    def shutdown(self):
        self.stop_sensing()
        self.k230_client.stop()

if __name__ == "__main__":
    # ── Step 1: 安装 stdout 重定向器 ──────────────────────────────────────────
    # 在 QApplication 和 hub 创建之前安装，确保 RFToolchain 初始化期间的
    # 所有 print() 输出（如 S3 阈值加载信息）都被捕获，最终显示在 GUI 日志框。
    _redirector = _GuiLogRedirector()
    sys.stdout  = _redirector
    sys.stderr  = _redirector   # 同时捕获异常 traceback 与警告信息

    # ── Step 2: 启动 Qt 应用与 Hub（GUI 优先弹出，不等待标定）────────────────
    app = QApplication(sys.argv)
    hub = CentralHubEngine()       # 内部创建 MainWindow 及 RFToolchain
                                   # 期间的 print() 进入 _redirector 缓冲区
    hub.ui_window.showFullScreen()  # 全屏显示，立即弹出无需等待标定完成

    # ── Step 3: 绑定信号，flush 缓冲区至 GUI 日志框 ───────────────────────────
    # attach_signal 调用后，已缓冲的所有初始化日志立即出现在日志框，
    # 后续任何 print() 也将实时路由到 GUI。
    _redirector.attach_signal(hub.signal_log.emit)

    # ── Step 4: 后台线程执行环境底噪校准（非阻塞，GUI 保持响应）─────────────
    # 标定期间的所有 print() 经 _redirector 实时显示在系统事件日志框。
    def _bg_calibrate():
        hub.signal_log.emit("=" * 58)
        hub.signal_log.emit("  [RF-Vision] 正在后台执行 S3 CAF-FFT 环境底噪校准...")
        hub.signal_log.emit("  校准期间可正常操作界面，结果将实时显示于此日志框")
        hub.signal_log.emit("=" * 58)
        try:
            from rf_zynq.calibrate_s3 import main as _calibrate
            _calibrate()   # 内部所有 print() 已被 _redirector 捕获并实时显示
            hub.signal_log.emit("  [RF-Vision] ✓ 底噪校准完成，新阈值已生效，系统就绪。")
        except Exception as _e:
            hub.signal_log.emit(f"  [RF-Vision] ⚠ 校准出错: {_e}")
            hub.signal_log.emit("  → 将使用上次保存的阈值继续运行。")

    threading.Thread(target=_bg_calibrate, daemon=True).start()

    # ── Step 5: 进入 Qt 主事件循环 ───────────────────────────────────────────
    exit_code = app.exec_()

    # ── Step 6: 恢复标准流并关闭子系统 ──────────────────────────────────────
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    hub.shutdown()
    sys.exit(exit_code)


# ==============================================================================
# [DEBUG ONLY 临时隔离运行区块]: 关闭主控时自动清理历史调试污染数据
# 注意：在完成所有早期开发与算法测试后，请直接注释下方的 atexit 注册块。
# 本代码块确保即使主进程被异常强杀（Ctrl+C 或报错闪退），也能坚定触发垃圾强制回收！
# ==============================================================================
""" import atexit
import subprocess
import os

def _auto_clean_debug_traces():
    print("\n[HOOK] 主控引擎开始降下帷幕，正在自动拉起清道夫 (clean_debug_data.bat)...")
    bat_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "clean_debug_data.bat")
    
    if os.path.exists(bat_path):
        try:
            # 采用静默的 shell 衍生执行，防止终端阻塞
            subprocess.run(f'cmd /c "{bat_path}"', shell=True)
            print("[HOOK] 数据库垃圾强制排空完毕，主控安全释放所有内存！")
        except Exception as e:
            print(f"[!] 清除脚本应急调用断帧故障: {e}")

atexit.register(_auto_clean_debug_traces) """
# ==============================================================================
