# -*- coding: utf-8 -*-
"""
mock_transmitter/uav_tx_gui.py — PlutoSDR 无人机射频靶机上位机
==============================================================
基于 PyQt5 构建的图形控制界面，提供全参数化的 PlutoSDR 发射机操控能力。

功能概览：
  ▸ 无人机型号/带宽选择（DRONE_CATALOG 中全部机型）
  ▸ 发射频点配置（三扇区 5.8GHz OcuSync 对齐接收端 SWEEP_SECTORS）
  ▸ 跳频模式选择（单频/三扇区跳频）
  ▸ 硬件增益调节（滑块 -89 dB → 0 dB）
  ▸ PlutoSDR 连接地址配置
  ▸ 实时发射功率估计（基于 IQ 均方根值）
  ▸ 滚动日志面板
  ▸ 一键发射/停止，连接状态 LED 指示

运行方式：
  python mock_transmitter/uav_tx_gui.py

依赖：PyQt5, numpy, scipy, pyadi-iio (pip install pyadi-iio)
"""

import sys
import os
import time
import threading
import numpy as np

# 将项目根目录加入路径，以便跨目录导入
_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGroupBox, QLabel, QComboBox, QSlider, QPushButton, QTextEdit,
    QLineEdit, QCheckBox, QDoubleSpinBox, QSpinBox, QSplitter,
    QFrame, QProgressBar, QStatusBar, QSizePolicy,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QObject
from PyQt5.QtGui import QFont, QColor, QPalette, QIcon, QTextCursor

# 延迟导入，避免在没有 pyadi-iio 的环境下启动失败
from mock_transmitter.dataset_replayer import (
    DRONE_CATALOG, _find_iq_files, _load_and_resample_segment, CHUNK_SAMPLES,
)
from mock_transmitter.pluto_hw_interface import PlutoTxInterface, normalize_iq_for_pluto

# ──────────────────────────────────────────────────────────────────────────────
# 系统预置参数（对齐接收端 config.py）
# ──────────────────────────────────────────────────────────────────────────────
DEFAULT_PLUTO_URI    = "ip:192.168.31.20"
DEFAULT_GAIN_DB      = -10.0
DEFAULT_FS_MHZ       = 40.0
DEFAULT_DWELL_MS     = 2000.0    # 每扇区驻留 2 秒（原 50 ms，过快易卡死）
DEFAULT_DITHER_RATIO = 0.10

# 接收端对齐的三个 5.8GHz OcuSync 扇区频点（单位 MHz）
SECTOR_FREQS_MHZ = [5745.0, 5785.0, 5825.0]
SECTOR_LABELS    = [
    "扇区A  5745 MHz",
    "扇区B  5785 MHz",
    "扇区C  5825 MHz",
]

# ──────────────────────────────────────────────────────────────────────────────
# 调色板（深色主题）
# ──────────────────────────────────────────────────────────────────────────────
STYLE_SHEET = """
/* ── 全局基础 ────────────────────────────────────────────── */
QMainWindow, QWidget {
    background-color: #05070f;
    color: #e2e8f0;
    font-family: 'Segoe UI', 'Microsoft YaHei UI', sans-serif;
    font-size: 13px;
}

/* ── QLabel 透明背景（Linux Qt5 兼容）──────────────────────── */
QLabel {
    background-color: transparent;
}

/* ── 分组框 ──────────────────────────────────────────────── */
QGroupBox {
    border: 1px solid #1e2843;
    border-radius: 10px;
    margin-top: 14px;
    padding: 10px 14px 14px 14px;
    background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
        stop:0 #0d1117, stop:1 #0a0d15);
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 14px;
    padding: 0 8px;
    color: #4f8ef7;
    font-weight: 700;
    font-size: 11px;
}

/* ── 下拉框 ──────────────────────────────────────────────── */
QComboBox {
    background-color: #0d1421;
    border: 1px solid #1e2843;
    border-radius: 6px;
    padding: 5px 12px;
    color: #e2e8f0;
    min-height: 30px;
    selection-background-color: #1e3a6e;
}
QComboBox::drop-down { border: none; width: 24px; }
QComboBox:hover { border-color: #2a4070; }
QComboBox QAbstractItemView {
    background-color: #0d1421;
    border: 1px solid #1e2843;
    selection-background-color: #1e3a6e;
    color: #e2e8f0;
    outline: none;
}

/* ── 输入框 ──────────────────────────────────────────────── */
QLineEdit, QDoubleSpinBox, QSpinBox {
    background-color: #0d1421;
    border: 1px solid #1e2843;
    border-radius: 6px;
    padding: 5px 12px;
    color: #e2e8f0;
    min-height: 30px;
}
QLineEdit:focus, QDoubleSpinBox:focus, QSpinBox:focus {
    border: 1px solid #4f8ef7;
    background-color: #0f1a2e;
}

/* ── 滑块 ────────────────────────────────────────────────── */
QSlider::groove:horizontal {
    height: 5px;
    background: #1e2843;
    border-radius: 2px;
}
QSlider::handle:horizontal {
    background: #4f8ef7;
    border: 2px solid #2a5ab0;
    width: 16px;
    height: 16px;
    margin: -6px 0;
    border-radius: 9px;
}
QSlider::handle:horizontal:hover { background: #6ba3ff; }
QSlider::sub-page:horizontal {
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
        stop:0 #1a3680, stop:1 #4f8ef7);
    border-radius: 2px;
}

/* ── 按钮 ────────────────────────────────────────────────── */
QPushButton {
    border-radius: 7px;
    padding: 8px 20px;
    font-weight: 700;
    font-size: 13px;
    min-height: 36px;
    border: none;
}
QPushButton#btn_start {
    background: qlineargradient(x1:0,y1:0,x2:1,y2:1,
        stop:0 #166534, stop:1 #22c55e);
    color: #f0fff4;
}
QPushButton#btn_start:hover { background: #16a34a; }
QPushButton#btn_start:pressed { background: #15803d; }
QPushButton#btn_stop {
    background: qlineargradient(x1:0,y1:0,x2:1,y2:1,
        stop:0 #7f1d1d, stop:1 #ef4444);
    color: #fff5f5;
}
QPushButton#btn_stop:hover { background: #dc2626; }
QPushButton#btn_stop:pressed { background: #b91c1c; }
QPushButton#btn_stop:disabled { background: #1c2030; color: #3d4560; }
QPushButton#btn_start:disabled { background: #1c2030; color: #3d4560; }
QPushButton#btn_connect {
    background: #0d1421;
    border: 1px solid #4f8ef7;
    color: #7bb0ff;
}
QPushButton#btn_connect:hover {
    background: #1a2a4a;
    border-color: #6ba3ff;
    color: #9bc8ff;
}

/* ── 日志文本框 ──────────────────────────────────────────── */
QTextEdit {
    background-color: #020409;
    border: 1px solid #1e2843;
    border-radius: 6px;
    color: #4ade80;
    font-family: 'Cascadia Code', 'Consolas', 'Courier New', monospace;
    font-size: 12px;
    padding: 8px;
    selection-background-color: #1e3a6e;
}

/* ── 复选框 ──────────────────────────────────────────────── */
QCheckBox {
    spacing: 10px;
    color: #94a3b8;
}
QCheckBox::indicator {
    width: 16px;
    height: 16px;
    border-radius: 4px;
    border: 1px solid #2a3a5e;
    background: #0d1421;
}
QCheckBox::indicator:checked {
    background: qlineargradient(x1:0,y1:0,x2:1,y2:1,
        stop:0 #1a3680, stop:1 #4f8ef7);
    border: none;
}

/* ── 进度条 ──────────────────────────────────────────────── */
QProgressBar {
    background-color: #0d1421;
    border: 1px solid #1e2843;
    border-radius: 4px;
    text-align: center;
    color: #94a3b8;
    height: 12px;
}
QProgressBar::chunk {
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
        stop:0 #1a3680, stop:0.6 #4f8ef7, stop:1 #ef4444);
    border-radius: 3px;
}

/* ── LED 标签 ────────────────────────────────────────────── */
QLabel#label_led_on  { color: #22c55e; font-size: 22px; }
QLabel#label_led_off { color: #2d3550; font-size: 22px; }
QLabel#label_tx_on   { color: #ef4444; font-size: 22px; }

/* ── 状态栏 ──────────────────────────────────────────────── */
QStatusBar {
    background-color: #020409;
    color: #475569;
    border-top: 1px solid #1e2843;
    font-size: 11px;
    padding: 2px 8px;
}

/* ── 分割器 ──────────────────────────────────────────────── */
QSplitter::handle { background: #1e2843; }

/* ── 滚动条 ──────────────────────────────────────────────── */
QScrollBar:vertical {
    background: #0d1117; width: 6px; border-radius: 3px;
}
QScrollBar::handle:vertical {
    background: #2a3a6e; border-radius: 3px; min-height: 30px;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
"""


# ──────────────────────────────────────────────────────────────────────────────
# 工作线程：发射控制（在独立线程中执行硬件操作，避免 GUI 卡顿）
# ──────────────────────────────────────────────────────────────────────────────
class TxWorker(QObject):
    """
    发射控制工作对象（运行在独立 QThread 中）。

    信号：
      sig_log       : 日志字符串
      sig_connected : 连接状态变更（bool）
      sig_tx_state  : 发射状态变更（bool）
      sig_rms_db    : 当前 IQ 均方根功率（dBFS）
    """
    sig_log       = pyqtSignal(str)
    sig_connected = pyqtSignal(bool)
    sig_tx_state  = pyqtSignal(bool)
    sig_rms_db    = pyqtSignal(float)
    sig_sector    = pyqtSignal(int)   # 当前跳频扇区索引

    def __init__(self):
        super().__init__()
        self._hw: PlutoTxInterface | None = None
        self._stop_ev = threading.Event()
        self._params  = {}

    # ------------------------------------------------------------------
    def configure(self, params: dict):
        """接收来自 GUI 的参数字典，在启动前调用。"""
        self._params = params

    # ------------------------------------------------------------------
    def do_connect(self):
        """建立与 PlutoSDR 的连接（由主线程槽函数调用）。"""
        p = self._params
        self._hw = PlutoTxInterface(
            uri      = p.get("uri",      DEFAULT_PLUTO_URI),
            fs       = p.get("fs",       DEFAULT_FS_MHZ) * 1e6,
            bw       = p.get("bw_mhz",  20.0) * 1e6,
            lo_hz    = p.get("lo_hz",   SECTOR_FREQS_MHZ[0] * 1e6),
            gain_db  = p.get("gain_db", DEFAULT_GAIN_DB),
            log_cb   = lambda m: self.sig_log.emit(m),
        )
        ok = self._hw.connect()
        self.sig_connected.emit(ok)

    def do_disconnect(self):
        """断开连接（由主线程槽函数调用）。"""
        if self._hw:
            self._hw.disconnect()
            self._hw = None
        self.sig_connected.emit(False)
        self.sig_tx_state.emit(False)

    # ------------------------------------------------------------------
    def do_start_tx(self):
        """
        启动重放发射循环（运行在工作线程中）。

        两阶段设计：
          · 预加载阶段：启动前一次性读取并重采样数据集的前 N 个 Chunk
                       （规避热路径中的磁盘IO和 resample_poly CPU计算阻塞）
          · 发射循环：纯 libiio 调用 + sleep，无任何阻塞操作
        """
        if self._hw is None or not self._hw.is_connected:
            self.sig_log.emit("[Worker] 错误：发射机未连接，无法启动。")
            return

        p            = self._params
        catalog_key  = p.get("catalog_key",  list(DRONE_CATALOG.keys())[0])
        hop_sectors  = p.get("hop_sectors",  [SECTOR_FREQS_MHZ[0] * 1e6])
        dwell_ms     = p.get("dwell_ms",     DEFAULT_DWELL_MS)
        dither_ratio = p.get("dither_ratio", DEFAULT_DITHER_RATIO)

        self._stop_ev.clear()

        info     = DRONE_CATALOG[catalog_key]
        iq_files = _find_iq_files(info["path"], info["pack"])
        if not iq_files:
            self.sig_log.emit(f"[Worker] 错误：未找到 IQ 文件（{info['path']}）")
            self.sig_tx_state.emit(False)
            return

        # ══ 阶段一：预加载 IQ 数据块（启动前一次性完成磁盘IO+重采样）══════════
        n_preload = min(len(iq_files), 8)
        self.sig_log.emit(
            f"[TX] 预加载 {catalog_key}（{n_preload} 个块，请稍候）..."
        )
        preloaded: list = []
        for i in range(n_preload):
            if self._stop_ev.is_set():
                self.sig_tx_state.emit(False)
                return
            try:
                chunk = _load_and_resample_segment(
                    iq_files[i], offset_samples=0, n_samples=CHUNK_SAMPLES
                )
                if len(chunk) > 0:
                    preloaded.append(normalize_iq_for_pluto(chunk))
                    self.sig_log.emit(f"[TX] 块 {i+1}/{n_preload} 重采样完成")
            except Exception as e:
                self.sig_log.emit(f"[TX] 块 {i+1} 加载失败：{e}")

        if not preloaded:
            self.sig_log.emit("[TX] 预加载全部失败，无可用 IQ 数据。")
            self.sig_tx_state.emit(False)
            return

        # 预计算每块 RMS（dBFS），避免在热路径中重复计算
        rms_cache = [
            20.0 * np.log10(
                float(np.sqrt(np.mean(np.abs(b) ** 2))) / 32767.0 + 1e-12
            )
            for b in preloaded
        ]
        avg_rms = sum(rms_cache) / len(rms_cache)
        self.sig_log.emit(
            f"[TX] 预加载完成：{len(preloaded)} 块就绪，"
            f"平均 RMS={avg_rms:.1f} dBFS，开始发射..."
        )
        self.sig_tx_state.emit(True)

        # ══ 阶段二：发射循环（热路径：纯 libiio + sleep，零磁盘IO）══════════
        n_sectors  = len(hop_sectors)
        sector_idx = 0
        chunk_idx  = 0

        while not self._stop_ev.is_set():
            iq_tx      = preloaded[chunk_idx % len(preloaded)]
            rms_db     = rms_cache[chunk_idx % len(rms_cache)]
            current_lo = hop_sectors[sector_idx % n_sectors]

            # 切换 LO 频率（push_cyclic 内部已处理旧缓冲的销毁）
            self._hw.set_lo(current_lo)
            self.sig_sector.emit(sector_idx % n_sectors)

            # 装载 Cyclic Buffer，FPGA 自主循环播放
            try:
                self._hw.push_cyclic(iq_tx)
            except Exception as e:
                self.sig_log.emit(f"[Worker] 缓冲压入异常：{e}")
                time.sleep(1.0)
                continue

            # 上报 RMS（仅在扇区切换时记一条日志，避免日志洪泛）
            self.sig_rms_db.emit(rms_db)
            if n_sectors > 1 or chunk_idx == 0:
                self.sig_log.emit(
                    f"[TX] → 扇区 {sector_idx % n_sectors + 1}/{n_sectors}  "
                    f"{current_lo/1e6:.1f} MHz  RMS={rms_db:.1f} dBFS"
                )

            # 等待驻留时间：FPGA 自主循环，Python 线程仅轮询停止事件
            jitter  = np.random.uniform(-dither_ratio, dither_ratio)
            dwell_s = dwell_ms / 1000.0 * (1.0 + jitter)
            t_end   = time.monotonic() + dwell_s
            while time.monotonic() < t_end and not self._stop_ev.is_set():
                time.sleep(0.05)

            chunk_idx  += 1
            sector_idx += 1

        self.sig_tx_state.emit(False)
        self.sig_log.emit("[TX] 发射已停止。")

    def request_stop(self):
        """
        线程安全的停止请求（可从任意线程直接调用，无需经过 Qt 信号队列）。

        问题根因：do_start_tx() 在工作线程中长期阻塞运行，
        Qt 队列连接无法投递信号到正在阻塞的线程。
        threading.Event.set() 是原子操作，直接调用安全。
        """
        self._stop_ev.set()

    def do_stop_tx(self):
        """Qt 信号连接备用（由 request_stop 内部调用）。"""
        self._stop_ev.set()

    def update_gain(self, gain_db: float):
        """实时更新 TX 增益（无需重启发射）。"""
        if self._hw and self._hw.is_connected:
            self._hw.set_gain(gain_db)


# ──────────────────────────────────────────────────────────────────────────────
# 主窗口
# ──────────────────────────────────────────────────────────────────────────────
class UAVTransmitterWindow(QMainWindow):
    """PlutoSDR 无人机射频靶机上位机主窗口。"""

    # 触发工作线程操作的信号
    _sig_do_connect    = pyqtSignal()
    _sig_do_disconnect = pyqtSignal()
    _sig_do_start      = pyqtSignal()
    _sig_do_stop       = pyqtSignal()
    _sig_update_gain   = pyqtSignal(float)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("⚡ PlutoSDR  ·  无人机射频靶机控制台  v2.0")
        self.setMinimumSize(1000, 720)
        self.setStyleSheet(STYLE_SHEET)

        # ── 工作线程 ──────────────────────────────────────────────────
        self._worker_thread = QThread(self)
        self._worker        = TxWorker()
        self._worker.moveToThread(self._worker_thread)

        self._worker.sig_log.connect(self._append_log)
        self._worker.sig_connected.connect(self._on_connected)
        self._worker.sig_tx_state.connect(self._on_tx_state)
        self._worker.sig_rms_db.connect(self._on_rms_update)
        self._worker.sig_sector.connect(self._on_sector_changed)

        self._sig_do_connect.connect(self._worker.do_connect)
        self._sig_do_disconnect.connect(self._worker.do_disconnect)
        self._sig_do_start.connect(self._worker.do_start_tx)
        self._sig_do_stop.connect(self._worker.do_stop_tx)
        self._sig_update_gain.connect(self._worker.update_gain)

        self._worker_thread.start()

        # 内部状态
        self._connected  = False
        self._tx_running = False

        # ── 日志缓冲：sig_log 高频信号 → 队列 → QTimer 250ms 批量刷新到 QTextEdit ──
        self._log_buffer: list = []
        self._log_timer = QTimer(self)
        self._log_timer.setInterval(250)
        self._log_timer.timeout.connect(self._flush_log)
        self._log_timer.start()

        # ── 构建 UI ───────────────────────────────────────────────────
        self._build_ui()
        self._refresh_sector_highlight(0)

    # ==================================================================
    # UI 构建
    # ==================================================================
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QHBoxLayout(central)
        root_layout.setContentsMargins(16, 16, 16, 16)
        root_layout.setSpacing(16)

        # 左侧控制面板
        left_panel = QWidget()
        left_panel.setFixedWidth(420)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(14)
        left_layout.setContentsMargins(0, 0, 0, 0)

        left_layout.addWidget(self._build_connection_group())
        left_layout.addWidget(self._build_drone_group())
        left_layout.addWidget(self._build_freq_group())
        left_layout.addWidget(self._build_gain_group())
        left_layout.addWidget(self._build_hopping_group())
        left_layout.addWidget(self._build_action_group())
        left_layout.addStretch()

        # 右侧日志面板
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(10)
        right_layout.addWidget(self._build_status_panel())
        right_layout.addWidget(self._build_log_panel())

        root_layout.addWidget(left_panel)
        root_layout.addWidget(right_panel, stretch=1)

        # 状态栏
        self.statusBar().showMessage("就绪 · 请先配置参数并连接 PlutoSDR")

    # ------------------------------------------------------------------
    def _build_connection_group(self) -> QGroupBox:
        grp = QGroupBox("硬件连接")
        lay = QHBoxLayout(grp)
        lay.setSpacing(8)

        self._edit_uri = QLineEdit(DEFAULT_PLUTO_URI)
        self._edit_uri.setPlaceholderText("ip:192.168.x.x")

        self._btn_connect = QPushButton("连接")
        self._btn_connect.setObjectName("btn_connect")
        self._btn_connect.setFixedWidth(80)
        self._btn_connect.clicked.connect(self._on_btn_connect)

        self._lbl_led = QLabel("●")
        self._lbl_led.setObjectName("label_led_off")
        self._lbl_led.setFixedWidth(24)
        self._lbl_led.setAlignment(Qt.AlignCenter)

        lay.addWidget(QLabel("地址："))
        lay.addWidget(self._edit_uri, stretch=1)
        lay.addWidget(self._btn_connect)
        lay.addWidget(self._lbl_led)
        return grp

    def _build_drone_group(self) -> QGroupBox:
        grp = QGroupBox("无人机型号")
        lay = QVBoxLayout(grp)

        self._combo_drone = QComboBox()
        for key in DRONE_CATALOG:
            self._combo_drone.addItem(key)

        lay.addWidget(self._combo_drone)
        return grp

    def _build_freq_group(self) -> QGroupBox:
        grp = QGroupBox("发射频点  (5.8 GHz OcuSync)")
        lay = QVBoxLayout(grp)
        lay.setSpacing(6)

        self._sector_labels: list[QLabel] = []
        for i, lbl in enumerate(SECTOR_LABELS):
            row = QHBoxLayout()
            dot = QLabel("◆")
            dot.setFixedWidth(18)
            dot.setAlignment(Qt.AlignCenter)
            self._sector_labels.append(dot)
            row.addWidget(dot)
            row.addWidget(QLabel(lbl))
            lay.addLayout(row)

        return grp

    def _build_gain_group(self) -> QGroupBox:
        grp = QGroupBox("TX 硬件增益")
        lay = QVBoxLayout(grp)

        h = QHBoxLayout()
        self._lbl_gain_val = QLabel(f"{DEFAULT_GAIN_DB:.0f} dB")
        self._lbl_gain_val.setFixedWidth(60)
        self._lbl_gain_val.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        h.addWidget(QLabel("-89 dB"))
        self._slider_gain = QSlider(Qt.Horizontal)
        self._slider_gain.setRange(-89, 0)
        self._slider_gain.setValue(int(DEFAULT_GAIN_DB))
        self._slider_gain.setTickInterval(10)
        self._slider_gain.valueChanged.connect(self._on_gain_changed)
        h.addWidget(self._slider_gain, stretch=1)
        h.addWidget(QLabel("0 dB"))
        lay.addLayout(h)
        lay.addWidget(self._lbl_gain_val, alignment=Qt.AlignHCenter)
        return grp

    def _build_hopping_group(self) -> QGroupBox:
        grp = QGroupBox("跳频参数")
        lay = QVBoxLayout(grp)
        lay.setSpacing(8)

        # 跳频开关
        self._chk_hopping = QCheckBox("启用三扇区跳频（5745 / 5785 / 5825 MHz）")
        self._chk_hopping.setChecked(True)
        self._chk_hopping.stateChanged.connect(self._on_hopping_toggled)
        lay.addWidget(self._chk_hopping)

        # 单频扇区选择（仅关闭跳频时显示）
        self._row_single_sector = QHBoxLayout()
        self._row_single_sector.addWidget(QLabel("发射扇区："))
        self._combo_sector = QComboBox()
        for lbl in SECTOR_LABELS:
            self._combo_sector.addItem(lbl)
        self._row_single_sector.addWidget(self._combo_sector)
        lay.addLayout(self._row_single_sector)
        self._set_single_sector_visible(False)   # 默认跳频模式，隐藏桇区选择框

        # 驻留时间
        h1 = QHBoxLayout()
        h1.addWidget(QLabel("扇区驻留时长："))
        self._spin_dwell = QDoubleSpinBox()
        self._spin_dwell.setRange(500.0, 60000.0)
        self._spin_dwell.setValue(DEFAULT_DWELL_MS)
        self._spin_dwell.setSuffix("  ms")
        self._spin_dwell.setSingleStep(500.0)
        h1.addWidget(self._spin_dwell)
        lay.addLayout(h1)

        # 抖动比例
        h2 = QHBoxLayout()
        h2.addWidget(QLabel("时序随机抖动："))
        self._spin_dither = QDoubleSpinBox()
        self._spin_dither.setRange(0.0, 0.5)
        self._spin_dither.setValue(DEFAULT_DITHER_RATIO)
        self._spin_dither.setSuffix("  (±比例)")
        self._spin_dither.setSingleStep(0.05)
        h2.addWidget(self._spin_dither)
        lay.addLayout(h2)

        return grp

    def _set_single_sector_visible(self, visible: bool):
        """Show/hide the single-sector combo row."""
        for i in range(self._row_single_sector.count()):
            w = self._row_single_sector.itemAt(i).widget()
            if w:
                w.setVisible(visible)

    def _on_hopping_toggled(self, state):
        hopping = (state == Qt.Checked)
        self._set_single_sector_visible(not hopping)

    def _build_action_group(self) -> QGroupBox:
        grp = QGroupBox("发射控制")
        lay = QVBoxLayout(grp)
        lay.setSpacing(10)

        h = QHBoxLayout()
        self._btn_start = QPushButton("▶  开始发射")
        self._btn_start.setObjectName("btn_start")
        self._btn_start.setEnabled(False)
        self._btn_start.clicked.connect(self._on_btn_start)

        self._btn_stop = QPushButton("■  停止发射")
        self._btn_stop.setObjectName("btn_stop")
        self._btn_stop.setEnabled(False)
        self._btn_stop.clicked.connect(self._on_btn_stop)

        h.addWidget(self._btn_start)
        h.addWidget(self._btn_stop)
        lay.addLayout(h)
        return grp

    def _build_status_panel(self) -> QGroupBox:
        grp = QGroupBox("实时状态")
        lay = QVBoxLayout(grp)
        lay.setSpacing(10)

        # TX 状态指示
        h_tx = QHBoxLayout()
        h_tx.addWidget(QLabel("发射状态："))
        self._lbl_tx_led = QLabel("●  待机")
        self._lbl_tx_led.setObjectName("label_led_off")
        self._lbl_tx_led.setFont(QFont("Segoe UI", 13, QFont.Bold))
        h_tx.addWidget(self._lbl_tx_led)
        h_tx.addStretch()
        lay.addLayout(h_tx)

        # 当前频点
        h_freq = QHBoxLayout()
        h_freq.addWidget(QLabel("当前频点："))
        self._lbl_current_freq = QLabel("—")
        self._lbl_current_freq.setFont(QFont("Consolas", 13, QFont.Bold))
        self._lbl_current_freq.setStyleSheet("color: #5c7cfa;")
        h_freq.addWidget(self._lbl_current_freq)
        h_freq.addStretch()
        lay.addLayout(h_freq)

        # RMS 功率计
        h_rms = QHBoxLayout()
        h_rms.addWidget(QLabel("TX 功率 (dBFS)："))
        self._lbl_rms = QLabel("—  dBFS")
        self._lbl_rms.setFont(QFont("Consolas", 12))
        self._lbl_rms.setStyleSheet("color: #f39c12;")
        h_rms.addWidget(self._lbl_rms)
        h_rms.addStretch()
        lay.addLayout(h_rms)

        self._progress_rms = QProgressBar()
        self._progress_rms.setRange(0, 100)
        self._progress_rms.setValue(0)
        self._progress_rms.setFormat("")
        lay.addWidget(self._progress_rms)

        return grp

    def _build_log_panel(self) -> QGroupBox:
        grp = QGroupBox("系统日志")
        lay = QVBoxLayout(grp)
        self._log_box = QTextEdit()
        self._log_box.setReadOnly(True)
        self._log_box.setMinimumHeight(200)
        lay.addWidget(self._log_box)

        btn_clear = QPushButton("清空日志")
        btn_clear.setObjectName("btn_connect")
        btn_clear.setFixedWidth(100)
        btn_clear.clicked.connect(self._log_box.clear)
        lay.addWidget(btn_clear, alignment=Qt.AlignRight)
        return grp

    # ==================================================================
    # 槽函数 / 事件处理
    # ==================================================================
    def _on_btn_connect(self):
        if not self._connected:
            self._push_params_to_worker()
            self._btn_connect.setEnabled(False)
            self._btn_connect.setText("连接中...")
            self._sig_do_connect.emit()
        else:
            # 直接调用 request_stop()（线程安全），绕过队列信号避免死锁
            if self._tx_running:
                self._worker.request_stop()
            self._sig_do_disconnect.emit()

    def _on_connected(self, ok: bool):
        self._connected = ok
        self._btn_connect.setEnabled(True)
        if ok:
            self._btn_connect.setText("断开")
            self._lbl_led.setObjectName("label_led_on")
            self._lbl_led.setText("●")
            self._lbl_led.setStyleSheet("color: #27ae60; font-size: 22px;")
            self._btn_start.setEnabled(True)
            self.statusBar().showMessage("PlutoSDR 已连接 · 可以开始发射")
        else:
            self._btn_connect.setText("连接")
            self._lbl_led.setStyleSheet("color: #555; font-size: 22px;")
            self._btn_start.setEnabled(False)
            self._btn_stop.setEnabled(False)
            self.statusBar().showMessage("PlutoSDR 已断开")

    def _on_tx_state(self, running: bool):
        self._tx_running = running
        if running:
            self._lbl_tx_led.setText("●  发射中")
            self._lbl_tx_led.setStyleSheet("color: #e74c3c; font-size: 13px; font-weight: bold;")
            self._btn_start.setEnabled(False)
            self._btn_stop.setEnabled(True)
            self.statusBar().showMessage("⚡ 正在向空中辐射射频信号...")
        else:
            self._lbl_tx_led.setText("●  待机")
            self._lbl_tx_led.setStyleSheet("color: #555; font-size: 13px; font-weight: bold;")
            self._btn_start.setEnabled(self._connected)
            self._btn_stop.setEnabled(False)
            self._lbl_current_freq.setText("—")
            self._lbl_rms.setText("—  dBFS")
            self._progress_rms.setValue(0)
            self.statusBar().showMessage("发射已停止")

    def _on_btn_start(self):
        self._push_params_to_worker()
        self._sig_do_start.emit()

    def _on_btn_stop(self):
        """
        停止发射按钮处理器。

        直接调用 request_stop()（threading.Event.set() 原子操作），
        绕过 Qt 队列信号机制——因为工作线程正在 do_start_tx() 中阻塞，
        队列信号无法投递，必须从主线程直接访问。
        """
        self._worker.request_stop()
        self._btn_stop.setEnabled(False)   # 立即禁用，给用户即时反馈

    def _on_gain_changed(self, val: int):
        self._lbl_gain_val.setText(f"{val} dB")
        if self._connected:
            self._sig_update_gain.emit(float(val))

    def _on_rms_update(self, rms_db: float):
        self._lbl_rms.setText(f"{rms_db:.1f}  dBFS")
        # dBFS ∈ [-60, 0]，映射至进度条 [0, 100]
        pct = int(np.clip((rms_db + 60.0) / 60.0 * 100.0, 0, 100))
        self._progress_rms.setValue(pct)

    def _on_sector_changed(self, idx: int):
        freq_mhz = SECTOR_FREQS_MHZ[idx]
        self._lbl_current_freq.setText(f"{freq_mhz:.1f} MHz")
        self._refresh_sector_highlight(idx)

    def _refresh_sector_highlight(self, active_idx: int):
        colors = ["#5c7cfa", "#27ae60", "#f39c12"]
        for i, dot in enumerate(self._sector_labels):
            if i == active_idx:
                dot.setStyleSheet(f"color: {colors[i]}; font-size: 16px;")
            else:
                dot.setStyleSheet("color: #333; font-size: 16px;")

    # ------------------------------------------------------------------
    def _push_params_to_worker(self):
        """将 GUI 当前参数同步至 TxWorker。"""
        hop_en = self._chk_hopping.isChecked()
        if hop_en:
            sectors_hz = [f * 1e6 for f in SECTOR_FREQS_MHZ]
        else:
            # 用户选定的单个扇区
            idx = self._combo_sector.currentIndex()
            sectors_hz = [SECTOR_FREQS_MHZ[idx] * 1e6]
        params = {
            "uri":          self._edit_uri.text().strip(),
            "fs":           DEFAULT_FS_MHZ,
            "bw_mhz":       20.0,
            "lo_hz":        sectors_hz[0],
            "gain_db":      float(self._slider_gain.value()),
            "catalog_key":  self._combo_drone.currentText(),
            "hop_sectors":  sectors_hz,
            "dwell_ms":     self._spin_dwell.value(),
            "dither_ratio": self._spin_dither.value(),
        }
        self._worker.configure(params)

    def _append_log(self, msg: str):
        """将日志消息压入缓冲队列，由 QTimer 批量刷新，避免每帧独立更新 QTextEdit。"""
        ts = time.strftime("%H:%M:%S")
        self._log_buffer.append(f"<span style='color:#555;'>[{ts}]</span> {msg}")
        if len(self._log_buffer) > 300:           # 防止缓冲无限增长
            self._log_buffer = self._log_buffer[-300:]

    def _flush_log(self):
        """QTimer 触发（250 ms），将缓冲日志批量写入 QTextEdit（在主线程执行）。"""
        if not self._log_buffer:
            return
        batch = "<br>".join(self._log_buffer)
        self._log_buffer.clear()
        self._log_box.append(batch)
        sb = self._log_box.verticalScrollBar()
        sb.setValue(sb.maximum())

    # ------------------------------------------------------------------
    def closeEvent(self, event):
        """
        关闭窗口：先请求停止发射（线程安全直接调用），
        再异步通知工作线程退出，最多等待 4 秒。
        """
        self._worker.request_stop()          # 立即设置停止标志（不经过队列信号）
        self._log_timer.stop()               # 停止日志刷新定时器
        self._sig_do_disconnect.emit()       # 异步通知硬件安全关停
        self._worker_thread.quit()           # 请求工作线程退出事件循环
        self._worker_thread.wait(4000)       # 最多等待 4 秒
        super().closeEvent(event)


# ──────────────────────────────────────────────────────────────────────────────
# 入口
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = UAVTransmitterWindow()
    win.show()
    sys.exit(app.exec_())
