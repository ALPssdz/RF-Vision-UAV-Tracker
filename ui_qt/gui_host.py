# -*- coding: utf-8 -*-
"""
ui_qt/gui_host.py — RF-Vision 主上位机界面（现代化重构版）
==========================================================
基于 PyQt5 的深色玻璃拟态（Glassmorphism）风格主控界面。

架构层次：
  · 严格视图层（Strict View Layer）— 零业务逻辑，仅单向数据流渲染
  · 与 CentralHubEngine 通过 Qt 信号槽解耦，无直接调用

新增功能（v2.0）：
  · 香橙派 RK3588 系统资源实时监控面板
      ─ CPU 总体占用率（% + 动态进度环）
      ─ 网络上行带宽（KB/s）
      ─ 网络下行带宽（KB/s）
      ─ SSH 在线状态指示灯
  · 现代化玻璃拟态 + 霓虹渐变设计语言
  · 告警事件数量实时徽章
  · 响应式布局，底部状态栏增强
"""

import os
import cv2
import time
import numpy as np

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QPlainTextEdit, QTabWidget,
    QTableWidget, QTableWidgetItem, QHeaderView,
    QMessageBox, QShortcut, QFrame, QProgressBar,
    QSizePolicy, QGraphicsDropShadowEffect,
)
from PyQt5.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve, pyqtProperty
from PyQt5.QtGui import (QImage, QPixmap, QFont, QKeySequence,
                          QColor, QPalette, QPainter, QPen, QBrush,
                          QConicalGradient, QRadialGradient, QLinearGradient)
from database.db_manager import DBManager
from ui_qt.orangepi_monitor import OrangePiMonitor

# ──────────────────────────────────────────────────────────────────────────────
# 全局设计令牌
# ──────────────────────────────────────────────────────────────────────────────
PALETTE = {
    "bg_deep":     "#05070f",
    "bg_card":     "#0d1117",
    "bg_surface":  "#161b27",
    "bg_elevated": "#1c2336",
    "border":      "#252d45",
    "border_glow": "#2a3a6e",
    "accent_blue":  "#4f8ef7",
    "accent_cyan":  "#22d3ee",
    "accent_green": "#22c55e",
    "accent_amber": "#f59e0b",
    "accent_red":   "#ef4444",
    "accent_purple":"#a855f7",
    "text_primary": "#e2e8f0",
    "text_secondary": "#94a3b8",
    "text_muted":   "#475569",
}

GLOBAL_STYLESHEET = f"""
/* ── 全局基础 ─────────────────────────────────────────────────────────── */
QMainWindow, QWidget {{
    background-color: {PALETTE['bg_deep']};
    color: {PALETTE['text_primary']};
    font-family: 'Segoe UI', 'Microsoft YaHei UI', 'PingFang SC', sans-serif;
    font-size: 13px;
}}

/* ── 选项卡栏 ─────────────────────────────────────────────────────────── */
QTabWidget::pane {{
    background-color: {PALETTE['bg_card']};
    border: 1px solid {PALETTE['border']};
    border-top: none;
    border-radius: 0 0 10px 10px;
}}
QTabBar::tab {{
    background-color: {PALETTE['bg_surface']};
    color: {PALETTE['text_secondary']};
    padding: 12px 32px;
    font-size: 14px;
    font-weight: 600;
    border: 1px solid {PALETTE['border']};
    border-bottom: none;
    border-radius: 8px 8px 0 0;
    margin-right: 2px;
    min-width: 220px;
}}
QTabBar::tab:selected {{
    background-color: {PALETTE['bg_card']};
    color: {PALETTE['accent_blue']};
    border-bottom: 2px solid {PALETTE['accent_blue']};
}}
QTabBar::tab:hover:!selected {{
    background-color: {PALETTE['bg_elevated']};
    color: {PALETTE['text_primary']};
}}

/* ── 按钮 ─────────────────────────────────────────────────────────────── */
QPushButton {{
    border-radius: 7px;
    padding: 9px 22px;
    font-weight: 600;
    font-size: 13px;
    border: none;
    min-height: 36px;
}}
QPushButton#btn_start {{
    background: qlineargradient(x1:0,y1:0,x2:1,y2:1,
        stop:0 #166534, stop:1 #22c55e);
    color: #f0fff0;
}}
QPushButton#btn_start:hover {{ background: #16a34a; }}
QPushButton#btn_stop {{
    background: qlineargradient(x1:0,y1:0,x2:1,y2:1,
        stop:0 #7f1d1d, stop:1 #ef4444);
    color: #fff5f5;
}}
QPushButton#btn_stop:hover {{ background: #dc2626; }}
QPushButton#btn_secondary {{
    background-color: {PALETTE['bg_elevated']};
    color: {PALETTE['text_secondary']};
    border: 1px solid {PALETTE['border']};
}}
QPushButton#btn_secondary:hover {{
    background-color: {PALETTE['border_glow']};
    color: {PALETTE['text_primary']};
}}
QPushButton#btn_danger {{
    background: qlineargradient(x1:0,y1:0,x2:1,y2:1,
        stop:0 #7f1d1d, stop:1 #dc2626);
    color: white;
}}
QPushButton#btn_danger:hover {{ background: #b91c1c; }}

/* ── 卡片容器 ─────────────────────────────────────────────────────────── */
QFrame#card {{
    background-color: {PALETTE['bg_card']};
    border: 1px solid {PALETTE['border']};
    border-radius: 12px;
}}

/* ── 表格 ─────────────────────────────────────────────────────────────── */
QTableWidget {{
    background-color: {PALETTE['bg_card']};
    border: 1px solid {PALETTE['border']};
    border-radius: 8px;
    gridline-color: {PALETTE['bg_surface']};
    color: {PALETTE['text_primary']};
    selection-background-color: {PALETTE['bg_elevated']};
    alternate-background-color: {PALETTE['bg_surface']};
}}
QTableWidget::item {{ padding: 6px 12px; border: none; }}
QTableWidget::item:selected {{ background-color: {PALETTE['border_glow']}; color: {PALETTE['accent_blue']}; }}
QHeaderView::section {{
    background-color: {PALETTE['bg_elevated']};
    color: {PALETTE['text_secondary']};
    font-weight: 600;
    font-size: 12px;
    padding: 8px 12px;
    border: none;
    border-right: 1px solid {PALETTE['border']};
}}
QHeaderView::section:first {{ border-left: none; border-radius: 8px 0 0 0; }}
QHeaderView::section:last {{ border-right: none; border-radius: 0 8px 0 0; }}

/* ── 日志文本框 ────────────────────────────────────────────────────────── */
QPlainTextEdit {{
    background-color: #020409;
    border: 1px solid {PALETTE['border']};
    border-radius: 8px;
    color: #4ade80;
    font-family: 'Cascadia Code', 'Consolas', 'Courier New', monospace;
    font-size: 12px;
    padding: 8px;
    selection-background-color: {PALETTE['border_glow']};
}}

/* ── 进度条 ────────────────────────────────────────────────────────────── */
QProgressBar {{
    background-color: {PALETTE['bg_elevated']};
    border: none;
    border-radius: 5px;
    height: 8px;
    text-align: center;
    color: transparent;
}}
QProgressBar::chunk#chunk_green {{
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
        stop:0 #166534, stop:1 #22c55e);
    border-radius: 5px;
}}
QProgressBar::chunk#chunk_blue {{
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
        stop:0 #1e3a8a, stop:1 #4f8ef7);
    border-radius: 5px;
}}
QProgressBar::chunk#chunk_amber {{
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
        stop:0 #92400e, stop:1 #f59e0b);
    border-radius: 5px;
}}
QProgressBar::chunk#chunk_red {{
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
        stop:0 #7f1d1d, stop:1 #ef4444);
    border-radius: 5px;
}}

/* ── 状态栏 ────────────────────────────────────────────────────────────── */
QStatusBar {{
    background-color: {PALETTE['bg_card']};
    color: {PALETTE['text_muted']};
    border-top: 1px solid {PALETTE['border']};
    padding: 4px 12px;
    font-size: 11px;
}}

/* ── 滚动条 ────────────────────────────────────────────────────────────── */
QScrollBar:vertical {{
    background: {PALETTE['bg_card']};
    width: 6px;
    border-radius: 3px;
}}
QScrollBar::handle:vertical {{
    background: {PALETTE['border_glow']};
    border-radius: 3px;
    min-height: 30px;
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QScrollBar:horizontal {{
    background: {PALETTE['bg_card']};
    height: 6px;
    border-radius: 3px;
}}
QScrollBar::handle:horizontal {{
    background: {PALETTE['border_glow']};
    border-radius: 3px;
    min-width: 30px;
}}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; }}

/* ── 消息框 ────────────────────────────────────────────────────────────── */
QMessageBox {{
    background-color: {PALETTE['bg_card']};
    color: {PALETTE['text_primary']};
}}
QMessageBox QPushButton {{
    background-color: {PALETTE['bg_elevated']};
    border: 1px solid {PALETTE['border']};
    color: {PALETTE['text_primary']};
    padding: 6px 18px;
}}
QMessageBox QPushButton:hover {{
    background-color: {PALETTE['border_glow']};
}}
"""


# ──────────────────────────────────────────────────────────────────────────────
# 自定义组件：霓虹徽章标签
# ──────────────────────────────────────────────────────────────────────────────
class NeonBadge(QLabel):
    """带发光效果的状态徽章标签。"""
    def __init__(self, text: str = "", color: str = "#22c55e", parent=None):
        super().__init__(text, parent)
        self._color = color
        self.setAlignment(Qt.AlignCenter)
        self._apply_style()

    def _apply_style(self):
        glow = QGraphicsDropShadowEffect()
        glow.setBlurRadius(18)
        glow.setColor(QColor(self._color))
        glow.setOffset(0, 0)
        self.setGraphicsEffect(glow)
        self.setStyleSheet(f"""
            QLabel {{
                color: {self._color};
                font-weight: 700;
                font-size: 12px;
                background: transparent;
            }}
        """)

    def set_color(self, color: str):
        self._color = color
        self._apply_style()


# ──────────────────────────────────────────────────────────────────────────────
# 自定义组件：圆弧 CPU 占用仪表盘
# ──────────────────────────────────────────────────────────────────────────────
class CpuGaugeDial(QWidget):
    """
    圆弧形 CPU 占用率仪表盘组件。

    渲染规格：
      · 外圈：暗灰色背景弧（270° 扫描范围，起始角 -225°）
      · 内圈：渐变色进度弧（Qt 使用 1/16 度单位 → 角度 × 16）
      · 中心：百分比数值文本
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._value   = 0.0         # 0.0 ~ 100.0
        self._animated = 0.0
        self._timer   = QTimer(self)
        self._timer.setInterval(30)
        self._timer.timeout.connect(self._tick_animation)
        self.setMinimumSize(120, 120)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.setFixedSize(130, 130)

    def set_value(self, val: float):
        self._value = max(0.0, min(100.0, val))
        if not self._timer.isActive():
            self._timer.start()

    def _tick_animation(self):
        diff = self._value - self._animated
        self._animated += diff * 0.15   # 指数平滑
        if abs(diff) < 0.1:
            self._animated = self._value
            self._timer.stop()
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        w, h   = self.width(), self.height()
        margin = 12
        rect   = self.rect().adjusted(margin, margin, -margin, -margin)

        # 背景弧
        pen_bg = QPen(QColor(PALETTE['bg_elevated']), 10, Qt.SolidLine, Qt.RoundCap)
        painter.setPen(pen_bg)
        painter.drawArc(rect, -225 * 16, -270 * 16)   # 顺时针 270°

        # 颜色映射（CPU 占用率 → 渐变色）
        pct = self._animated / 100.0
        if pct < 0.5:
            r = int(pct * 2 * 245)
            g = 197
            b = 11
        else:
            r = 239
            g = int((1 - (pct - 0.5) * 2) * 178)
            b = 68
        arc_color = QColor(r, g, b)

        # 进度弧
        pen_fg = QPen(arc_color, 10, Qt.SolidLine, Qt.RoundCap)
        painter.setPen(pen_fg)
        span = int(-270 * 16 * pct)
        painter.drawArc(rect, -225 * 16, span)

        # 中心文字
        painter.setPen(QColor(PALETTE['text_primary']))
        painter.setFont(QFont("Segoe UI", 16, QFont.Bold))
        painter.drawText(rect, Qt.AlignCenter, f"{self._animated:.0f}%")

        # 小标签
        painter.setFont(QFont("Segoe UI", 8))
        painter.setPen(QColor(PALETTE['text_muted']))
        label_rect = rect.adjusted(0, rect.height() // 2 + 8, 0, 0)
        painter.drawText(label_rect, Qt.AlignCenter, "CPU")

        painter.end()


# ──────────────────────────────────────────────────────────────────────────────
# 自定义组件：网络速度迷你仪表卡
# ──────────────────────────────────────────────────────────────────────────────
class NetSpeedCard(QFrame):
    """
    显示单向网络速率的迷你卡片组件。

    参数：
      label   : 方向标签，如 "↑ 上行" 或 "↓ 下行"
      color   : 进度条/数值颜色（十六进制字符串）
      max_kbps: 速率满量程（KB/s），用于映射进度条
    """

    def __init__(self, label: str, color: str, max_kbps: float = 10240.0, parent=None):
        super().__init__(parent)
        self._max  = max_kbps
        self._color = color
        self.setObjectName("card")
        self.setFixedHeight(80)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 8, 12, 8)
        lay.setSpacing(4)

        # 标签行
        title_row = QHBoxLayout()
        lbl_dir = QLabel(label)
        lbl_dir.setStyleSheet(f"color: {PALETTE['text_secondary']}; font-size: 11px; font-weight: 600;")
        self._lbl_val = QLabel("0  KB/s")
        self._lbl_val.setStyleSheet(f"color: {color}; font-size: 16px; font-weight: 700;")
        self._lbl_val.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        title_row.addWidget(lbl_dir)
        title_row.addWidget(self._lbl_val)
        lay.addLayout(title_row)

        # 进度条
        self._bar = QProgressBar()
        self._bar.setRange(0, 1000)
        self._bar.setValue(0)
        self._bar.setFixedHeight(6)
        self._bar.setFormat("")
        lay.addWidget(self._bar)

    def update_speed(self, kbps: float):
        """更新显示的速率数值。"""
        # 格式化显示：< 1024 KB/s 显示 KB/s，否则显示 MB/s
        if kbps >= 1024.0:
            self._lbl_val.setText(f"{kbps/1024.0:.1f}  MB/s")
        else:
            self._lbl_val.setText(f"{kbps:.0f}  KB/s")

        # 对数映射：speed_bar = log10(kbps+1) / log10(max+1) × 1000
        import math
        raw = math.log10(kbps + 1.0) / math.log10(self._max + 1.0) * 1000.0
        self._bar.setValue(int(min(raw, 1000)))

        # 根据速率动态着色
        if kbps > self._max * 0.7:
            chunk_name = "chunk_red"
        elif kbps > self._max * 0.3:
            chunk_name = "chunk_amber"
        else:
            chunk_name = "chunk_blue"
        style = f"QProgressBar::chunk {{ background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 {self._color}80, stop:1 {self._color}); border-radius: 3px; }}"
        self._bar.setStyleSheet(style)


# ──────────────────────────────────────────────────────────────────────────────
# 状态徽章（顶部状态栏卡片）
# ──────────────────────────────────────────────────────────────────────────────
class StatusBannerCard(QFrame):
    """顶部水平状态栏中的单个状态卡片。"""

    def __init__(self, icon: str, title: str, color: str, parent=None):
        super().__init__(parent)
        self.setObjectName("card")
        self._color = color
        lay = QHBoxLayout(self)
        lay.setContentsMargins(16, 10, 16, 10)
        lay.setSpacing(10)

        # 图标/指示灯
        self._lbl_dot = QLabel("●")
        self._lbl_dot.setStyleSheet(f"color: {color}; font-size: 18px;")
        glow = QGraphicsDropShadowEffect()
        glow.setBlurRadius(12)
        glow.setColor(QColor(color))
        glow.setOffset(0, 0)
        self._lbl_dot.setGraphicsEffect(glow)

        # 文字块
        text_col = QVBoxLayout()
        text_col.setSpacing(0)
        self._lbl_icon = QLabel(icon)
        self._lbl_icon.setStyleSheet(f"color: {PALETTE['text_muted']}; font-size: 10px; font-weight: 600;")
        self._lbl_status = QLabel(title)
        self._lbl_status.setStyleSheet(f"color: {PALETTE['text_primary']}; font-size: 12px; font-weight: 600;")
        text_col.addWidget(self._lbl_icon)
        text_col.addWidget(self._lbl_status)

        lay.addWidget(self._lbl_dot)
        lay.addLayout(text_col)
        lay.addStretch()

        # 卡片边框颜色
        self.setStyleSheet(f"""
            QFrame#card {{
                background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
                    stop:0 {PALETTE['bg_elevated']}, stop:1 {PALETTE['bg_card']});
                border: 1px solid {PALETTE['border']};
                border-left: 3px solid {color};
                border-radius: 10px;
            }}
        """)

    def update_status(self, text: str, active: bool = True, color: str = None):
        self._lbl_status.setText(text)
        c = color or self._color
        dot_style = f"color: {c}; font-size: 18px;"
        if not active:
            dot_style = f"color: {PALETTE['text_muted']}; font-size: 18px;"
        self._lbl_dot.setStyleSheet(dot_style)
        # 刷新 glow
        glow = QGraphicsDropShadowEffect()
        glow.setBlurRadius(12 if active else 0)
        glow.setColor(QColor(c))
        glow.setOffset(0, 0)
        self._lbl_dot.setGraphicsEffect(glow)


# ──────────────────────────────────────────────────────────────────────────────
# 主窗口
# ──────────────────────────────────────────────────────────────────────────────
class MainWindow(QMainWindow):
    """
    RF-Vision 主上位机表现层（v2.0 现代化重构）。

    严格视图层：零业务逻辑，与 CentralHubEngine 通过 Qt 信号槽单向连接。
    """

    def __init__(self, hub=None):
        super().__init__()
        self.setWindowTitle("RF-Vision  ·  无人机多模态探测系统  v2.0")
        self.resize(1680, 980)
        self.setStyleSheet(GLOBAL_STYLESHEET)

        self.hub = hub
        self.db_engine = DBManager()   # 仅只读挂载点

        # 构建中央控件
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── 顶部标题栏 ────────────────────────────────────────────────
        root.addWidget(self._build_titlebar())

        # ── 选项卡体 ─────────────────────────────────────────────────
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(False)
        root.addWidget(self.tabs, stretch=1)

        # Tab 1：实时监测数据流
        self.tab1 = QWidget()
        self.setup_live_dashboard()
        self.tabs.addTab(self.tab1, "  ◈  监测数据流")

        # Tab 2：告警日志库
        self.tab2 = QWidget()
        self.setup_evidence_database()
        self.tabs.addTab(self.tab2, "  ▦  告警日志库")

        self.tabs.currentChanged.connect(self.on_tab_changed)

        # ── 状态栏 ──────────────────────────────────────────────────
        self.statusBar().showMessage(
            f"RF-Vision v2.0  ·  就绪  ·  {time.strftime('%Y-%m-%d %H:%M:%S')}"
        )
        self._status_timer = QTimer(self)
        self._status_timer.setInterval(1000)
        self._status_timer.timeout.connect(self._refresh_clock)
        self._status_timer.start()

        # ── Hub 信号连接 ─────────────────────────────────────────────
        if self.hub:
            self.hub.signal_rf_frame.connect(self.update_rf_frame)
            self.hub.signal_k230_frame.connect(self.update_k230_frame)
            self.hub.signal_log.connect(self.append_log)
            self.hub.signal_system_status.connect(self.update_status_labels)
            self.hub.signal_db_updated.connect(self.load_db_data)

        # ── 告警计数器 ────────────────────────────────────────────────
        self._alert_count = 0

        # ── 香橙派系统监控 ────────────────────────────────────────────
        self._setup_orangepi_monitor()

        # Esc 退出全屏
        esc_sc = QShortcut(QKeySequence(Qt.Key_Escape), self)
        esc_sc.activated.connect(self.showMaximized)

    # ==================================================================
    # 橙派监控初始化
    # ==================================================================
    def _setup_orangepi_monitor(self):
        """启动本地 psutil 系统资源监控器（上位机直接运行于香橙派本地）。"""
        self._opi_monitor = OrangePiMonitor(
            interval_s=2.0,
            net_iface="eth0",
        )
        self._opi_monitor.sig_stats.connect(self._on_opi_stats)
        self._opi_monitor.start()

    def _on_opi_stats(self, stats: dict):
        """接收香橙派系统资源数据并更新监控面板。"""
        online   = stats.get("online", False)
        cpu      = stats.get("cpu", 0.0)
        tx_kbps  = stats.get("net_tx_kbps", 0.0)
        rx_kbps  = stats.get("net_rx_kbps", 0.0)

        # 在线状态
        if online:
            self._opi_online_dot.setStyleSheet(
                f"color: {PALETTE['accent_green']}; font-size: 14px; font-weight: 700;"
            )
            self._opi_online_lbl.setText("在线")
            self._opi_online_lbl.setStyleSheet(f"color: {PALETTE['accent_green']}; font-size: 11px; font-weight: 600;")
        else:
            self._opi_online_dot.setStyleSheet(
                f"color: {PALETTE['accent_red']}; font-size: 14px; font-weight: 700;"
            )
            self._opi_online_lbl.setText("离线")
            self._opi_online_lbl.setStyleSheet(f"color: {PALETTE['accent_red']}; font-size: 11px; font-weight: 600;")

        # CPU 仪表盘
        self._cpu_gauge.set_value(cpu)

        # 网络速度卡
        self._net_tx_card.update_speed(tx_kbps)
        self._net_rx_card.update_speed(rx_kbps)

    # ==================================================================
    # UI 构建：顶部标题栏
    # ==================================================================
    def _build_titlebar(self) -> QFrame:
        bar = QFrame()
        bar.setFixedHeight(56)
        bar.setStyleSheet(f"""
            QFrame {{
                background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                    stop:0 #0a0d1a, stop:1 #0d1628);
                border-bottom: 1px solid {PALETTE['border']};
            }}
        """)
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(20, 0, 20, 0)

        # 左侧：系统名称
        lbl_brand = QLabel("RF-Vision")
        lbl_brand.setStyleSheet(f"""
            color: {PALETTE['accent_blue']};
            font-size: 18px;
            font-weight: 800;
        """)
        glow = QGraphicsDropShadowEffect()
        glow.setBlurRadius(20)
        glow.setColor(QColor(PALETTE['accent_blue']))
        glow.setOffset(0, 0)
        lbl_brand.setGraphicsEffect(glow)

        lbl_sub = QLabel("无人机多模态射频 · 视觉融合探测系统")
        lbl_sub.setStyleSheet(f"color: {PALETTE['text_muted']}; font-size: 12px; margin-left: 12px;")

        # 右侧：版本
        lbl_ver = QLabel("v2.0  ·  RK3588 Edition")
        lbl_ver.setStyleSheet(f"color: {PALETTE['text_muted']}; font-size: 11px;")

        lay.addWidget(lbl_brand)
        lay.addWidget(lbl_sub)
        lay.addStretch()
        lay.addWidget(lbl_ver)
        return bar

    # ==================================================================
    # UI 构建：Tab1 — 实时监测仪表板
    # ==================================================================
    def setup_live_dashboard(self):
        root = QVBoxLayout(self.tab1)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(12)

        # ── 顶部状态横幅 ──────────────────────────────────────────────
        banner = QHBoxLayout()
        banner.setSpacing(10)

        self._card_sdr     = StatusBannerCard("SDR 射频节点", "PlutoSDR  ·  休眠",   PALETTE['accent_blue'])
        self._card_vision  = StatusBannerCard("视觉节点",     "K230  ·  休眠",        PALETTE['accent_amber'])
        self._card_system  = StatusBannerCard("系统状态",     "待机  ·  等待启动指令", PALETTE['accent_cyan'])

        # 告警计数器卡片
        self._card_alert   = StatusBannerCard("告警事件", "0 次告警", PALETTE['accent_red'])

        for c in [self._card_sdr, self._card_vision, self._card_system, self._card_alert]:
            banner.addWidget(c)

        root.addLayout(banner)

        # ── 主内容区（左：RF频谱 | 中：K230视频 | 右：香橙派监控）───
        content = QHBoxLayout()
        content.setSpacing(12)

        # 左列：RF 瀑布图
        left_col = QVBoxLayout()
        left_col.setSpacing(8)

        rf_card = QFrame()
        rf_card.setObjectName("card")
        rf_card_lay = QVBoxLayout(rf_card)
        rf_card_lay.setContentsMargins(0, 0, 0, 0)
        rf_card_lay.setSpacing(0)

        rf_hdr = self._make_card_header("▤  SDR 射频瀑布图", PALETTE['accent_blue'])
        rf_card_lay.addWidget(rf_hdr)

        self.img_rf = QLabel()
        self.img_rf.setFixedSize(580, 580)
        self.img_rf.setAlignment(Qt.AlignCenter)
        self.img_rf.setStyleSheet("background-color: #010205; border-radius: 0 0 12px 12px;")
        rf_card_lay.addWidget(self.img_rf)
        left_col.addWidget(rf_card)

        # 中列：K230 视频
        mid_col = QVBoxLayout()
        mid_col.setSpacing(8)

        k230_card = QFrame()
        k230_card.setObjectName("card")
        k230_card_lay = QVBoxLayout(k230_card)
        k230_card_lay.setContentsMargins(0, 0, 0, 0)

        k230_hdr = self._make_card_header("◉  K230 光电视觉推流", PALETTE['accent_amber'])
        k230_card_lay.addWidget(k230_hdr)

        self.img_k230 = QLabel()
        self.img_k230.setMinimumSize(640, 480)
        self.img_k230.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.img_k230.setAlignment(Qt.AlignCenter)
        self.img_k230.setStyleSheet("background-color: #010205; border-radius: 0 0 12px 12px;")
        k230_card_lay.addWidget(self.img_k230, stretch=1)
        mid_col.addWidget(k230_card, stretch=1)

        # 右列：香橙派系统监控面板 + 控制按钮
        right_col = QVBoxLayout()
        right_col.setSpacing(10)
        right_col.addWidget(self._build_opi_monitor_panel())
        right_col.addWidget(self._build_control_panel())
        right_col.addStretch()

        content.addLayout(left_col)
        content.addLayout(mid_col, stretch=1)
        content.addLayout(right_col)
        root.addLayout(content, stretch=1)

        # ── 底部日志区 ─────────────────────────────────────────────────
        log_card = QFrame()
        log_card.setObjectName("card")
        log_card.setFixedHeight(180)
        log_card_lay = QVBoxLayout(log_card)
        log_card_lay.setContentsMargins(0, 0, 0, 0)

        log_hdr = self._make_card_header("▸  系统事件日志", PALETTE['accent_green'])
        log_card_lay.addWidget(log_hdr)

        self.log_textbox = QPlainTextEdit()
        self.log_textbox.setReadOnly(True)
        log_card_lay.addWidget(self.log_textbox, stretch=1)

        root.addWidget(log_card)

    def _make_card_header(self, title: str, accent: str) -> QFrame:
        """创建统一样式的卡片标题头部。"""
        hdr = QFrame()
        hdr.setFixedHeight(40)
        hdr.setStyleSheet(f"""
            QFrame {{
                background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                    stop:0 {accent}22, stop:1 transparent);
                border-bottom: 1px solid {PALETTE['border']};
                border-radius: 12px 12px 0 0;
            }}
        """)
        lay = QHBoxLayout(hdr)
        lay.setContentsMargins(14, 0, 14, 0)
        lbl = QLabel(title)
        lbl.setStyleSheet(f"color: {accent}; font-size: 13px; font-weight: 700; background: transparent; border: none;")
        lay.addWidget(lbl)
        lay.addStretch()
        return hdr

    # ------------------------------------------------------------------
    def _build_opi_monitor_panel(self) -> QFrame:
        """
        构建香橙派 RK3588 系统资源监控面板。

        面板结构：
          ┌─ 香橙派监控 ─────────────────────────────┐
          │  ● 在线/离线状态                          │
          │  ┌─ CPU ─────────────┐  (圆弧仪表盘)      │
          │  │   [arc gauge]     │                    │
          │  └───────────────────┘                    │
          │  ┌─ 网络上行 ─────────────────────────────┤
          │  │ ↑ 上行  xxx KB/s  [progress bar]      │
          │  ├─ 网络下行 ─────────────────────────────┤
          │  │ ↓ 下行  xxx KB/s  [progress bar]      │
          │  └───────────────────────────────────────┘
        """
        panel = QFrame()
        panel.setObjectName("card")
        panel.setFixedWidth(260)
        panel.setMinimumHeight(340)
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(0, 0, 0, 12)
        lay.setSpacing(10)

        # 标题头
        hdr = self._make_card_header("▣  香橙派 RK3588 监控", PALETTE['accent_purple'])
        lay.addWidget(hdr)

        # 在线状态行
        status_row = QHBoxLayout()
        status_row.setContentsMargins(14, 0, 14, 0)
        self._opi_online_dot = QLabel("●")
        self._opi_online_dot.setStyleSheet(f"color: {PALETTE['text_muted']}; font-size: 14px;")
        self._opi_online_lbl = QLabel("等待连接...")
        self._opi_online_lbl.setStyleSheet(f"color: {PALETTE['text_muted']}; font-size: 11px;")
        lbl_host = QLabel("本地  ·  psutil")
        lbl_host.setStyleSheet(f"color: {PALETTE['text_muted']}; font-size: 10px;")
        lbl_host.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        status_row.addWidget(self._opi_online_dot)
        status_row.addWidget(self._opi_online_lbl)
        status_row.addStretch()
        status_row.addWidget(lbl_host)
        lay.addLayout(status_row)

        # CPU 仪表盘居中
        cpu_row = QHBoxLayout()
        self._cpu_gauge = CpuGaugeDial()
        cpu_info_col = QVBoxLayout()
        cpu_info_col.setSpacing(4)
        lbl_cpu_title = QLabel("处理器占用率")
        lbl_cpu_title.setStyleSheet(f"color: {PALETTE['text_secondary']}; font-size: 11px; font-weight: 600;")
        lbl_cpu_sub = QLabel("RK3588  ·  8核 ARM A76/A55")
        lbl_cpu_sub.setStyleSheet(f"color: {PALETTE['text_muted']}; font-size: 10px;")
        lbl_cpu_sub.setWordWrap(True)
        cpu_info_col.addWidget(lbl_cpu_title)
        cpu_info_col.addWidget(lbl_cpu_sub)
        cpu_info_col.addStretch()
        cpu_row.addWidget(self._cpu_gauge)
        cpu_row.addLayout(cpu_info_col)
        cpu_row.setContentsMargins(14, 0, 14, 0)
        lay.addLayout(cpu_row)

        # 分隔线
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet(f"color: {PALETTE['border']};")
        lay.addWidget(sep)

        # 网络速度标题
        net_hdr = QLabel("  ⊕  网络吞吐量")
        net_hdr.setStyleSheet(f"color: {PALETTE['text_secondary']}; font-size: 11px; font-weight: 600;")
        lay.addWidget(net_hdr)

        # 网速卡片
        net_container = QVBoxLayout()
        net_container.setContentsMargins(14, 0, 14, 0)
        net_container.setSpacing(8)
        self._net_tx_card = NetSpeedCard("↑  上行带宽", PALETTE['accent_cyan'],  max_kbps=51200.0)
        self._net_rx_card = NetSpeedCard("↓  下行带宽", PALETTE['accent_amber'], max_kbps=51200.0)
        net_container.addWidget(self._net_tx_card)
        net_container.addWidget(self._net_rx_card)
        lay.addLayout(net_container)

        return panel

    # ------------------------------------------------------------------
    def _build_control_panel(self) -> QFrame:
        """构建启动/停止控制面板。"""
        panel = QFrame()
        panel.setObjectName("card")
        panel.setFixedWidth(260)
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(0, 0, 0, 14)
        lay.setSpacing(10)

        hdr = self._make_card_header("◎  采集控制", PALETTE['accent_green'])
        lay.addWidget(hdr)

        btns = QVBoxLayout()
        btns.setContentsMargins(14, 4, 14, 4)
        btns.setSpacing(8)

        self.btn_play = QPushButton("▶  启动数据采集")
        self.btn_play.setObjectName("btn_start")
        self.btn_play.setMinimumHeight(48)
        self.btn_play.clicked.connect(self.toggle_play)

        self.btn_exit = QPushButton("■  安全终止进程组")
        self.btn_exit.setObjectName("btn_secondary")
        self.btn_exit.setMinimumHeight(38)
        self.btn_exit.clicked.connect(self.close)

        btns.addWidget(self.btn_play)
        btns.addWidget(self.btn_exit)
        lay.addLayout(btns)

        return panel

    # ==================================================================
    # UI 构建：Tab2 — 告警日志库
    # ==================================================================
    def setup_evidence_database(self):
        layout = QVBoxLayout(self.tab2)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        # 工具栏卡片
        toolbar_card = QFrame()
        toolbar_card.setObjectName("card")
        toolbar_card.setFixedHeight(60)
        tb_lay = QHBoxLayout(toolbar_card)
        tb_lay.setContentsMargins(14, 0, 14, 0)

        lbl_tb = QLabel("▦  告警证据数据库")
        lbl_tb.setStyleSheet(f"color: {PALETTE['text_primary']}; font-size: 14px; font-weight: 700;")

        self.btn_clear_db = QPushButton("✕  清除全部记录")
        self.btn_clear_db.setObjectName("btn_danger")
        self.btn_clear_db.setMaximumWidth(180)
        self.btn_clear_db.setFixedHeight(36)
        self.btn_clear_db.clicked.connect(self.on_clear_db_clicked)

        self._lbl_db_count = QLabel("0 条记录")
        self._lbl_db_count.setStyleSheet(f"color: {PALETTE['text_muted']}; font-size: 12px;")

        tb_lay.addWidget(lbl_tb)
        tb_lay.addWidget(self._lbl_db_count)
        tb_lay.addStretch()
        tb_lay.addWidget(self.btn_clear_db)
        layout.addWidget(toolbar_card)

        # 内容区：表格 + 图片预览
        content = QHBoxLayout()
        content.setSpacing(10)

        # 表格
        table_card = QFrame()
        table_card.setObjectName("card")
        table_card.setFixedWidth(420)
        t_lay = QVBoxLayout(table_card)
        t_lay.setContentsMargins(0, 0, 0, 0)

        t_hdr = self._make_card_header("▤  事件列表", PALETTE['accent_blue'])
        t_lay.addWidget(t_hdr)

        self.db_table = QTableWidget()
        self.db_table.setColumnCount(4)
        self.db_table.setHorizontalHeaderLabels(["ID", "触发时间", "频率 (MHz)", "置信度"])
        self.db_table.setAlternatingRowColors(True)
        self.db_table.setShowGrid(False)
        self.db_table.verticalHeader().setVisible(False)
        hdr = self.db_table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(1, QHeaderView.Stretch)
        hdr.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.db_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.db_table.setSelectionMode(QTableWidget.SingleSelection)
        self.db_table.itemSelectionChanged.connect(self.on_db_row_selected)
        self.db_table.setStyleSheet("border: none; border-radius: 0 0 12px 12px;")
        t_lay.addWidget(self.db_table, stretch=1)
        content.addWidget(table_card)

        self.current_db_paths = []

        # 证据图片预览
        img_card = QFrame()
        img_card.setObjectName("card")
        img_card_lay = QVBoxLayout(img_card)
        img_card_lay.setContentsMargins(0, 0, 0, 0)

        img_hdr = self._make_card_header("▢  证据图像预览", PALETTE['accent_amber'])
        img_card_lay.addWidget(img_hdr)

        self.db_img_label = QLabel("请选择告警记录以显示关联的多模态证据图像")
        self.db_img_label.setAlignment(Qt.AlignCenter)
        self.db_img_label.setStyleSheet(
            f"background: transparent; color: {PALETTE['text_muted']};"
            f"font-size: 14px; border: none;"
        )
        img_card_lay.addWidget(self.db_img_label, stretch=1)
        content.addWidget(img_card, stretch=1)

        layout.addLayout(content, stretch=1)

    # ==================================================================
    # 视图渲染回调
    # ==================================================================
    def update_rf_frame(self, frame):
        self.render_cv2_to_qlabel(frame, self.img_rf)

    def update_k230_frame(self, frame):
        self.render_cv2_to_qlabel(frame, self.img_k230)

    def update_status_labels(self, status_dict: dict):
        if "sdr" in status_dict:
            self._card_sdr.update_status(status_dict["sdr"])
        if "vision" in status_dict:
            self._card_vision.update_status(status_dict["vision"])
        if "system" in status_dict:
            active = "△" in status_dict["system"] or "●" in status_dict["system"]
            color = status_dict.get("color", PALETTE['accent_cyan'])
            self._card_system.update_status(status_dict["system"], active=active, color=color)

        # 告警检测
        if status_dict.get("color", "") == "#e74c3c":
            self._alert_count += 1
            self._card_alert.update_status(f"{self._alert_count} 次告警", active=True, color=PALETTE['accent_red'])

    def append_log(self, text: str):
        html_text = text.replace('\n', '<br>')
        self.log_textbox.appendHtml(html_text)
        self.log_textbox.verticalScrollBar().setValue(
            self.log_textbox.verticalScrollBar().maximum()
        )

    def toggle_play(self):
        if not self.hub:
            return
        if self.hub.running:
            self.hub.stop_sensing()
            self.btn_play.setText("▶  启动数据采集")
            self.btn_play.setObjectName("btn_start")
            self.btn_play.setStyleSheet("")   # 重置样式触发 QSS 重新解析
            self.btn_play.setStyle(self.btn_play.style())
        else:
            self.hub.start_sensing()
            self.btn_play.setText("||  停止采集管道")
            self.btn_play.setObjectName("btn_stop")
            self.btn_play.setStyleSheet("")
            self.btn_play.setStyle(self.btn_play.style())

    def on_clear_db_clicked(self):
        reply = QMessageBox.question(
            self, "确认清除",
            "确定要删除全部告警记录和关联图片吗？\n此操作不可撤销。",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            return
        n = self.db_engine.clear_all()
        self._alert_count = 0
        self._card_alert.update_status("0 次告警", active=False)
        self.load_db_data()
        self.db_img_label.setText(f"已清除 {n} 条告警记录及关联图片。")

    def on_tab_changed(self, index: int):
        if index == 1:
            self.load_db_data()

    def load_db_data(self):
        rows = self.db_engine.get_all_alerts()
        self.db_table.setRowCount(len(rows))
        self.current_db_paths = []
        for row_idx, data in enumerate(rows):
            self.db_table.setItem(row_idx, 0, QTableWidgetItem(f"REC-{data[0]}"))
            self.db_table.setItem(row_idx, 1, QTableWidgetItem(str(data[1])))
            self.db_table.setItem(row_idx, 2, QTableWidgetItem(f"{data[2]} MHz"))
            conf_item = QTableWidgetItem(f"{data[3] * 100:.1f} %")
            if data[3] >= 0.9:
                conf_item.setForeground(QColor(PALETTE['accent_red']))
            elif data[3] >= 0.7:
                conf_item.setForeground(QColor(PALETTE['accent_amber']))
            else:
                conf_item.setForeground(QColor(PALETTE['accent_green']))
            self.db_table.setItem(row_idx, 3, conf_item)
            self.current_db_paths.append(data[4])

        self._lbl_db_count.setText(f"{len(rows)} 条记录")

    def on_db_row_selected(self):
        selected = self.db_table.selectedItems()
        if not selected:
            return
        row = selected[0].row()
        img_path = self.current_db_paths[row]
        cv_img = cv2.imread(img_path)
        if cv_img is not None:
            self.render_cv2_to_qlabel(cv_img, self.db_img_label)
        else:
            self.db_img_label.setText("△  I/O 错误：图像文件寻址失败 —— 文件可能已被移动或删除。")

    def render_cv2_to_qlabel(self, cv_img, qlabel: QLabel):
        qlabel.clear()
        target_w = qlabel.width()
        target_h = qlabel.height()
        if target_w <= 0 or target_h <= 0:
            return

        if len(cv_img.shape) == 3:
            h, w, ch = cv_img.shape
            # 保持宽高比缩放至目标区域
            scale = min(target_w / w, target_h / h)
            nw, nh = int(w * scale), int(h * scale)
            resized = cv2.resize(cv_img, (nw, nh), interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            qt_img = QImage(rgb.data, nw, nh, nw * 3, QImage.Format_RGB888)
        else:
            h, w = cv_img.shape
            scale = min(target_w / w, target_h / h)
            nw, nh = int(w * scale), int(h * scale)
            resized = cv2.resize(cv_img, (nw, nh))
            qt_img = QImage(resized.data, nw, nh, nw, QImage.Format_Grayscale8)

        qlabel.setPixmap(QPixmap.fromImage(qt_img))

    # ==================================================================
    # 状态栏时钟
    # ==================================================================
    def _refresh_clock(self):
        self.statusBar().showMessage(
            f"RF-Vision v2.0  ·  {time.strftime('%Y-%m-%d  %H:%M:%S')}"
            f"  ·  香橙派本地监控已启用"
        )

    # ==================================================================
    # 窗口关闭
    # ==================================================================
    def closeEvent(self, event):
        self._opi_monitor.stop()
        super().closeEvent(event)
