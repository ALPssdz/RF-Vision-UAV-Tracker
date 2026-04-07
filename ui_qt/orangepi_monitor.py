# -*- coding: utf-8 -*-
from __future__ import annotations
"""
ui_qt/orangepi_monitor.py — 本机系统资源监控模块
==================================================
上位机直接运行于香橙派 RK3588 之上，通过 psutil 于本地采集：
  · CPU 总体占用率（%）
  · 实时网络上行带宽（KB/s）
  · 实时网络下行带宽（KB/s）

采集公式
---------
CPU 占用率（psutil 内部使用 /proc/stat 差分，采样间隔 Δt）：
    cpu_pct = (1 - Δt_idle / Δt_total) × 100   [%]

网络吞吐量（相邻两次采样之差除以时间间隔）：
    TX_{KB/s} = (bytes_sent₁ - bytes_sent₀) / (Δt × 1024)
    RX_{KB/s} = (bytes_recv₁ - bytes_recv₀) / (Δt × 1024)

信号接口
---------
    sig_stats(dict) — 每轮采集后发射，payload：
        {
          "cpu"         : float,   # 0.0 ~ 100.0 %
          "net_tx_kbps" : float,   # 上行 KB/s
          "net_rx_kbps" : float,   # 下行 KB/s
          "online"      : bool,    # 始终为 True（本地采集不存在离线）
        }

依赖
---------
    pip install psutil
"""

import time
import threading
from typing import Optional

from PyQt5.QtCore import QObject, QThread, pyqtSignal


# ──────────────────────────────────────────────────────────────────────────────
# 监控工作对象（运行在独立 QThread 中，避免阻塞 Qt 事件循环）
# ──────────────────────────────────────────────────────────────────────────────
class OrangePiMonitorWorker(QObject):
    """
    本地系统资源周期性采集工作对象。

    参数
    ----
    interval_s : float
        采集轮询间隔（秒）。psutil.cpu_percent 内部的采样窗口
        固定为 interval_s / 2，以保证 CPU 数据与网络差分
        在同一时间尺度内对齐。
        默认值 2.0 s，对应网络差分窗口 Δt ≈ 1.0 s。
    net_iface : str | None
        指定统计的网卡名称（如 'eth0'）。
        为 None 时聚合全部网卡（psutil.net_io_counters()
        pernic=False 模式），适用于不确定网卡名的场景。
    """

    sig_stats = pyqtSignal(dict)

    def __init__(self,
                 interval_s: float = 2.0,
                 net_iface: Optional[str] = "eth0"):
        super().__init__()
        self._interval  = interval_s
        self._iface     = net_iface
        self._stop_ev   = threading.Event()

    # ------------------------------------------------------------------
    def _collect(self) -> dict:
        """
        执行一次本地资源采集。

        实现细节
        --------
        1. 调用 psutil.cpu_percent(interval=None) + 手动 sleep，
           使 CPU 采样窗口与网络差分窗口共享同一 sleep 时间，
           避免串行双重阻塞导致实际轮询间隔翻倍。

        2. 网络差分：
               Δt = 1.0 s（固定，与 sleep 对齐）
               ΔB_tx = bytes_sent₁ - bytes_sent₀
               TX_{KB/s} = max(ΔB_tx, 0) / (Δt × 1024)

        3. net_iface 不存在时降级为全局聚合，防止因网卡名
           拼写错误导致整个监控线程崩溃。
        """
        try:
            import psutil
        except ImportError:
            # psutil 未安装时返回全零占位数据
            return {"cpu": 0.0, "net_tx_kbps": 0.0, "net_rx_kbps": 0.0, "online": True}

        # ── 采集前快照 ──────────────────────────────────────────────────
        psutil.cpu_percent(interval=None)   # 清空上次残留的累积值

        def _net_snapshot():
            """获取指定网卡（或全局）的字节计数器快照。"""
            if self._iface:
                per_nic = psutil.net_io_counters(pernic=True)
                if self._iface in per_nic:
                    c = per_nic[self._iface]
                    return c.bytes_sent, c.bytes_recv
            # 降级：全局聚合
            c = psutil.net_io_counters(pernic=False)
            return c.bytes_sent, c.bytes_recv

        tx0, rx0 = _net_snapshot()

        # ── 差分时间窗口（Δt = 1.0 s）──────────────────────────────────
        dt = 1.0
        time.sleep(dt)

        # ── 采集后快照 ──────────────────────────────────────────────────
        cpu_pct  = psutil.cpu_percent(interval=None)   # 基于上面 sleep 的窗口
        tx1, rx1 = _net_snapshot()

        # ── 计算带宽（KB/s）────────────────────────────────────────────
        # TX_{KB/s} = max(ΔB_tx, 0) / (Δt × 1024)
        # RX_{KB/s} = max(ΔB_rx, 0) / (Δt × 1024)
        tx_kbps = max(tx1 - tx0, 0) / (dt * 1024.0)
        rx_kbps = max(rx1 - rx0, 0) / (dt * 1024.0)

        return {
            "cpu":          float(cpu_pct),
            "net_tx_kbps":  tx_kbps,
            "net_rx_kbps":  rx_kbps,
            "online":       True,
        }

    # ------------------------------------------------------------------
    def run(self):
        """工作循环，由 QThread.started 信号触发。"""
        while not self._stop_ev.is_set():
            try:
                stats = self._collect()
                self.sig_stats.emit(stats)
            except Exception as e:
                self.sig_stats.emit({
                    "cpu": 0.0, "net_tx_kbps": 0.0,
                    "net_rx_kbps": 0.0, "online": False,
                    "error": str(e),
                })
            # 剩余等待时间（_collect 内部已 sleep 1s，此处补齐至 interval）
            remain = self._interval - 1.0
            if remain > 0:
                self._stop_ev.wait(timeout=remain)

    def stop(self):
        """线程安全停止。"""
        self._stop_ev.set()


# ──────────────────────────────────────────────────────────────────────────────
# 对外统一管理接口
# ──────────────────────────────────────────────────────────────────────────────
class OrangePiMonitor:
    """
    本机系统资源监控器（对外统一管理接口）。

    依赖 psutil 在本地香橙派直接读取 /proc 伪文件系统，
    无需任何网络通信或远程认证。

    使用示例
    --------
        monitor = OrangePiMonitor()
        monitor.sig_stats.connect(self.on_stats_updated)
        monitor.start()
        ...
        monitor.stop()
    """

    def __init__(self,
                 interval_s: float = 2.0,
                 net_iface: Optional[str] = "eth0",
                 **_ignored_kwargs):
        """
        参数
        ----
        interval_s : 采集间隔（秒）
        net_iface  : 网卡名称（None 表示聚合全部网卡）
        **_ignored_kwargs : 忽略历史遗留的 host/ssh_* 参数，保持调用兼容性
        """
        self._thread = QThread()
        self._worker = OrangePiMonitorWorker(
            interval_s=interval_s,
            net_iface=net_iface,
        )
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)

        # 转发信号，外部通过此属性连接槽函数
        self.sig_stats = self._worker.sig_stats

    def start(self):
        self._thread.start()

    def stop(self):
        self._worker.stop()
        self._thread.quit()
        self._thread.wait(5000)
