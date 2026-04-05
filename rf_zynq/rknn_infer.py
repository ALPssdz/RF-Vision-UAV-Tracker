# -*- coding: utf-8 -*-
"""
rknn_infer.py — RKNN NPU 推理封装模块
=======================================
在 RK3588 平台（Orange Pi 5）上，使用 rknn_toolkit_lite2 对频谱瀑布图
执行 YOLOv8 目标检测推理，替代基于 PyTorch 的 ultralytics 推理路径。

推理精度参数：
  - 输入尺寸：640 × 640 × 3（BGR，uint8）
  - 量化精度：INT8（由 rknn-toolkit2 离线转换时确定）
  - 单帧推理延迟（NPU）：约 20~40 ms（相比 CPU 加速约 10~30 倍）

接口兼容性：
  - `RKNNLiteInfer.predict()` 的返回值格式与 ultralytics `model.predict()` 保持一致，
    调用方无需区分推理后端，可无缝切换。

依赖安装（仅香橙派 5 上执行）：
  pip3 install rknn_toolkit_lite2-2.x.x-cpXXX-linux_aarch64.whl
  （从 https://github.com/airockchip/rknn-toolkit2/releases 获取 whl 文件）
"""

import numpy as np
import cv2
import os


def _letterbox(img: np.ndarray, target_size: int = 640):
    """
    Letterbox 缩放：在保持纵横比的前提下，将输入图像缩放填充至 target_size × target_size。

    Parameters
    ----------
    img         : BGR 格式图像，np.ndarray
    target_size : 目标正方形边长（像素），默认 640

    Returns
    -------
    padded   : 处理后的图像（target_size × target_size × 3，uint8）
    ratio    : 缩放比例（float）
    pad_left : 水平方向左侧填充像素数（用于坐标还原）
    pad_top  : 垂直方向顶部填充像素数（用于坐标还原）
    """
    h, w = img.shape[:2]
    ratio = min(target_size / h, target_size / w)
    new_w, new_h = int(w * ratio), int(h * ratio)

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_left = (target_size - new_w) // 2
    pad_top  = (target_size - new_h) // 2
    padded   = np.full((target_size, target_size, 3), 128, dtype=np.uint8)
    padded[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized

    return padded, ratio, pad_left, pad_top


def _decode_yolov8_output(outputs: list, conf_thresh: float,
                           orig_h: int, orig_w: int,
                           ratio: float, pad_left: int, pad_top: int) -> list:
    """
    解码 YOLOv8 NPU 输出张量为边界框列表。

    YOLOv8 输出格式（单类检测，经 ONNX 导出后）：
      outputs[0].shape = (1, 5 + num_classes, num_anchors)
      其中前 4 个通道为 cx, cy, w, h（相对 640×640 的归一化坐标乘以 640）

    Returns
    -------
    list of dict : [{'bbox': [x1,y1,x2,y2], 'conf': float}, ...]
    bbox 坐标为原始图像像素坐标
    """
    results = []
    preds = outputs[0]
    if preds.ndim == 3:
        preds = preds[0]         # (5+nc, anchors)
    preds = preds.T              # (anchors, 5+nc)

    for pred in preds:
        cx, cy, bw, bh = pred[0], pred[1], pred[2], pred[3]
        class_scores = pred[4:]
        conf = float(np.max(class_scores))
        if conf < conf_thresh:
            continue

        # 还原至原始图像坐标
        x1 = int((cx - bw / 2 - pad_left) / ratio)
        y1 = int((cy - bh / 2 - pad_top)  / ratio)
        x2 = int((cx + bw / 2 - pad_left) / ratio)
        y2 = int((cy + bh / 2 - pad_top)  / ratio)

        x1, x2 = max(0, x1), min(orig_w, x2)
        y1, y2 = max(0, y1), min(orig_h, y2)
        results.append({'bbox': [x1, y1, x2, y2], 'conf': conf})

    return results


class _FakeResult:
    """
    ultralytics Results 格式的轻量级兼容对象，供 active_yolo_inference() 解析。
    """
    class _Boxes:
        def __init__(self, detections):
            import torch
            if detections:
                self.conf = torch.tensor([d['conf'] for d in detections])
            else:
                self.conf = torch.tensor([])

        def __len__(self):
            return len(self.conf)

    def __init__(self, detections: list, img_bgr: np.ndarray):
        self.boxes = self._Boxes(detections)
        self._img  = img_bgr
        self._dets = detections

    def plot(self) -> np.ndarray:
        """在图像上绘制检测框，与 ultralytics r.plot() 接口一致。"""
        canvas = self._img.copy()
        for d in self._dets:
            x1, y1, x2, y2 = d['bbox']
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(canvas, f"UAV {d['conf']:.2f}",
                        (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2)
        return canvas


class RKNNLiteInfer:
    """
    基于 rknn_toolkit_lite2 的 YOLOv8 NPU 推理封装类。

    对外提供与 ultralytics.YOLO 一致的 predict() 接口，
    使 main_rf_pipeline.py 的推理逻辑对推理后端透明。

    Parameters
    ----------
    rknn_path  : .rknn 模型文件路径
    conf_thresh: 检测置信度阈值（默认 0.60）
    """

    def __init__(self, rknn_path: str, conf_thresh: float = 0.60):
        from rknnlite.api import RKNNLite

        if not os.path.exists(rknn_path):
            raise FileNotFoundError(
                f"[RKNN] 模型文件不存在: {rknn_path}\n"
                f"       请先在 WSL2/Linux x86_64 上运行 tools/convert_yolo_to_rknn.py 完成转换。"
            )

        self._conf_thresh = conf_thresh
        self._rknn = RKNNLite()
        ret = self._rknn.load_rknn(rknn_path)
        if ret != 0:
            raise RuntimeError(f"[RKNN] 模型加载失败（error code: {ret}）: {rknn_path}")

        # 初始化 NPU 运行时（core_mask=RKNN_NPU_CORE_AUTO 自动调度）
        ret = self._rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO)
        if ret != 0:
            raise RuntimeError(f"[RKNN] NPU 运行时初始化失败（error code: {ret}）")

        print(f"[RKNN] YOLOv8 NPU 推理引擎初始化完成: {rknn_path}")

    def predict(self, source: np.ndarray, verbose: bool = False) -> list:
        """
        对输入 BGR 图像执行 YOLOv8 NPU 推理。

        Parameters
        ----------
        source  : 640×640×3 BGR 图像（uint8），与 ultralytics predict() 接口一致
        verbose : 是否打印详细调试信息（默认 False）

        Returns
        -------
        list : [_FakeResult]，格式与 ultralytics Results 列表兼容
        """
        orig_h, orig_w = source.shape[:2]
        img_lb, ratio, pad_left, pad_top = _letterbox(source, 640)

        # RKNN 推理输入格式要求：
        #   - 颜色空间：RGB（YOLOv8 ONNX 以 RGB 训练导出）
        #   - 形状：(1, 640, 640, 3) NHWC，即需在 HWC 前加 batch 维度
        img_rgb = cv2.cvtColor(img_lb, cv2.COLOR_BGR2RGB)
        img_4d  = np.expand_dims(img_rgb, axis=0)       # (1, 640, 640, 3)

        outputs = self._rknn.inference(inputs=[img_4d])

        detections = _decode_yolov8_output(
            outputs, self._conf_thresh,
            orig_h, orig_w, ratio, pad_left, pad_top
        )

        if verbose:
            print(f"  [RKNN] 检测到 {len(detections)} 个目标: "
                  f"{[round(d['conf'], 3) for d in detections]}")

        return [_FakeResult(detections, source)]

    def __del__(self):
        try:
            self._rknn.release()
        except Exception:
            pass
