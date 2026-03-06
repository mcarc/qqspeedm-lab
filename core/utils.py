from datetime import timedelta

from typing import Optional, Tuple
import os
import base64
import cv2
import re

def hms_to_seconds(hms: str) -> Optional[float]:
    """将 hh:mm:ss 转换为秒数"""
    try:
        parts = hms.split(':')
        if len(parts) == 3:
            return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
        elif len(parts) == 2:
            return float(parts[0]) * 60 + float(parts[1])
        return float(parts[0])
    except:
        return None

def seconds_to_hms(seconds: float) -> str:
    """将秒数转换为 hh:mm:ss.ss"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:05.2f}"

def parse_roi_string(roi_str: str) -> Optional[Tuple[int, int, int, int]]:
    """解析形如 'X=820, Y=900, W=97, H=46' 的字符串"""
    pattern = r"X\s*=\s*(\d+),\s*Y\s*=\s*(\d+),\s*W\s*=\s*(\d+),\s*H\s*=\s*(\d+)"
    match = re.search(pattern, roi_str, re.IGNORECASE)
    if match:
        return tuple(map(int, match.groups()))
    return None

def img_path_to_base64(file_path):
    """读取本地图片转 Base64 供表格内显示 (ROI截图)"""
    if not os.path.exists(file_path):
        return None
    try:
        with open(file_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode()
        return f"data:image/jpeg;base64,{encoded}"
    except Exception:
        return None
