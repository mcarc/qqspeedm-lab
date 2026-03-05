import os
import streamlit as st
import subprocess
import cv2
import shutil
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, Dict

class VideoProcessor:
    """处理视频元数据和 FFmpeg 操作的业务逻辑类"""
    def __init__(self, file_path: Path):
        self.file_path = file_path
        self.file_name = file_path.name

    def get_metadata(self) -> Dict:
        """使用 OpenCV 获取视频时长和 FPS"""
        try:
            cap = cv2.VideoCapture(str(self.file_path))
            if not cap.isOpened():
                return {}
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0
            cap.release()
            
            return {"fps": fps, "duration": duration, "total_frames": total_frames}
        except Exception as e:
            st.error(f"无法读取视频信息: {e}")
            return {}

    def slice_video(self, start_time: str, end_time: str, output_path: str = "clipped_video.mp4") -> Tuple[bool, str]:
        """执行 FFmpeg 切片"""
        if not shutil.which("ffmpeg"):
            return False, "未找到 FFmpeg，请确保已安装并添加到系统环境变量。"

        cmd = [
            'ffmpeg', '-y',
            '-ss', start_time,
            '-to', end_time,
            '-i', str(self.file_path),
            '-c', 'copy',
            '-avoid_negative_ts', '1',
            output_path
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='ignore')
            if result.returncode == 0:
                return True, output_path
            else:
                return False, result.stderr
        except Exception as e:
            return False, str(e)