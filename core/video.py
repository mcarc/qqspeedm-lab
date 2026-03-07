import os
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
        cap = cv2.VideoCapture(str(self.file_path))
        if not cap.isOpened():
            return {}
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        cap.release()
        
        return {"fps": fps, "duration": duration, "total_frames": total_frames}


    def slice_video(self, start_time: str, end_time: str, output_path: str = "clipped_video.mp4") -> Tuple[bool, str]:
        """执行 FFmpeg 切片"""
        if not shutil.which("ffmpeg"):
            return False, "未找到 FFmpeg，请确保已安装并添加到系统环境变量。"

        cmd = [
            'ffmpeg', '-y',
            '-ss', str(start_time),
            '-to', str(end_time),
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
        
    def get_video_frame_count(self) -> int:
        """获取视频总帧数并缓存 (注意使用 _self 避免哈希实例)"""
        # 注意：如果你的 get_metadata 已经在其他地方被调用并缓存，
        # 也可以直接考虑复用 _self.get_metadata().get("total_frames")
        cap = cv2.VideoCapture(str(self.file_path))
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return count

    def get_video_frame(self, frame_no: int = 0):
        """
        读取指定帧并缓存。
        默认 frame_no=0 即等同于原先的 get_video_first_frame。
        """
        cap = cv2.VideoCapture(str(self.file_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def process_frames_generator(self, frame_processor, roi: tuple, output_dir: str):
        """
        纯后端逻辑：提取并处理视频帧。
        使用 yield 实时回传进度和结果，实现与 UI 的完全解耦。
        """

        os.makedirs(output_dir, exist_ok=True)

        cap = cv2.VideoCapture(str(self.file_path))
        if not cap.isOpened():
            raise ValueError(f"无法打开视频源: {self.file_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0: 
            total_frames = 100
        
        frame_idx = 0
        
        while cap.isOpened():
            current_video_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            ret, frame = cap.read()
            if not ret:
                break
            
            # 调用外部传入的图像/OCR处理器
            result = frame_processor.process_and_save(
                frame, roi, output_dir, current_video_ms, frame_idx
            )
            
            frame_idx += 1
            
            # 如果拿不到总帧数，就返回 None，让前端去决定怎么显示
            progress = min(frame_idx / total_frames, 1.0) if total_frames > 0 else None
            
            # 将当前进度和处理结果“打包”抛出，交由调用方处理
            yield {
                "frame_idx": frame_idx,
                "progress": progress,
                "result": result
            }

        cap.release()
