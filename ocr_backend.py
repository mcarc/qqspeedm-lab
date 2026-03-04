import cv2
import os
import numpy as np
from cnocr import CnOcr

class OcrProcessor:
    def __init__(self):
        # 初始化模型
        self.ocr = CnOcr(
            rec_model_name='densenet_lite_136-gru', 
            det_model_name='naive_det', 
            cand_alphabet='0123456789.'
        )

    def _format_timestamp(self, milliseconds):
        """将毫秒转换为 HH:MM:SS.mmm 格式"""
        seconds = int(milliseconds // 1000)
        ms = int(milliseconds % 1000)
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"

    def process_and_save(self, frame, roi, output_folder, video_ms, frame_idx):
        """
        处理单帧并保存
        :param video_ms: 当前帧在视频中的时间位置 (毫秒)
        :param frame_idx: 帧序号 (用于生成唯一文件名)
        """
        x, y, w, h = roi
        
        # 1. 裁剪 ROI
        roi_img = frame[y:y+h, x:x+w]
        roi_rgb = cv2.cvtColor(roi_img, cv2.COLOR_BGR2RGB)
        
        # 2. OCR 识别
        ocr_results = self.ocr.ocr(roi_rgb)
        if ocr_results:
            text = "".join([res['text'] for res in ocr_results])
            score = sum([res['score'] for res in ocr_results]) / len(ocr_results)
        else:
            text = ""
            score = 0.0

        # 3. 生成文件名
        filename = f"frame_{frame_idx:06d}_{video_ms}.jpg"
        file_path = os.path.join(output_folder, filename)
        
        # 保存 ROI 图片
        cv2.imwrite(file_path, roi_img)

        # 4. 返回数据
        return {
            "video_timestamp": video_ms,
            "frame_idx": frame_idx,
            "file_name": filename,
            "file_path": file_path,
            "value": text,
            "confidence": round(score, 4)
        }