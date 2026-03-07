import cv2
import os
import numpy as np
from cnocr import CnOcr


def preprocess_roi(roi_img):
    """
    预处理 ROI：灰度化 + 自适应二值化
    """

    # 1. 转换为灰度图
    gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
    
    # 2. 阈值处理 (突出文本，压制背景噪声)
    _, processed = cv2.threshold(gray, 190, 255, cv2.THRESH_BINARY)
    
    # 3. 如果你的 OCR 引擎要求 3 通道输入，可以将单通道转回 3 通道
    # (很多 OCR 库如 PaddleOCR 内部会处理，但这里保险起见)
    processed_bgr = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
    
    return processed_bgr

class OcrProcessor:
    def __init__(self, config):
        # 初始化模型
        self.ocr_model = CnOcr(
            rec_model_name=config['ocr_engine']['rec_model_name'], 
            det_model_name=config['ocr_engine']['det_model_name'], 
            cand_alphabet=config['ocr_engine']['cand_alphabet'],
            rec_more_configs=dict(font_path=config['ocr_engine'].get('font_path', None))
        )
        print("✅ OCR 模型加载完成！配置:", config['ocr_engine'])

    def process_and_save(self, frame, roi, output_folder, video_ms, frame_idx):
        """
        处理单帧并保存
        :param video_ms: 当前帧在视频中的时间位置 (毫秒)
        :param frame_idx: 帧序号 (用于生成唯一文件名)
        """
        x, y, w, h = roi
        
        # 1. 裁剪 ROI
        roi_img = frame[y:y+h, x:x+w]

        roi_img = preprocess_roi(roi_img)

        # OCR 模型通常要求 RGB 输入，所以这里转换一下
        roi_rgb = cv2.cvtColor(roi_img, cv2.COLOR_BGR2RGB)

        # 2. OCR 识别
        ocr_result = self.ocr_model.ocr_for_single_line(roi_rgb)
        if ocr_result:
            text = ocr_result['text']
            score = ocr_result['score']
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