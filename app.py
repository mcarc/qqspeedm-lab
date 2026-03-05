import os
import streamlit as st
import subprocess
import cv2
import shutil
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, Dict
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from ocr import render_ocr
from kinematic import render_kinematic_analysis
from utils import hms_to_seconds, seconds_to_hms, parse_roi_string


# ================= 业务逻辑类 =================
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


# ================= UI 渲染组件 =================
def reset_canvas():
    """重置回全图模式"""
    st.session_state['crop_stage'] = 'full'
    st.session_state['zoom_coords'] = None

def render_sidebar() -> Optional[Path]:
    """渲染侧边栏文件选择器"""
    st.sidebar.header("📁 文件浏览")
    
    if "base_dir" not in st.session_state:
        st.session_state.base_dir = "D:\\Videos"

    base_dir_input = st.sidebar.text_input("视频文件夹路径:", value=st.session_state.base_dir)
    base_path = Path(base_dir_input)

    if base_path.exists() and base_path.is_dir():
        st.session_state.base_dir = str(base_path)
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv'}
        files = [f.name for f in base_path.iterdir() if f.suffix.lower() in video_extensions]
        
        if files:
            selected_file = st.sidebar.selectbox("选择视频文件:", files)
            selected_path = base_path / selected_file
            
            if st.session_state.get('current_source_file') != str(selected_path):
                st.session_state['current_source_file'] = str(selected_path)
                st.session_state['clipped_video_path'] = None

                # 切换文件时，强制关闭 OCR 模块并清理状态
                st.session_state['show_ocr_module'] = False
                st.session_state['current_ocr_coords'] = None
                st.session_state['show_kinematic_module'] = False
                if 'data_df' in st.session_state:
                    del st.session_state['data_df'] # 清除子模块的数据缓存

                st.session_state.data_df = pd.DataFrame()

                reset_canvas()
                
            return selected_path
        else:
            st.sidebar.warning("该目录下无视频文件")
            return None
    else:
        st.sidebar.error("路径不存在")
        return None

def render_slicer(video_path: Path):
    """渲染第一阶段：切片器"""
    st.subheader("✂️ 阶段一：选择时间切片")
    processor = VideoProcessor(video_path)
    metadata = processor.get_metadata()
    
    if not metadata:
        st.error("无法加载视频元数据")
        return

    duration = metadata.get('duration', 0)
    st.info(f"🎞️ 原视频时长: {seconds_to_hms(duration)} ({duration:.2f}s) | FPS: {metadata.get('fps'):.2f}")

    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        start_str = st.text_input("开始时间", value="00:00:00")
    with col2:
        default_end = seconds_to_hms(min(duration, 10))
        end_str = st.text_input("结束时间", value=default_end)
    with col3:
        st.write("") 
        st.write("")
        do_slice = st.button("🚀 生成切片", type="primary", use_container_width=True)

    if do_slice:
        t1 = hms_to_seconds(start_str)
        t2 = hms_to_seconds(end_str)
        
        if t1 is None or t2 is None or t1 >= t2:
            st.error("时间格式错误或开始时间大于结束时间")
            return

        with st.spinner("FFmpeg 正在处理..."):
            os.makedirs("tmp", exist_ok=True)
            success, msg = processor.slice_video(start_str, end_str, output_path="tmp/clipped.mp4")
            if success:
                st.session_state['clipped_video_path'] = msg
                reset_canvas()
                st.rerun()
            else:
                st.error("切片失败")
                st.code(msg)

def render_roi_selector(clipped_video_path: str):
    """
    渲染 ROI 选择器（预设坐标 或 手动框选）。
    注意：此函数不再直接渲染 OCR，而是负责更新 session_state 中的坐标和标志位。
    """
    st.divider()

    st.subheader("🎯 阶段二：提取 ROI")
    
    roi_mode = st.radio(
        "选择提取方式:", 
        ["使用预设坐标", "手动在画面中框选"], 
        horizontal=True
    )

    PRESETS = {
        "预设 1: 底部数值区": "X=820, Y=900, W=97, H=46",
        "预设 2: 左上角状态": "X=50, Y=50, W=150, H=150",
        "预设 3: 仪表盘区域": "X=1600, Y=900, W=200, H=100",
        "自定义手动输入": ""
    }

    # ---------------- 分支 A: 使用预设坐标 ----------------
    if roi_mode == "使用预设坐标":
        st.markdown("### ✏️ 直接输入或选择坐标")
        p_col1, p_col2 = st.columns(2)
        with p_col1:
            preset_choice = st.selectbox("快捷选项", list(PRESETS.keys()))
        with p_col2:
            default_val = PRESETS[preset_choice] if preset_choice != "自定义手动输入" else ""
            roi_input = st.text_input("坐标参数", value=default_val, placeholder="例如: X=820, Y=900, W=97, H=46")

        coords = parse_roi_string(roi_input)
        
        if coords:
            x, y, w, h = coords
            st.success(f"坐标解析成功: X={x}, Y={y}, W={w}, H={h}")
            
            cap = cv2.VideoCapture(clipped_video_path)
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                try:
                    final_crop = frame_rgb[y:y+h, x:x+w]
                    if final_crop.size > 0:
                        st.image(final_crop, caption="直接提取的 ROI 预览")
                        
                        # --- 动作：确认坐标并开启 OCR 状态 ---
                        if st.button("🔍 确认区域", type="primary", key="btn_ocr_preset"):
                            st.session_state['show_ocr_module'] = True
                            st.session_state['current_ocr_coords'] = (x, y, w, h)
                            st.rerun()
                    else:
                        st.error("裁剪区域超出视频画面范围，请检查坐标！")
                except Exception as e:
                    st.error(f"裁剪失败: {e}")
        elif roi_input.strip() != "":
            st.error("格式解析失败！请确保格式为: X=数字, Y=数字, W=数字, H=数字")
        
        if st.button("🔄 重新选择视频时间段"):
            st.session_state['clipped_video_path'] = None
            st.rerun()

    # ---------------- 分支 B: 手动在画面中框选 ----------------
    else:
        col1, col2 = st.columns([1, 3])
        with col1:
            st.video(clipped_video_path)
            if st.button("🔄 重新选择视频时间段"):
                st.session_state['clipped_video_path'] = None
                st.rerun()
                
        with col2:
            cap = cv2.VideoCapture(clipped_video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            frame_no = st.slider("拖动选择用于画框的视频帧", 0, max(0, total_frames - 1), 0, on_change=reset_canvas)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            ret, frame = cap.read()
            cap.release()

            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                full_img_pil = Image.fromarray(frame_rgb)
                raw_w, raw_h = full_img_pil.size
                
                st.caption(f"当前帧分辨率: {raw_w}x{raw_h}")

                if st.button("🔙 取消放大 / 重置视图"):
                    reset_canvas()
                    st.rerun()

                # --- Canvas 步骤 1: 粗略框选 ---
                if st.session_state['crop_stage'] == 'full':
                    st.markdown("**第一步：粗略框选需要放大的区域**")

                    canvas_width = 800 if raw_w > 800 else raw_w
                    canvas_height = int(raw_h * (canvas_width / raw_w))
                    
                    canvas_result = st_canvas(
                        fill_color="rgba(255, 165, 0, 0.3)",
                        stroke_width=2, stroke_color="#FF0000",
                        background_image=full_img_pil,
                        update_streamlit=True,
                        height=canvas_height, width=canvas_width,
                        drawing_mode="rect", key="canvas_full",
                    )

                    if canvas_result.json_data is not None:
                        objects = canvas_result.json_data["objects"]
                        if objects:
                            obj = objects[-1]
                            scale = raw_w / canvas_width
                            z_w, z_h = int(obj["width"] * scale), int(obj["height"] * scale)
                            
                            if z_w > 10 and z_h > 10:
                                st.session_state['zoom_coords'] = (
                                    int(obj["left"] * scale), int(obj["top"] * scale), z_w, z_h
                                )
                                st.session_state['crop_stage'] = 'zoomed'
                                st.rerun() 

                # --- Canvas 步骤 2: 精确提取 ---
                elif st.session_state['crop_stage'] == 'zoomed':
                    st.markdown("**第二步：在放大区域中精确提取 ROI**")
                    
                    base_x, base_y, base_w, base_h = st.session_state['zoom_coords']
                    cropped_roi = frame_rgb[base_y:base_y+base_h, base_x:base_x+base_w]
                    cropped_pil = Image.fromarray(cropped_roi)
                    
                    zoom_canvas_width = 800
                    zoom_scale_ratio = zoom_canvas_width / base_w 
                    zoom_canvas_height = int(base_h * zoom_scale_ratio)

                    canvas_result_zoom = st_canvas(
                        fill_color="rgba(0, 255, 0, 0.3)",
                        stroke_width=1, stroke_color="#00FF00",
                        background_image=cropped_pil,
                        update_streamlit=True,
                        height=zoom_canvas_height, width=zoom_canvas_width,
                        drawing_mode="rect", key="canvas_zoom", 
                    )

                    if canvas_result_zoom.json_data is not None:
                        objects = canvas_result_zoom.json_data["objects"]
                        if objects:
                            obj = objects[-1]
                            local_scale = base_w / zoom_canvas_width 
                            
                            final_x = base_x + int(obj["left"] * local_scale)
                            final_y = base_y + int(obj["top"] * local_scale)
                            final_w = int(obj["width"] * local_scale)
                            final_h = int(obj["height"] * local_scale)

                            st.success(f"✅ 最终 ROI 坐标: X={final_x}, Y={final_y}, W={final_w}, H={final_h}")
                            
                            final_crop = frame_rgb[final_y:final_y+final_h, final_x:final_x+final_w]
                            if final_crop.size > 0:
                                st.image(final_crop, caption="提取的最终画面")
                                
                                # --- 动作：确认坐标并开启 OCR 状态 ---
                                if st.button("🔍 开始数值识别 (OCR)", type="primary"):
                                    st.session_state['show_ocr_module'] = True
                                    st.session_state['current_ocr_coords'] = (final_x, final_y, final_w, final_h)
                                    st.rerun()

# ================= 主函数 =================
def main():
    st.set_page_config(page_title="极速切片与 ROI 提取工具", layout="wide")
    st.title("🎬 视频预处理流水线")
    
    if 'crop_stage' not in st.session_state:
        st.session_state['crop_stage'] = 'full'
    if 'zoom_coords' not in st.session_state:
        st.session_state['zoom_coords'] = None
    if 'clipped_video_path' not in st.session_state:
        st.session_state['clipped_video_path'] = None
    # 初始化 OCR 状态
    if 'show_ocr_module' not in st.session_state:
        st.session_state['show_ocr_module'] = False
    if 'current_ocr_coords' not in st.session_state:
        st.session_state['current_ocr_coords'] = None
    # 初始化动力分析模块状态
    if 'show_kinematic_module' not in st.session_state:
        st.session_state['show_kinematic_module'] = False

    selected_path = render_sidebar()

    if selected_path:
        st.header(f"📁 当前文件: `{selected_path.name}`")
        
        if not st.session_state.get('show_ocr_module'):
            # 步骤 1: 切片
            if not st.session_state['clipped_video_path']:
                render_slicer(selected_path)
            else:
                # 步骤 2: ROI 选择 (只负责设置状态，不直接渲染 OCR)
                render_roi_selector(st.session_state['clipped_video_path'])

        # 步骤 3: 统一执行 OCR 渲染
        # 只要状态为开启且有坐标，就在主流程中渲染
        if st.session_state.get('show_ocr_module') and st.session_state.get('current_ocr_coords'):
            st.divider()
            # 传入切片后的视频路径和选定的坐标
            render_ocr(st.session_state['clipped_video_path'], st.session_state['current_ocr_coords'])

            # 步骤 4: 动力分析
            if st.session_state.get('show_kinematic_module'):
                render_kinematic_analysis(st.session_state.partial_df, {"name": selected_path.name}, residual_threshold=0.3)
            
    else:
        st.info("👈 请先在左侧侧边栏选择视频源文件夹和文件。")

if __name__ == "__main__":
    main()