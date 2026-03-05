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
from ui.utils import reset_canvas
from core.utils import hms_to_seconds, seconds_to_hms, parse_roi_string
from core.video import VideoProcessor

# 1. 提取常量：避免每次函数调用时重复创建
ROI_PRESETS = {
    "预设 1: 底部数值区": "X=820, Y=900, W=97, H=46",
    "预设 2: 左上角状态": "X=50, Y=50, W=150, H=150",
    "预设 3: 仪表盘区域": "X=1600, Y=900, W=200, H=100",
    "自定义手动输入": ""
}

# 2. 抽离视频读取逻辑并使用缓存：极大提升 UI 流畅度
@st.cache_data(show_spinner=False, max_entries=5)
def get_video_first_frame(video_path: str):
    """读取视频第一帧并缓存，避免 Streamlit 频繁触发磁盘 I/O"""
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

@st.cache_data(show_spinner=False)
def get_video_frame_count(video_path: str) -> int:
    """获取视频总帧数并缓存"""
    cap = cv2.VideoCapture(video_path)
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return count

@st.cache_data(show_spinner=False, max_entries=20)
def get_video_frame(video_path: str, frame_no: int):
    """读取指定帧并缓存，避免拖动 slider 和画框时重复读取硬盘"""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

def render_preset_selector(clipped_video_path: str):
    """
    渲染预设坐标输入界面。
    用户可以选择预设坐标或手动输入坐标，解析后显示 ROI 预览。
    """
    st.markdown("### ✏️ 直接输入或选择坐标")
    
    p_col1, p_col2 = st.columns(2)
    with p_col1:
        preset_choice = st.selectbox("快捷选项", list(ROI_PRESETS.keys()))
    with p_col2:
        default_val = ROI_PRESETS.get(preset_choice, "")
        roi_input = st.text_input("坐标参数", value=default_val, placeholder="例如: X=820, Y=900, W=97, H=46")

    # 底部通用按钮 (抽离到外层，无论上方逻辑如何都应显示)
    if st.button("🔄 重新选择视频时间段"):
        st.session_state['clipped_video_path'] = None
        st.rerun()

    # 3. 卫语句：处理空输入情况，提前结束逻辑
    if not roi_input.strip():
        return

    coords = parse_roi_string(roi_input)
    
    # 4. 卫语句：处理解析失败情况
    if not coords:
        st.error("格式解析失败！请确保格式为: X=数字, Y=数字, W=数字, H=数字")
        return

    x, y, w, h = coords
    st.success(f"坐标解析成功: X={x}, Y={y}, W={w}, H={h}")
    
    # 获取视频帧
    frame_rgb = get_video_first_frame(clipped_video_path)
    if frame_rgb is None:
        st.error("无法读取视频画面，请检查视频文件是否有效。")
        return

    # 5. 显式边界验证：代替原本的 try-except 和 size > 0 检查
    img_h, img_w = frame_rgb.shape[:2]
    if x < 0 or y < 0 or x + w > img_w or y + h > img_h or w <= 0 or h <= 0:
        st.error(f"裁剪区域超出视频画面范围或无效！(当前画面尺寸: {img_w}x{img_h})")
        return

    # 裁剪并显示预览
    final_crop = frame_rgb[y:y+h, x:x+w]
    st.image(final_crop, caption="直接提取的 ROI 预览")
    
    # --- 动作：确认坐标并开启 OCR 状态 ---
    if st.button("🔍 确认区域", type="primary", key="btn_ocr_preset"):
        st.session_state['show_ocr_module'] = True
        st.session_state['current_ocr_coords'] = (x, y, w, h)
        st.rerun()

def render_manual_selector(clipped_video_path: str):
    """
    渲染手动框选界面。
    用户可以在视频帧上直接框选 ROI 区域，实时预览并确认。
    """
    col1, col2 = st.columns([1, 3])
    
    # --- 左侧列：视频预览与重置 ---
    with col1:
        st.video(clipped_video_path)
        if st.button("🔄 重新选择视频时间段"):
            st.session_state['clipped_video_path'] = None
            st.rerun()
            
    # --- 右侧列：核心画框逻辑 ---
    with col2:
        total_frames = get_video_frame_count(clipped_video_path)
        if total_frames <= 0:
            st.error("无法获取视频长度，请检查视频文件。")
            return

        # 获取用户选择的帧号
        frame_no = st.slider(
            "拖动选择用于画框的视频帧", 
            0, max(0, total_frames - 1), 0, 
            on_change=reset_canvas
        )

        # 读取帧数据 (利用缓存)
        frame_rgb = get_video_frame(clipped_video_path, frame_no)
        
        # 卫语句：如果帧读取失败，直接中断后续逻辑
        if frame_rgb is None:
            st.error("无法读取当前帧，请尝试选择其他帧。")
            return

        full_img_pil = Image.fromarray(frame_rgb)
        raw_w, raw_h = full_img_pil.size
        st.caption(f"当前帧分辨率: {raw_w}x{raw_h}")

        if st.button("🔙 取消放大 / 重置视图"):
            reset_canvas()
            st.rerun()

        # ==========================================
        # 步骤 1: 粗略框选
        # ==========================================
        if st.session_state.get('crop_stage', 'full') == 'full':
            st.markdown("**第一步：粗略框选需要放大的区域**")

            canvas_width = min(800, raw_w)
            canvas_height = int(raw_h * (canvas_width / raw_w))
            
            canvas_result = st_canvas(
                fill_color="rgba(255, 165, 0, 0.3)",
                stroke_width=2, stroke_color="#FF0000",
                background_image=full_img_pil,
                update_streamlit=True,
                height=canvas_height, width=canvas_width,
                drawing_mode="rect", key="canvas_full",
            )

            # 解析画板数据
            if canvas_result.json_data and canvas_result.json_data["objects"]:
                obj = canvas_result.json_data["objects"][-1]
                scale = raw_w / canvas_width
                z_w, z_h = int(obj["width"] * scale), int(obj["height"] * scale)
                
                # 过滤过小的无效框选
                if z_w > 10 and z_h > 10:
                    st.session_state['zoom_coords'] = (
                        int(obj["left"] * scale), int(obj["top"] * scale), z_w, z_h
                    )
                    st.session_state['crop_stage'] = 'zoomed'
                    st.rerun() 

        # ==========================================
        # 步骤 2: 精确提取
        # ==========================================
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

            if canvas_result_zoom.json_data and canvas_result_zoom.json_data["objects"]:
                obj = canvas_result_zoom.json_data["objects"][-1]
                local_scale = base_w / zoom_canvas_width 
                
                final_x = base_x + int(obj["left"] * local_scale)
                final_y = base_y + int(obj["top"] * local_scale)
                final_w = int(obj["width"] * local_scale)
                final_h = int(obj["height"] * local_scale)

                st.success(f"✅ 最终 ROI 坐标: X={final_x}, Y={final_y}, W={final_w}, H={final_h}")
                
                # 边界保护与预览展示
                final_crop = frame_rgb[final_y:final_y+final_h, final_x:final_x+final_w]
                if final_crop.size > 0:
                    st.image(final_crop, caption="提取的最终画面")
                    
                    if st.button("🔍 开始数值识别 (OCR)", type="primary"):
                        st.session_state['show_ocr_module'] = True
                        st.session_state['current_ocr_coords'] = (final_x, final_y, final_w, final_h)
                        st.rerun()
                else:
                    st.warning("框选区域无效，请重新画框。")

def render_roi_selector(clipped_video_path: str):
    """
    渲染 ROI 选择器（预设坐标 或 手动框选）。
    此函数负责更新 session_state 中的坐标和标志位。
    """
    st.divider()

    st.subheader("🎯 阶段二：提取 ROI")
    
    roi_mode = st.radio(
        "选择提取方式:", 
        ["使用预设坐标", "手动在画面中框选"], 
        horizontal=True
    )

    # ---------------- 分支 A: 使用预设坐标 ----------------
    if roi_mode == "使用预设坐标":
        render_preset_selector(clipped_video_path)
    # ---------------- 分支 B: 手动在画面中框选 ----------------
    else:
        render_manual_selector(clipped_video_path)