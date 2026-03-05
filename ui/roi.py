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

def render_preset_selector(clipped_video_path: str):
    """
    渲染预设坐标输入界面。
    用户可以选择预设坐标或手动输入坐标，解析后显示 ROI 预览。
    """
    PRESETS = {
        "预设 1: 底部数值区": "X=820, Y=900, W=97, H=46",
        "预设 2: 左上角状态": "X=50, Y=50, W=150, H=150",
        "预设 3: 仪表盘区域": "X=1600, Y=900, W=200, H=100",
        "自定义手动输入": ""
    }

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

def render_manual_selector(clipped_video_path: str):
    """
    渲染手动框选界面。
    用户可以在视频帧上直接框选 ROI 区域，实时预览并确认。
    """
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