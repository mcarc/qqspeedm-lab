import os
import streamlit as st
from pathlib import Path
from ui.utils import reset_canvas
from core.utils import hms_to_seconds, seconds_to_hms
from core.video import VideoProcessor


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
