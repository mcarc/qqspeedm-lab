import streamlit as st
from pathlib import Path
from typing import Optional
from core.recorder import Recorder
from ui.utils import init_app_state

VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.flv'}

def render_recorder(config):
    # ---------------- 录屏功能模块 ----------------
    st.sidebar.header("⏺️ 设备录屏")
            
    # 初始化录屏状态
    if "is_recording" not in st.session_state:
        st.session_state.is_recording = False
    if "scrcpy_process" not in st.session_state:
        st.session_state.scrcpy_process = None

    # 从 config 中获取 scrcpy 路径（请确保你的 config 字典中包含此配置）
    scrcpy_exe_path = config['paths'].get('scrcpy_exe_path', 'scrcpy.exe')
    recorder = Recorder(scrcpy_exe_path)

    record_filename = st.sidebar.text_input("录制文件名:", value=".mp4")
    
    # 录屏控制按钮
    if not st.session_state.is_recording:
        if st.sidebar.button("▶️ 开始录屏", use_container_width=True):
            # 构建输出路径 (默认保存在 video_base_dir 中)
            save_dir = st.session_state.get('base_dir', '.')
            output_filepath = str(Path(save_dir) / record_filename)
            
            # 启动进程并保存到 session_state
            process = recorder.start_recording(output_filepath)
            st.session_state.scrcpy_process = process
            st.session_state.is_recording = True
            st.rerun()  # 刷新 UI 状态
    else:
        st.sidebar.warning("正在录制中...")
        if st.sidebar.button("⏹️ 停止录屏", use_container_width=True):
            with st.spinner("正在保存视频文件..."):
                recorder.stop_recording(st.session_state.scrcpy_process)
            
            # 清理状态
            st.session_state.scrcpy_process = None
            st.session_state.is_recording = False
            st.rerun()  # 刷新 UI，以便新录制的视频能出现在下方的文件列表中


def render_sidebar(config) -> Optional[Path]:
    """渲染侧边栏文件选择器"""

    # 状态初始化
    if "base_dir" not in st.session_state:
        st.session_state.base_dir = config['paths'].get('video_base_dir')

    render_recorder(config)  # 录屏模块放在文件浏览模块上方，方便用户先录制再选择

    st.sidebar.markdown("---")

    # ---------------- 文件浏览模块 ----------------
    st.sidebar.header("📁 文件浏览")

    base_dir_input = st.sidebar.text_input("视频文件夹路径:", value=st.session_state.base_dir)
    base_path = Path(base_dir_input)

    # 处理非法路径，提前返回
    if not base_path.exists() or not base_path.is_dir():
        st.sidebar.error("路径不存在或不是一个有效的文件夹")
        return None

    # 记录合法的路径
    st.session_state.base_dir = str(base_path)

    # 获取视频文件列表 (加入 f.is_file() 防御性检查，防止文件夹带有后缀名)
    files = [
        f.name for f in base_path.iterdir() 
        if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS
    ]

    # 处理空文件夹情况，提前返回
    if not files:
        st.sidebar.warning("该目录下无受支持的视频文件")
        return None

    # 4. 核心逻辑变得非常扁平、清晰
    selected_file = st.sidebar.selectbox("选择视频文件:", files)
    selected_path = base_path / selected_file
    
    # 检查是否切换了文件
    current_file_str = str(selected_path)
    if st.session_state.get('current_source_file_path') != current_file_str:
        init_app_state(current_file_str, force_reset=True)  # 切换文件时重置状态
        
    return selected_path
