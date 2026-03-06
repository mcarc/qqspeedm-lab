import streamlit as st
from pathlib import Path
from typing import Optional
from ui.utils import reset_app_state

DEFAULT_BASE_DIR = "D:\\Videos"
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.flv'}

def render_sidebar() -> Optional[Path]:
    """渲染侧边栏文件选择器"""
    st.sidebar.header("📁 文件浏览")
    
    # 状态初始化
    if "base_dir" not in st.session_state:
        st.session_state.base_dir = DEFAULT_BASE_DIR

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
    if st.session_state.get('current_source_file') != current_file_str:
        reset_app_state(current_file_str)
        
    return selected_path