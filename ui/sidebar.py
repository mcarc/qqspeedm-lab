import streamlit as st
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, Dict
from ui.utils import reset_canvas


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