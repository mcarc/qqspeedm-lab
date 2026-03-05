import streamlit as st
from ui.sidebar import render_sidebar
from ui.slicer import render_slicer
from ui.roi import render_roi_selector
from ui.ocr import render_ocr
from ui.kinematic import render_kinematic_analysis

def init_session_state():
    """集中处理所有 session_state 的初始化"""
    defaults = {
        'crop_stage': 'full',
        'zoom_coords': None,
        'clipped_video_path': None,
        'show_ocr_module': False,
        'current_ocr_coords': None,
        'show_kinematic_module': False
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# ================= 主函数 =================
def execute_video_pipeline():
    st.set_page_config(page_title="Lab", layout="wide")
    st.title("🎬 实验室")
    
    # 1. 状态初始化
    init_session_state()

    # 2. 侧边栏渲染
    selected_path = render_sidebar()

    # 3. 主页面路由
    if not selected_path:
        st.info("👈 请先在左侧侧边栏选择视频源文件夹和文件。")
        return # 提前返回，减少代码嵌套层级

    st.header(f"📁 当前文件: `{selected_path.name}`")
    
    if not st.session_state.get('show_ocr_module'):
        # 步骤 1: 切片
        if not st.session_state['clipped_video_path']:
            render_slicer(selected_path)
        else:
            # 步骤 2: ROI 选择
            render_roi_selector(st.session_state['clipped_video_path'])

    # 步骤 3: 统一执行 OCR 渲染
    if st.session_state.get('show_ocr_module') and st.session_state.get('current_ocr_coords'):
        st.divider()
        render_ocr(st.session_state['clipped_video_path'], st.session_state['current_ocr_coords'])

        # 步骤 4: 动力分析
        if st.session_state.get('show_kinematic_module'):
            # 注意：确保 partial_df 在到达这里时已经被正确赋值并存在于 session_state 中
            render_kinematic_analysis(st.session_state.partial_df, {"name": selected_path.name}, residual_threshold=0.3)

if __name__ == "__main__":
    execute_video_pipeline()