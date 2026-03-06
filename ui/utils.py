import streamlit as st
from core.data_service import DataService

def reset_app_state(new_file_path: str = None):
    states = {
        'crop_stage': 'full',
        'zoom_coords': None,
        'clipped_video_path': None,
        'current_source_file': new_file_path,
        'current_ocr_coords': None,
        'show_ocr_module': False,
        'show_kinematic_module': False,
        'data_df': DataService.reset_data(),
        'partial_df': DataService.reset_data()
    }

    for key, value in states.items():
        if key not in st.session_state:
            st.session_state[key] = value

def reset_canvas():
    """重置回全图模式"""
    st.session_state['crop_stage'] = 'full'
    st.session_state['zoom_coords'] = None
