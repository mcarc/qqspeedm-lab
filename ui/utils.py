import streamlit as st
from pathlib import Path
from core.data_service import DataService

def init_app_state(new_file_path: str = None, force_reset: bool = False):
    states = {
        'current_source_file_path': new_file_path,
        'current_source_filename': Path(new_file_path).stem if new_file_path else None,

        # roi
        'crop_stage': 'full',
        'zoom_coords': None,
        'current_ocr_coords': None,

        # slicer
        'clipped_video_path': None,
        'clipped_time_range': None,

        # ocr
        'show_ocr_module': False,
        'selected_frame_range': None,

        # kinematics
        'show_kinematic_module': False,

        # data
        'data_df': DataService.reset_data(),
        'selected_df': DataService.reset_data()
    }

    for key, value in states.items():
        if force_reset or key not in st.session_state:
            st.session_state[key] = value

def reset_canvas():
    """重置回全图模式"""
    st.session_state['crop_stage'] = 'full'
    st.session_state['zoom_coords'] = None
