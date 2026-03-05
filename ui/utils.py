import streamlit as st

def reset_canvas():
    """重置回全图模式"""
    st.session_state['crop_stage'] = 'full'
    st.session_state['zoom_coords'] = None
