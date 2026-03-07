import streamlit as st
from pathlib import Path
import os
from core.ocr import OcrProcessor
from core.video import VideoProcessor
from core.data_service import DataService

@st.cache_resource
def get_frame_processor(config):
    return OcrProcessor(config)

def render_video_processor(video_processor: VideoProcessor, frame_processor: OcrProcessor, roi, status_container, output_dir):
    """
    前端 UI 逻辑：接收后端生成器的数据并刷新界面。
    """
    progress_bar = status_container.progress(0, text="准备开始...")
    data_list = []
    
    try:
        # 遍历 VideoProcessor 抛出的状态流
        for step_info in video_processor.process_frames_generator(frame_processor, roi, output_dir):
            result = step_info["result"]
            frame_idx = step_info["frame_idx"]
            
            # 收集有效数据
            if result.get('value', '').strip():
                data_list.append(result)
            
            # 控制 UI 刷新频率，避免卡顿 (每 10 帧更新一次 UI)
            if frame_idx % 10 == 0:
                prog = step_info["progress"]
                progress_bar.progress(prog, text=f"处理中... 视频时间: {result.get('video_timestamp', 0)}")
                
        progress_bar.progress(1.0, text="✅ 处理完成！")
        return data_list
        
    except Exception as e:
        # 统一在前端捕获并展示错误信息
        st.error(f"视频处理异常: {e}")
        return None

def render_ocr(video_path, roi, config):

    status_container = st.container()

    # --- 1. 自动触发与重跑控制区 ---
    
    # 逻辑开关
    # 情况 A：Session 中完全没有数据（初次进入或被主模块清空）
    is_data_ready = DataService.has_records(st.session_state.data_df)
    should_run = not is_data_ready

    # 情况 B：保留手动强制重跑按钮
    # 放在顶部的角落，或者数据编辑区的上方
    col_info, col_btn = st.columns([5, 1])
    with col_btn:
        if st.button("🔄 强制重跑", use_container_width=True):
            should_run = True

    # --- 执行核心逻辑 ---
    if should_run:
        # 使用 spinner 做一个极简的加载提示，不占用进度条位置
        with st.spinner("正在后台提取数值..."):
            frame_processor = get_frame_processor(config)
            
            # new_data = process_video(processor, video_path, roi, status_container)
            video_processor = VideoProcessor(Path(video_path))
            new_data = render_video_processor(video_processor, frame_processor, roi, status_container, config['paths']['output_tmp_dir'])

            if new_data:
                df = DataService.save_new_data(new_data, config['paths']['output_tmp_dir'])
                st.session_state.data_df = df
                st.rerun() # 跑完直接刷新，进入结果展示阶段
            else:
                st.error("未能提取到有效数据，请检查 ROI 区域。")

    # --- 2. 结果展示与编辑区 ---
    if DataService.has_records(st.session_state.data_df):
        st.subheader("🔍 范围选择与预览")

        df_all = st.session_state.data_df
        min_f, max_f = DataService.get_frame_range(df_all)

        # --- 新增布局：左侧预览，右侧控制 ---
        preview_col, control_col = st.columns([1, 1])

        with control_col:
            st.markdown("### ⚙️ 设置筛选范围 (帧)")
            # 两个输入框选择起止范围
            sel_start = st.number_input("起始帧 (Start Frame)", min_value=min_f, max_value=max_f, value=min_f, step=1)
            sel_end = st.number_input("结束帧 (End Frame)", min_value=min_f, max_value=max_f, value=max_f, step=1)
            
            # 简单的校验
            if sel_start > sel_end:
                st.warning("起始帧不能大于结束帧")
                sel_end = sel_start

            st.info(f"当前选中范围包含 {DataService.get_selected_length(df_all, sel_start, sel_end)} 条数据")

        with preview_col:
            # 获取起始帧的完整画面
            video_processor = VideoProcessor(Path(video_path))
            preview_img = video_processor.get_video_frame(sel_start)
            if preview_img is not None:
                st.image(preview_img, caption=f"当前起始帧: {sel_start} (完整画面)", use_column_width=True)
            else:
                st.warning("无法加载视频预览，请确认视频文件是否存在。")

        st.divider()
        st.subheader("📝 选中范围内数据编辑")

        # 调用 DataService 处理展示数据
        filtered_df, display_df = DataService.prepare_display_data(df_all, sel_start, sel_end)

        if not DataService.has_records(display_df):
            st.warning("当前范围内没有数据。")
            return

        with st.form(key="editor_form"):
            edited_filtered_df = st.data_editor(
                display_df,
                column_config={
                    "image_display": st.column_config.ImageColumn("ROI截图", width=120),
                    "video_timestamp": st.column_config.TextColumn("视频时间", disabled=True),
                    "value": st.column_config.TextColumn("识别数值", required=True),
                    "confidence": st.column_config.ProgressColumn("置信度", format="%.2f", min_value=0, max_value=1),
                    "frame_idx": st.column_config.NumberColumn("帧号", disabled=True),
                },
                column_order=["frame_idx", "video_timestamp", "image_display", "value", "confidence"],
                use_container_width=True,
                num_rows="dynamic"
            )

            submit_btn = st.form_submit_button("💾 保存范围修正结果", type="primary")

        # --- 3. 保存逻辑交由 DataService 处理 ---
        if submit_btn:
            final_df, save_part_df = DataService.merge_and_save_edits(
                original_df=df_all,
                edited_df=edited_filtered_df,
                start_f=sel_start,
                end_f=sel_end,
                output_dir=config['paths']['output_tmp_dir']
            )
            
            # 更新 Session State (状态管理仍然留在前端)
            st.session_state.data_df = final_df
            st.session_state.selected_df = save_part_df
            st.session_state.selected_frame_range = (sel_start, sel_end)
            st.session_state.show_kinematic_module = True 
            
            st.success(f"✅ 保存成功！已更新 {len(save_part_df)} 条记录，总记录数 {len(final_df)}。")

    else:
        st.info("暂无数据，请点击上方按钮开始提取，或确保目录下存在 data_log.csv")

if __name__ == "__main__":
    st.warning("请从主程序 main.py 运行以获取完整功能。以下为测试模式：")
    # 提供默认值用于测试
    test_video = "temp_clipped.mp4" 
    test_roi = (820, 900, 97, 46)
    if os.path.exists(test_video):
        render_ocr(test_video, test_roi)
    else:
        st.error(f"找不到测试视频 {test_video}")