import streamlit as st
from pathlib import Path
import pandas as pd
import os
from core.ocr import OcrProcessor
from core.video import VideoProcessor
from core.utils import img_path_to_base64, get_video_frame

# 配置项
OUTPUT_DIR = "tmp/ocr_results"
CSV_PATH = os.path.join(OUTPUT_DIR, "data_log.csv")
# VIDEO_SOURCE = 'temp_clipped.mp4'  # 将视频路径提取为常量，方便多处调用

os.makedirs(OUTPUT_DIR, exist_ok=True)

@st.cache_resource
def get_frame_processor():
    return OcrProcessor()

import streamlit as st

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

def render_ocr(video_path, roi):
    # st.set_page_config(page_title="OCR 批处理工具", layout="wide")
    # st.title("📂 视频数值提取")

    # --- 1. 初始化 Session State ---
    # Has initialized in main.py.
    # if "data_df" not in st.session_state:
    #     if os.path.exists(CSV_PATH):
    #         try:
    #             st.session_state.data_df = pd.read_csv(CSV_PATH)
    #         except:
    #             st.session_state.data_df = pd.DataFrame()
    #     else:
    #         st.session_state.data_df = pd.DataFrame()

    status_container = st.container()

    # --- 2. 自动触发与重跑控制区 ---
    
    # 逻辑开关
    should_run = False
    
    # 情况 A：Session 中完全没有数据（初次进入或被主模块清空）
    if st.session_state.data_df.empty:
        should_run = True

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
            frame_processor = get_frame_processor()
            
            # new_data = process_video(processor, video_path, roi, status_container)
            video_processor = VideoProcessor(Path(video_path))
            new_data = render_video_processor(video_processor, frame_processor, roi, status_container, OUTPUT_DIR)

            if new_data:
                df = pd.DataFrame(new_data)
                df.to_csv(CSV_PATH, index=False)
                st.session_state.data_df = df
                st.rerun() # 跑完直接刷新，进入结果展示阶段
            else:
                st.error("未能提取到有效数据，请检查 ROI 区域。")

    # --- 3. 结果展示与编辑区 ---
    if not st.session_state.data_df.empty:
        st.subheader("🔍 范围选择与预览")

        df_all = st.session_state.data_df
        min_f = int(df_all['frame_idx'].min())
        max_f = int(df_all['frame_idx'].max())

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
            
            st.info(f"当前选中范围包含 {len(df_all[(df_all['frame_idx'] >= sel_start) & (df_all['frame_idx'] <= sel_end)])} 条数据")

        with preview_col:
            # 获取起始帧的完整画面
            preview_img = get_video_frame(video_path, sel_start)
            if preview_img is not None:
                st.image(preview_img, caption=f"当前起始帧: {sel_start} (完整画面)", use_column_width=True)
            else:
                st.warning("无法加载视频预览，请确认视频文件是否存在。")

        st.divider()
        st.subheader("📝 选中范围内数据编辑")

        # --- 数据过滤逻辑 ---
        # 仅筛选出范围内的数据用于显示和编辑
        mask = (df_all['frame_idx'] >= sel_start) & (df_all['frame_idx'] <= sel_end)
        filtered_df = df_all[mask].copy()

        if filtered_df.empty:
            st.warning("当前范围内没有数据。")
        else:
            # 构造显示用的 DataFrame (增加 base64 图片列)
            display_df = filtered_df.copy()
            display_df['value'] = display_df['value'].astype(str)
            
            if 'file_path' in display_df.columns:
                display_df['image_display'] = display_df['file_path'].apply(img_path_to_base64)

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

            # --- 4. 保存逻辑 (拼接回未选中部分) ---
            if submit_btn:
                # 1. 处理编辑后的数据：移除临时的图片显示列
                if 'image_display' in edited_filtered_df.columns:
                    save_part_df = edited_filtered_df.drop(columns=['image_display'])
                else:
                    save_part_df = edited_filtered_df

                # 2. 获取未被选中的数据 (Outside the mask)
                # 注意：使用原始 df_all 的 mask 取反
                remaining_df = df_all[~mask]

                # 3. 拼接：未选中部分 + 编辑过的部分
                final_df = pd.concat([remaining_df, save_part_df])
                
                # 4. 按帧号重新排序，确保数据顺序正确
                final_df = final_df.sort_values(by="frame_idx")

                # 5. 保存并更新 Session State
                final_df.to_csv(CSV_PATH, index=False)
                st.session_state.data_df = final_df
                st.session_state.partial_df = save_part_df
                st.session_state.show_kinematic_module = True # 触发动力分析模块显示
                
                st.success(f"✅ 保存成功！已更新 {len(save_part_df)} 条记录，总记录数 {len(final_df)}。")
                # 稍微延迟后重跑，或者让用户手动刷新，这里不强制 rerun 避免 form 提交后的逻辑中断
                # st.rerun() 

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