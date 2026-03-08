import streamlit as st
import uuid
from core.kinematics import KinematicAnalyzer
from core.data_service import ExperimentDataManager

# 1. 缓存计算函数，并额外返回一个唯一指纹 (run_id)
@st.cache_data(show_spinner=False)
def run_analysis_and_plot(df, video_meta_dict, exp_params):
    analyzer = KinematicAnalyzer(
        video_meta=video_meta_dict,
        **exp_params
    )
    
    is_success, clean_df, metrics = analyzer.process_and_fit(df)
    
    if "error" in metrics:
         return is_success, clean_df, metrics, None, None, None, None

    # 提前生成图表
    fig_vt = analyzer.plot_vt_interactive(clean_df, metrics)
    acc_df, trend_metrics = analyzer.analyze_acceleration_trend(clean_df)
    fig_acc = analyzer.plot_acceleration_interactive(acc_df, trend_metrics)
    static_fig = analyzer.plot_vt_static(clean_df, metrics)

    # 只有在缓存未命中（即 df 或参数发生变化，导致重新执行此函数）时，才会生成一个新的 UUID。如果缓存命中，它会直接返回上一次缓存的旧 UUID。
    run_id = str(uuid.uuid4())

    return is_success, clean_df, metrics, fig_vt, fig_acc, static_fig, run_id

def render_kinematic_analysis(df):
    """
    供上层 Streamlit 调用的运动学分析组件。
    
    :param df: 包含 'value', 'video_timestamp' (可能包含 'confidence') 的 DataFrame
    :return: (is_success, clean_df, metrics) 供上层函数做后续业务逻辑判断
    """
    st.subheader("📈 动力数据分析")
    
    video_meta = {
        'path': st.session_state.get('current_source_file_path', '未知视频'),
        'name': st.session_state.get('current_source_filename', '未知视频'),
        'clipped_time_range': st.session_state.get('clipped_time_range'),
        'selected_frame_range': st.session_state.get('selected_frame_range'),
        'ocr_coords': st.session_state.get('current_ocr_coords'),
    }

    with st.expander("⚙️ 高级分析参数设置", expanded=False):
        col_p1, col_p2, col_p3 = st.columns(3)
        
        with col_p1:
            conf_threshold = st.number_input(
                "置信度阈值 (conf_threshold)", 
                min_value=0.0, 
                max_value=1.0, 
                value=0.0, 
                step=0.05,
                help="用于过滤低置信度数据的阈值（0.0 - 1.0）"
            )
            
        with col_p2:
            r2_min = st.number_input(
                "R² 最小阈值 (r2_min)", 
                min_value=0.0, 
                max_value=1.0, 
                value=0.999, 
                step=0.001,
                help="判断线性拟合是否成功的最小 R² 决定系数"
            )
            
        with col_p3:
            residual_threshold = st.number_input(
                "残差阈值 (residual_threshold)", 
                min_value=0.0, 
                value=0.3, 
                step=0.1,
                help="RANSAC 算法用于区分内点和离群点的残差阈值"
            )

    exp_params = {
        "conf_threshold": conf_threshold,
        "r2_min": r2_min,
        "residual_threshold": residual_threshold
    }

    # 1. 实例化分析器
    analyzer = KinematicAnalyzer(
        video_meta=video_meta,
        **exp_params
    )
    
    # 执行核心逻辑（解构出 run_id）
    with st.spinner("正在进行数据清洗与 RANSAC 拟合..."):
        is_success, clean_df, metrics, fig_vt, fig_acc, static_fig, run_id = run_analysis_and_plot(
            df, video_meta, exp_params
        )
        
    # 3. 错误处理与中断
    if "error" in metrics:
        st.error(f"❌ 分析异常: {metrics['error']}")
        # 发生错误时，提前返回
        return False, clean_df, metrics

    # 4. 状态提示
    if is_success:
        st.success(f"✅ 拟合成功！数据符合物理预期 (R² ≥ {r2_min})")
    else:
        st.warning(f"⚠️ 拟合 R² 分数 ({metrics.get('r_squared', 0):.4f}) 低于设定阈值 ({r2_min})，请参考下方图表排查数据质量。")

    # 5. 使用 Streamlit 原生 Metric 组件展示核心指标
    st.markdown("### 核心指标摘要")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric(label="动力 ($a$)", value=f"{metrics['acceleration']:.2f} (km/h)/s")
    with col2:
        st.metric(label="初速度 ($v_0$)", value=f"{metrics['start_velocity']:.1f} km/h")
    with col3:
        st.metric(label="末速度 ($v_1$)", value=f"{metrics['end_velocity']:.1f} km/h")
    with col4:
        st.metric(label="$R^2$ 拟合优度", value=f"{metrics['r_squared']:.6f}")
    with col5:
        st.metric(label="有效内点 / 离群点", value=f"{metrics['inliers_count']} / {metrics['outliers_count']}")

    # 直接渲染缓存好的图表
    st.markdown("### 速度-时间 (V-T) 拟合图")
    if fig_vt is not None:
        st.plotly_chart(fig_vt, use_container_width=True, theme=None)

    st.markdown("### 速度差分-时间 (A-T) 拟合图")
    if fig_acc is not None:
        st.plotly_chart(fig_acc, use_container_width=True, theme=None)

    # ---------------------------------------------------------
    # 静默自动保存逻辑
    # ---------------------------------------------------------
    # 确保拟合成功，且获取到了有效的 run_id
    if is_success and run_id is not None:
        # 检查这个版本的分析结果是否已经保存过
        if st.session_state.get('last_saved_run_id') != run_id:
            # 如果没保存过（说明是新产生的数据，或者参数被修改了），执行真正的写入动作
            records_to_save = {
                "video_meta": video_meta,
                "exp_params": exp_params,
                "metrics": metrics,
            }
            exp_manager = ExperimentDataManager()
            exp_manager.save_all_results(clean_df, records_to_save, static_fig)
            
            # 将当前 run_id 写入 session_state，标记为“已落盘”
            st.session_state['last_saved_run_id'] = run_id
            
            # (可选) 可以在这里闪现一个短暂的成功提示，让用户知道后台保存了
            st.toast("💾 数据已自动保存")

    return is_success, clean_df, metrics