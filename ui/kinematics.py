import streamlit as st
from core.kinematics import KinematicAnalyzer
from core.data_service import ExperimentDataManager

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
        conf_threshold=conf_threshold,
        r2_min=r2_min,
        residual_threshold=residual_threshold
    )
    
    # 2. 核心处理 (使用 spinner 提供更好的 UX)
    with st.spinner("正在进行数据清洗与 RANSAC 拟合..."):
        is_success, clean_df, metrics = analyzer.process_and_fit(df)
        
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

    # 6. 可视化图表渲染
    st.markdown("### 速度-时间 (V-T) 拟合图")
    # 因为您的 plot_vt_graph 已经修改为返回 fig，直接接收并用 st.pyplot 渲染
    # fig = analyzer.plot_static(clean_df, metrics)
    fig = analyzer.plot_interactive(clean_df, metrics)
    
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True, theme=None)
    else:
        st.error("未能生成可视化图表。")

    # 7. 实验数据管理
    static_fig = analyzer.plot_static(clean_df, metrics)

    records_to_save = {
        "video_meta": video_meta, # convert tuple to list
        "exp_params": exp_params,
        "metrics": metrics,
    }

    exp_manager = ExperimentDataManager()
    exp_manager.save_all_results(clean_df, records_to_save, static_fig)

    # 将结果返回给上层，方便进行数据保存或跨组件联动
    return is_success, clean_df, metrics
