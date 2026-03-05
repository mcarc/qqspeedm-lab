import streamlit as st
from kinematic_backend import KinematicAnalyzer

def render_kinematic_analysis(df, video_meta, conf_threshold=0.0, r2_min=0.99, residual_threshold=None):
    """
    供上层 Streamlit 调用的运动学分析组件。
    
    :param df: 包含 'value', 'video_timestamp' (可能包含 'confidence') 的 DataFrame
    :param video_meta: 视频元数据字典
    :param conf_threshold: OCR 置信度过滤阈值
    :param r2_min: 最小 R² 分数要求
    :param residual_threshold: RANSAC 残差容忍度
    :return: (is_success, clean_df, metrics) 供上层函数做后续业务逻辑判断
    """
    st.subheader("### 📈 动力数据分析")
    
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
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="加速度", value=f"{metrics['acceleration']:.2f} (km/h)/s")
    with col2:
        st.metric(label="初速度 ($v_0$)", value=f"{metrics['v0']:.2f} km/h")
    with col3:
        st.metric(label="$R^2$ 拟合优度", value=f"{metrics['r_squared']:.4f}")
    with col4:
        st.metric(label="有效内点 / 离群点", value=f"{metrics['inliers_count']} / {metrics['outliers_count']}")

    # 6. 可视化图表渲染
    st.markdown("### 速度-时间 (V-T) 拟合图")
    # 因为您的 plot_vt_graph 已经修改为返回 fig，直接接收并用 st.pyplot 渲染
    fig = analyzer.plot_vt_graph(clean_df, metrics)
    
    if fig is not None:
        st.pyplot(fig)
        # 渲染后清理内存，防止在 Streamlit 长时间运行中出现内存泄漏
        # fig.clf() 
    else:
        st.error("未能生成可视化图表。")

    # 将结果返回给上层，方便进行数据保存或跨组件联动
    return is_success, clean_df, metrics