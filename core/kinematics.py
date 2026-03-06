import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.linear_model import RANSACRegressor, LinearRegression
from sklearn.metrics import r2_score
from core.exceptions import NotInitializedError

plt.style.use('seaborn-v0_8-whitegrid') # 或者 'ggplot'
plt.rcParams['font.sans-serif'] = ['DengXian', 'Microsoft YaHei', 'SimHei'] # 设置中文字体优先级

class KinematicAnalyzer:
    def __init__(self, video_meta, conf_threshold=0, r2_min=0.99, residual_threshold=None):
        """
        :param conf_threshold: OCR 置信度过滤阈值
        :param r2_min: 判定拟合良好的最小 R² 分数 (0~1)
        :param residual_threshold: RANSAC 判定内点的残差容忍度。
                                   如果不设(None)，sklearn会自动计算一个基于中位数绝对偏差(MAD)的合理默认值。
        """
        self.video_meta = video_meta
        self.conf_threshold = conf_threshold
        self.r2_min = r2_min
        self.residual_threshold = residual_threshold
        self.model = None

    def _clean_value(self, val):
        """清洗 OCR 字符串：处理多个小数点或非法字符"""
        if pd.isna(val): 
            return np.nan
            
        s = str(val)
        # 移除非数字和非小数点字符
        s = re.sub(r'[^\d.]', '', s)
        
        parts = s.split('.')
        if len(parts) > 2:
            # 如果有多个小数点，强制将第一个小数点后的所有内容拼作小数部分
            s = f"{parts[0]}.{''.join(parts[1:])}"
        try:
            return float(s)
        except ValueError:
            return np.nan

    def _preprocess(self, df):
        """执行数据清洗与初步过滤"""
        working_df = df.copy()
        
        # 1. 置信度过滤
        # 如果有 confidence 列，则筛选：
        if 'confidence' in working_df.columns:
            working_df = working_df[working_df['confidence'] >= self.conf_threshold]
        
        # 2. 清洗 value 列
        working_df['clean_value'] = working_df['value'].apply(self._clean_value)
        
        # 3. 去除转换后变为 NaN 的废数据
        working_df = working_df.dropna(subset=['clean_value', 'video_timestamp'])
        
        # 4. 按时间戳排序
        working_df = working_df.sort_values(by='video_timestamp').reset_index(drop=True)
        
        return working_df

    def process_and_fit(self, df):
        """核心管线：预处理 -> RANSAC 找内点 -> OLS 回归"""
        clean_df = self._preprocess(df)
        
        if len(clean_df) < 2:
            return False, clean_df, {"error": "有效数据点不足，无法拟合"}

        X = clean_df[['video_timestamp']].values
        y = clean_df['clean_value'].values

        # --- 步骤 A: 使用 RANSAC 抵抗重复帧与离群点 ---
        ransac = RANSACRegressor(
            estimator=LinearRegression(),
            residual_threshold=self.residual_threshold,
            random_state=42 # 固定随机种子以保证结果可复现
        )
        # 
        try:
            ransac.fit(X, y)
        except ValueError as e:
            return False, clean_df, {"error": f"RANSAC 拟合失败: {str(e)}"}

        inlier_mask = ransac.inlier_mask_
        clean_df['is_inlier'] = inlier_mask

        # --- 步骤 B: 对纯净的内点进行 OLS 评估 ---
        X_inliers = X[inlier_mask]
        y_inliers = y[inlier_mask]
        
        if len(X_inliers) < 2:
            return False, clean_df, {"error": "内点过少，无法进行最终回归"}

        final_model = LinearRegression()
        final_model.fit(X_inliers, y_inliers)
        self.model = final_model # 保存模型供画图使用
        
        # 计算评估指标
        y_pred = final_model.predict(X_inliers)
        r2 = r2_score(y_inliers, y_pred)
        acceleration = final_model.coef_[0] * 1000 # 转换为 (km/h)/s 单位

        metrics = {
            'acceleration': acceleration,
            'start_velocity': y_inliers[0],
            'end_velocity': y_inliers[-1],
            'r_squared': r2,
            'inliers_count': len(X_inliers),
            'outliers_count': len(X) - len(X_inliers)
        }

        # --- 步骤 C: 判断是否符合物理预期 ---
        is_success = r2 >= self.r2_min
        
        return is_success, clean_df, metrics

    def plot_static(self, clean_df, metrics, title=None):
        if self.model is None:
            raise NotInitializedError("没有可用的模型进行绘制。")

        # --- 1. 风格设置 ---
        fig, ax = plt.subplots(figsize=(11/1.4, 7/1.4), dpi=100)
        
        # 定义高级感颜色
        color_inlier = '#2471A3'  # 深蓝色
        color_outlier = '#E74C3C' # 珊瑚红
        color_line = '#27AE60'    # 翡翠绿

        inliers = clean_df[clean_df['is_inlier']]
        outliers = clean_df[~clean_df['is_inlier']]

        # --- 2. 绘制数据 ---
        # 内点：圆点，带白色细边框
        ax.scatter(inliers['video_timestamp'], inliers['clean_value'], 
                color=color_inlier, s=35, label='Valid points', 
                alpha=0.6, edgecolors='w', linewidth=0.5, zorder=3)
        
        # 外点：叉号，稍微淡化
        ax.scatter(outliers['video_timestamp'], outliers['clean_value'], 
                color=color_outlier, marker='x', s=40, label='Outliers', 
                alpha=0.8, zorder=2)

        # 拟合直线
        x_min, x_max = clean_df['video_timestamp'].min(), clean_df['video_timestamp'].max()
        line_x = np.array([[x_min], [x_max]])
        line_y = self.model.predict(line_x)
        ax.plot(line_x, line_y, color=color_line, linewidth=2.5, 
                label='Linear Fit', linestyle='-', zorder=4)

        # --- 3. 装饰与标注 ---
        # 更加整洁的指标文本框
        stats_text = (
            f"$\\mathbf{{Stats:}}$\n"
            f"$a$: {metrics['acceleration']:>11.2f} $(km/h)/s$\n"
            f"$v_0$: {metrics['start_velocity']:>10.1f} km/h\n"
            f"$v_1$: {metrics['end_velocity']:>10.1f} km/h\n"
            f"$R^2$: {metrics['r_squared']:>10.5f}\n"
            f"Outliers: {metrics['outliers_count']:>4}"
        )
        
        # 放在左上角，增加虚线框效果
        props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='#CCCCCC')
        ax.text(0.03, 0.96, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', family='monospace', bbox=props)

        # 标题与标签
        final_title = title if title else f"V-T Analysis: {self.video_meta.get('name', 'Unknown')}"
        ax.set_title(final_title, fontsize=14, pad=15, fontweight='bold', loc='left')
        ax.set_xlabel('Time (ms)', fontsize=11, labelpad=10)
        ax.set_ylabel('Velocity (km/h)', fontsize=11, labelpad=10)

        # --- 4. 智能坐标轴缩放 (防止异常值撑大视野) ---
        if not inliers.empty:
            # 获取内点的范围
            x_data = inliers['video_timestamp']
            y_data = inliers['clean_value']
            
            x_min_in, x_max_in = x_data.min(), x_data.max()
            y_min_in, y_max_in = y_data.min(), y_data.max()
            
            # 计算留白 (例如 10%)
            x_margin = (x_max_in - x_min_in) * 0.1 if x_max_in != x_min_in else 1
            y_margin = (y_max_in - y_min_in) * 0.2 if y_max_in != y_min_in else 1
            
            # 设置显示范围，仅参考内点
            ax.set_xlim(x_min_in - x_margin, x_max_in + x_margin)
            ax.set_ylim(y_min_in - y_margin, y_max_in + y_margin)

        # --- 5. 细节微调 ---
        ax.legend(frameon=True, facecolor='white', loc='lower right', fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.5, zorder=1)
        
        # 移除上方和右侧不必要的边框线
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)

        plt.tight_layout()
        return fig

    def plot_interactive(self, clean_df, metrics, title=None):
        if self.model is None:
            raise NotInitializedError("没有可用的模型进行绘制。")

        # --- 1. 风格设置与数据准备 ---
        color_inlier = '#2471A3'  # 深蓝色
        color_outlier = '#E74C3C' # 珊瑚红
        color_line = '#27AE60'    # 翡翠绿

        inliers = clean_df[clean_df['is_inlier']]
        outliers = clean_df[~clean_df['is_inlier']]

        fig = go.Figure()


        # --- 2. 绘制数据 ---
        # 内点：圆点，带白色细边框
        fig.add_trace(go.Scatter(
            x=inliers['video_timestamp'].tolist(), 
            y=inliers['clean_value'].tolist(),
            mode='markers',
            name='Valid points',
            marker=dict(
                color=color_inlier, 
                size=7, 
                opacity=0.6, 
                line=dict(color='white', width=1)
            ),
            hovertemplate="Time: %{x:.1f} ms<br>Velocity: %{y:.2f} km/h<extra></extra>"
        ))

        # 外点：叉号，稍微淡化
        fig.add_trace(go.Scatter(
            x=outliers['video_timestamp'].tolist(), 
            y=outliers['clean_value'].tolist(),
            mode='markers',
            name='Outliers',
            marker=dict(
                color=color_outlier, 
                size=8, 
                symbol='x', 
                opacity=0.8
            ),
            hovertemplate="Time: %{x:.1f} ms<br>Velocity: %{y:.2f} km/h<extra></extra>"
        ))

        # 拟合直线
        x_min, x_max = clean_df['video_timestamp'].min(), clean_df['video_timestamp'].max()
        line_x = np.array([[x_min], [x_max]])
        line_y = self.model.predict(line_x).flatten() # 展平以匹配 Plotly 要求
        
        print(f"{line_x.flatten()}, {line_y}")

        fig.add_trace(go.Scatter(
            x=line_x.flatten().tolist(), 
            y=line_y.tolist(),
            mode='lines',
            name='Linear Fit',
            line=dict(color=color_line, width=2.5)
        ))

        # --- 3. 装饰与标注 ---
        # Plotly 注释支持 HTML 标签，用于格式化下标和上标
        stats_text = (
            f"<b>Stats:</b><br>"
            f"a: {metrics['acceleration']:>9.2f} (km/h)/s<br>"
            f"v<sub>0</sub>: {metrics['start_velocity']:>8.1f} km/h<br>"
            f"v<sub>1</sub>: {metrics['end_velocity']:>8.1f} km/h<br>"
            f"R<sup>2</sup>: {metrics['r_squared']:>8.6f}<br>"
            f"Outliers: {metrics['outliers_count']}"
        )

        # 在左上角添加统计文本框
        fig.add_annotation(
            xref="paper", yref="paper",
            x=0.02, y=0.98,
            text=stats_text,
            showarrow=False,
            font=dict(family="monospace", size=12, color="#333333"),
            align="left",
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor="#CCCCCC",
            borderwidth=1,
            borderpad=6
        )

        # --- 4. 智能坐标轴缩放 (防止异常值撑大视野) ---
        x_range, y_range = None, None
        if not inliers.empty:
            x_min_in, x_max_in = inliers['video_timestamp'].min(), inliers['video_timestamp'].max()
            y_min_in, y_max_in = inliers['clean_value'].min(), inliers['clean_value'].max()
            
            x_margin = (x_max_in - x_min_in) * 0.1 if x_max_in != x_min_in else 1
            y_margin = (y_max_in - y_min_in) * 0.2 if y_max_in != y_min_in else 1
            
            x_range = [x_min_in - x_margin, x_max_in + x_margin]
            y_range = [y_min_in - y_margin, y_max_in + y_margin]

        # --- 5. 整体布局微调 ---
        final_title = title if title else f"V-T Analysis: {self.video_meta.get('name', 'Unknown')}"
        
        fig.update_layout(
            title=dict(text=final_title, font=dict(size=18), x=0.02),
            xaxis_title="Time (ms)",
            yaxis_title="Velocity (km/h)",
            plot_bgcolor='white', # 移除默认的灰色背景
            xaxis=dict(
                range=x_range,
                showgrid=True,
                gridcolor='rgba(0,0,0,0.1)',
                zeroline=False
            ),
            yaxis=dict(
                range=y_range,
                showgrid=True,
                gridcolor='rgba(0,0,0,0.1)',
                zeroline=False
            ),
            legend=dict(
                x=0.98, y=0.02,
                xanchor='right', yanchor='bottom',
                font=dict(color='#333333'),
                bgcolor='rgba(255,255,255,0.9)',
                bordercolor='#CCCCCC',
                borderwidth=1
            ),
            margin=dict(l=60, r=40, t=60, b=50)
        )

        return fig

    def _fallback_analysis(self, clean_df):
        """如果拟合失败（比如 R² 太低）的后备方案"""
        raise NotImplementedError("拟合失败的后备分析方案尚未实现。")
