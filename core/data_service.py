import pandas as pd
import numpy as np
import os
from core.utils import img_path_to_base64
import yaml
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

class DataService:
    @staticmethod
    def has_records(df: pd.DataFrame) -> bool:
        """判定当前内存中是否有有效记录"""
        return df is not None and not df.empty
    
    @staticmethod
    def reset_data():
        """重置数据状态，清空 DataFrame"""
        return pd.DataFrame()
    
    @staticmethod
    def get_frame_range(df: pd.DataFrame):
        """获取数据的最小和最大帧号"""
        if df is None or df.empty:
            return 0, 0
        return int(df['frame_idx'].min()), int(df['frame_idx'].max())

    @staticmethod
    def get_selected_length(df: pd.DataFrame, sel_start: int, sel_end: int):
        """获取选中范围内的数据条数"""
        if df is None or df.empty:
            return 0
        return len(df[(df['frame_idx'] >= sel_start) & (df['frame_idx'] <= sel_end)])

    @staticmethod
    def prepare_display_data(df: pd.DataFrame, start_f: int, end_f: int):
        """
        过滤指定范围的数据，并生成前端展示专用的 DataFrame
        :return: (原始过滤后的 df, 注入 Base64 的展示用 df)
        """
        mask = (df['frame_idx'] >= start_f) & (df['frame_idx'] <= end_f)
        filtered_df = df[mask].copy()

        if filtered_df.empty:
            return filtered_df, filtered_df.copy()

        # 构造展示用的 DataFrame
        display_df = filtered_df.copy()
        display_df['value'] = display_df['value'].astype(str)
        
        if 'file_path' in display_df.columns:
            display_df['image_display'] = display_df['file_path'].apply(img_path_to_base64)

        return filtered_df, display_df

    @staticmethod
    def save_new_data(new_data: list, output_dir: str):
        """将新提取的数据保存到 data_log.csv 中，返回保存后的 DataFrame"""
        df = pd.DataFrame(new_data)
        df.to_csv(os.path.join(output_dir, "data_log.csv"), index=False)
        return df

    @staticmethod
    def merge_and_save_edits(original_df: pd.DataFrame, edited_df: pd.DataFrame, start_f: int, end_f: int, output_dir: str):
        """
        将编辑后的切片数据与原数据合并，重新排序并落盘保存
        :return: (合并后的完整 df, 本次保存的局部 df)
        """
        # 1. 移除前端专用的图片显示列
        if 'image_display' in edited_df.columns:
            save_part_df = edited_df.drop(columns=['image_display'])
        else:
            save_part_df = edited_df

        # 2. 获取未被选中的数据 (Outside the mask)
        mask = (original_df['frame_idx'] >= start_f) & (original_df['frame_idx'] <= end_f)
        remaining_df = original_df[~mask]

        # 3. 拼接：未选中部分 + 编辑过的部分
        final_df = pd.concat([remaining_df, save_part_df])
        
        # 4. 按帧号重新排序，确保数据顺序正确
        final_df = final_df.sort_values(by="frame_idx")

        # 5. 落盘保存
        final_df.to_csv(os.path.join(output_dir, "data_log.csv"), index=False)

        return final_df, save_part_df

class ExperimentDataManager:
    """
    负责实验数据的持久化存储。
    默认目录结构（扁平化）：
    base_dir/
      └── experiment_name (默认为时间戳)/
          ├── selected_data.csv
          ├── vt_plot.pdf
          └── records.yaml
    """

    def __init__(self, base_dir: str = "experiment_results", experiment_name: str = None):
        self.base_dir = base_dir
        # 如果没有提供实验名称，默认使用当前时间戳保证唯一性
        self.experiment_name = experiment_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_dir = os.path.join(self.base_dir, self.experiment_name)
        
        # 初始化时自动创建主目录
        self._setup_directories()

    @staticmethod
    def _sanitize_data(data):
        """
        递归处理字典、列表和元组，将 NumPy 类型转换为 Python 标准类型，
        并将元组转换为列表，以便于 YAML/JSON 序列化。
        """
        if isinstance(data, dict):
            return {k: ExperimentDataManager._sanitize_data(v) for k, v in data.items()}
        elif isinstance(data, (list, tuple, set)):
            return [ExperimentDataManager._sanitize_data(v) for v in data]
        elif isinstance(data, (np.float32, np.float64)):
            return float(data)
        elif isinstance(data, (np.int32, np.int64)):
            return int(data)
        elif isinstance(data, np.ndarray):
            return ExperimentDataManager._sanitize_data(data.tolist())
        else:
            return data

    def _setup_directories(self):
        """创建实验主目录，取消了内部的分类文件夹以保持扁平化。"""
        os.makedirs(self.exp_dir, exist_ok=True)

    def save_dataframe(self, df: pd.DataFrame, filename: str = "selected_data.csv") -> str:
        """保存 DataFrame 为 CSV 文件。"""
        if df is None or df.empty:
            return None
        
        filepath = os.path.join(self.exp_dir, filename)
        df.to_csv(filepath, index=False, encoding='utf-8-sig')
        return filepath

    def save_records(self, records: dict, filename: str = "records.yaml") -> str:
        """保存字典格式的记录数据为 YAML 文件。"""
        if not records:
            return None

        # 🌟 在这里进行数据清洗
        clean_records = self._sanitize_data(records)

        filepath = os.path.join(self.exp_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            # allow_unicode=True 保证中文正常显示
            # sort_keys=False 保持字典写入时的原有顺序
            # default_flow_style=False 确保输出为标准的层级块状 YAML 格式
            yaml.dump(clean_records, f, allow_unicode=True, sort_keys=False, default_flow_style=False)
        return filepath

    def save_figure(self, fig: plt.Figure, filename: str = "vt_plot.pdf") -> str:
        """保存 Matplotlib Figure 为 PDF。"""
        if fig is None:
            return None
            
        filepath = os.path.join(self.exp_dir, filename)
        # 保存为 PDF，支持高质量矢量图
        fig.savefig(filepath, format='pdf', bbox_inches='tight', dpi=300)
        return filepath

    def save_all_results(self, df: pd.DataFrame, records: dict, fig: plt.Figure) -> dict:
        """
        一键保存所有实验数据，返回保存的文件路径字典。
        """
        saved_paths = {
            "data_path": self.save_dataframe(df),
            "records_path": self.save_records(records),
            "plot_path": self.save_figure(fig)
        }
        return saved_paths