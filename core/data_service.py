import pandas as pd
import os
from core.utils import img_path_to_base64

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
    def save_new_data(new_data: list, csv_path: str):
        """将新提取的数据保存到 CSV 文件"""
        df = pd.DataFrame(new_data)
        df.to_csv(csv_path, index=False)
        return df

    @staticmethod
    def merge_and_save_edits(original_df: pd.DataFrame, edited_df: pd.DataFrame, start_f: int, end_f: int, csv_path: str):
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
        final_df.to_csv(csv_path, index=False)

        return final_df, save_part_df
