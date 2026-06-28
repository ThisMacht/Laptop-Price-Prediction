import pandas as pd
import numpy as np
import re

def clean_newlines(df):
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.replace('\n', ', ', regex=False)
            df[col] = df[col].str.strip()
            df[col] = df[col].replace('nan', np.nan)
    return df

def clean_manufacturer(df):
    if 'Hãng sản xuất' in df.columns:
        df['Hãng sản xuất'] = df['Hãng sản xuất'].apply(lambda x: str(x).split(' ')[0].strip() if pd.notnull(x) else np.nan)
        df['Hãng sản xuất'] = df['Hãng sản xuất'].str.capitalize()
    return df

def clean_ram(df):
    if 'Dung lượng RAM' in df.columns:
        df['Dung lượng RAM'] = df['Dung lượng RAM'].astype(str).str.replace(r'[a-zA-Z\s]', '', regex=True)
        df['Dung lượng RAM'] = pd.to_numeric(df['Dung lượng RAM'], errors='coerce')
    return df

def clean_storage(df):
    if 'Dung lượng ổ cứng' in df.columns:
        def parse_storage(val):
            if pd.isnull(val): return np.nan
            val = str(val).upper().replace(' ', '')
            total_gb = 0
            parts = val.split('+')
            for part in parts:
                num = re.findall(r'\d+', part)
                if not num: continue
                num = int(num[0])
                if 'TB' in part:
                    total_gb += num * 1024
                elif 'GB' in part:
                    total_gb += num
            return total_gb if total_gb > 0 else np.nan

        df['Dung lượng ổ cứng (GB)'] = df['Dung lượng ổ cứng'].apply(parse_storage)
    return df

def clean_resolution(df):
    if 'Độ phân giải' in df.columns:
        def parse_resolution(val):
            if pd.isnull(val): return np.nan, np.nan
            match = re.findall(r'(\d+)\s*[xX*]\s*(\d+)', str(val))
            if match:
                return float(match[0][0]), float(match[0][1])
            return np.nan, np.nan
        
        resolutions = df['Độ phân giải'].apply(parse_resolution)
        df['Độ phân giải ngang (px)'] = [res[0] for res in resolutions]
        df['Độ phân giải dọc (px)'] = [res[1] for res in resolutions]
    return df

def clean_screen_size(df):
    if 'Kích thước' in df.columns:
        df['Kích thước (inch)'] = df['Kích thước'].astype(str).apply(lambda x: re.search(r'\d+(\.\d+)?', x).group() if pd.notnull(x) and re.search(r'\d+(\.\d+)?', x) else np.nan)
        df['Kích thước (inch)'] = pd.to_numeric(df['Kích thước (inch)'], errors='coerce')
    return df

def clean_weight(df):
    if 'Trọng lượng' in df.columns:
        def parse_weight(val):
            if pd.isnull(val): return np.nan
            val = str(val).lower().replace(',', '.')
            match = re.search(r'(\d+(\.\d+)?)', val)
            if not match: return np.nan
            num = float(match.group())
            if 'g' in val and 'kg' not in val and num > 30:
                num = num / 1000
            return num
        df['Trọng lượng (kg)'] = df['Trọng lượng'].apply(parse_weight)
    return df

def clean_cpu_speed(df):
    for col in ['Tốc độ CPU', 'Tốc độ tối đa']:
        if col in df.columns:
            new_col = col + ' (GHz)'
            df[new_col] = df[col].astype(str).apply(lambda x: re.search(r'\d+(\.\d+)?', x).group() if pd.notnull(x) and re.search(r'\d+(\.\d+)?', x) else np.nan)
            df[new_col] = pd.to_numeric(df[new_col], errors='coerce')
    return df

def clean_dimensions(df):
    if 'Kích thước_2' in df.columns:
        def parse_dimensions(val):
            if pd.isnull(val): return np.nan, np.nan, np.nan
            val = str(val).lower().replace(',', '.')
            parts = re.findall(r'(\d+(\.\d+)?)', val)
            nums = [float(p[0]) for p in parts]
            if len(nums) >= 3:
                nums = sorted(nums[:3], reverse=True)
                return nums[0], nums[1], nums[2]
            return np.nan, np.nan, np.nan
        
        dims = df['Kích thước_2'].apply(parse_dimensions)
        df['Chiều dài (mm)'] = [d[0] for d in dims]
        df['Chiều rộng (mm)'] = [d[1] for d in dims]
        df['Độ dày (mm)'] = [d[2] for d in dims]
    return df

def clean_gpu(df):
    if 'Bộ xử lý' in df.columns:
        def extract_gpu(gpu):
            if pd.isnull(gpu): return 'Unknown'
            gpu_str = str(gpu).upper()
            if 'RTX' in gpu_str:
                match = re.search(r'RTX\s*\d+', gpu_str)
                return match.group() if match else 'RTX Series'
            elif 'GTX' in gpu_str:
                match = re.search(r'GTX\s*\d+(TI)?', gpu_str)
                return match.group() if match else 'GTX Series'
            elif 'IRIS' in gpu_str:
                return 'Intel Iris Xe'
            elif 'UHD' in gpu_str:
                return 'Intel UHD Graphics'
            elif 'RADEON' in gpu_str:
                if 'RX' in gpu_str:
                    match = re.search(r'RX\s*\d+', gpu_str)
                    return match.group() if match else 'AMD Radeon RX'
                return 'AMD Radeon Graphics'
            elif 'ARC' in gpu_str:
                return 'Intel Arc Graphics'
            elif 'APPLE' in gpu_str or 'CORE' in gpu_str:
                return 'Apple GPU'
            return 'Other GPU'
        
        df['Đồ họa đã làm sạch'] = df['Bộ xử lý'].apply(extract_gpu)
    return df

def format_prices(df):
    for col in ['shop_1_price', 'shop_2_price', 'shop_3_price']:
        if col in df.columns:
             df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def fill_missing(df):
    cat_cols_to_fill = ['Webcam', 'Đèn bàn phím', 'Tính năng khác', 'Công nghệ âm thanh', 'Đồ họa đã làm sạch', 'Hệ điều hành']
    for col in cat_cols_to_fill:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")
    return df

def drop_unnecessary_columns(df):
    cols_to_drop = ['Dung lượng ổ cứng', 'Độ phân giải', 'Kích thước', 'Trọng lượng', 'Kích thước_2', 'Tốc độ CPU', 'Tốc độ tối đa', 'Bộ xử lý']
    cols_to_drop = [c for c in cols_to_drop if c in df.columns]
    return df.drop(columns=cols_to_drop)
