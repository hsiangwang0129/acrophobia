import numpy as np  # 用於數值計算
import math  # 提供數學函數
import pandas as pd  # 用於數據處理和分析
import torch  # 用於高效的向量運算
import json  # 用於處理 JSON 格式數據
import traceback  # 用於處理和格式化異常

# 計算固定注視次數
def fixation_count(data:pd.DataFrame, time:float) -> float:  
    """
    計算用戶的固定注視次數（Fixation Count）。
    固定注視是指視線在一段時間內保持穩定，通常用於衡量專注程度。

    參數:
        data (pd.DataFrame): 包含 FocusNormal 的數據框。
        time (float): 該段數據的總時長（秒）。

    返回:
        float: 固定注視次數除以時間，用於標準化。
    """
    direction_X = []
    direction_Y = []
    direction_Z = []

    # 提取視線方向的 x, y, z 分量 (從 FocusNormal 欄位)
    for _, row in data.iterrows():
        normal = row['FocusNormal']
        if isinstance(normal, dict):
            x, y, z = normal.get('x', 0.0), normal.get('y', 0.0), normal.get('z', 0.0)
        elif isinstance(normal, str):
            try:
                normal_dict = json.loads(normal.replace("'", '"'))
                x, y, z = normal_dict.get('x', 0.0), normal_dict.get('y', 0.0), normal_dict.get('z', 0.0)
            except:
                continue
        else:
            continue
            
        if x == 0.0:  # 過濾無效數據（x = 0.0）
            continue
        direction_X.append(x)
        direction_Y.append(y)
        direction_Z.append(z)
        
    # 將方向數據轉換為 NumPy 陣列
    direction_X = np.array(direction_X)
    direction_Y = np.array(direction_Y)
    direction_Z = np.array(direction_Z)
    
    fixation_count = 0  # 初始化固定注視次數計數器

    # 判斷是否為固定注視
    for i in range(5, len(direction_X)):
        isFixation = True  # 假設當前為固定注視
        for j in range(0, 5):  # 比較當前點與前 5 個點的角度
            vector_dot = direction_X[i] * direction_X[i-j] + direction_Y[i] * direction_Y[i-j] + direction_Z[i] * direction_Z[i-j]
            try:
                # 計算角度（餘弦逆函數）
                angle = float(math.acos(vector_dot)) * 180 / math.pi
            except:
                # 若計算錯誤，將角度設為 0
                angle = float(math.acos(1)) * 180 / math.pi
            if angle >= 3:  # 如果角度大於 3 度，則不屬於固定注視
                isFixation = False
        if isFixation:  # 如果是固定注視，計數器加 1
            fixation_count += 1

    # 計算標準化的固定注視次數
    print("fixation count", fixation_count / time)
    return fixation_count / time

# 計算兩個向量之間的角度
def calculate_angle(v1:torch.Tensor, v2:torch.Tensor) -> float:
    """
    計算兩個向量之間的角度（以度為單位）。

    參數:
        v1 (torch.Tensor): 第一個向量。
        v2 (torch.Tensor): 第二個向量。

    返回:
        float: 兩個向量之間的角度（度）。
    """
    dot_product = torch.dot(v1, v2)  # 計算點積
    length_v1 = torch.norm(v1)  # 計算第一個向量的模
    length_v2 = torch.norm(v2)  # 計算第二個向量的模
    cosine_similarity = dot_product / (length_v1 * length_v2)  # 餘弦相似度
    
    # 確保 cosine_similarity 在有效範圍內 (-1 到 1)
    cosine_similarity = torch.clamp(cosine_similarity, -1.0, 1.0)
    
    radians = torch.acos(cosine_similarity)  # 餘弦逆函數，得到弧度
    degrees = radians * 180 / math.pi  # 將弧度轉換為度
    return degrees

# 累積序列的變化量
def accumulate_on_series(series:list) -> float:
    """
    計算序列中相鄰元素之間變化量的累積和。

    參數:
        series (list): 包含數值的序列。

    返回:
        float: 累積變化量。
    """
    accumulate = 0  # 初始化累積值
    for i in range(len(series)-1):
        accumulate += abs(series[i] - series[i+1])  # 計算相鄰元素的差值並累加
    return accumulate

# 計算視線角度的累積變化量
def accumulate_gaze_angle(data:pd.DataFrame, time:float) -> float:
    """
    計算視線方向的角度累積變化量。

    參數:
        data (pd.DataFrame): 包含 FocusNormal 的數據框。
        time (float): 該段數據的總時長（秒）。

    返回:
        float: 累積變化量除以時間，用於標準化。
    """
    baseVector = torch.tensor([0., 0., 1.]).float()  # 基準向量（z 軸方向）
    angles = []

    # 計算每個視線方向與基準向量的角度
    for _, row in data.iterrows():
        normal = row['FocusNormal']
        if isinstance(normal, dict):
            x, y, z = normal.get('x', 0.0), normal.get('y', 0.0), normal.get('z', 0.0)
        elif isinstance(normal, str):
            try:
                normal_dict = json.loads(normal.replace("'", '"'))
                x, y, z = normal_dict.get('x', 0.0), normal_dict.get('y', 0.0), normal_dict.get('z', 0.0)
            except:
                continue
        else:
            continue
            
        if x == 0.0:  # 過濾無效數據
            continue
        
        gazeDirectionVector = torch.tensor([x, y, z]).float()
        angle = calculate_angle(gazeDirectionVector, baseVector)
        angles.append(angle)

    # 計算角度序列的累積變化量
    if not angles:  # 檢查是否有有效數據
        print("No valid gaze data found")
        return 0.0
        
    output = accumulate_on_series(angles)
    print("accumulate_on_series", output)
    output = output.float().item()  # 將結果轉為浮點數
    print("accumulate_gaze_angle output", output)
    return output / time  # 標準化

# 特徵提取
def feature_extract(df:pd.DataFrame) -> list:
    """
    從數據框中提取特徵。

    參數:
        df (pd.DataFrame): 包含眼動追蹤數據的數據框。

    返回:
        list: 提取的特徵列表。
    """
    result_data = []  # 初始化結果列表
    phases = split_phases(df)  # 將數據分段
    for phase_name, p in phases.items():
        print(f"Extracting features for phase: {phase_name}")
        # 計算該段數據的持續時間
        time = (pd.to_datetime(p['Timestamp'].iloc[-1]) - pd.to_datetime(p['Timestamp'].iloc[0])).total_seconds()
        print(f"Phase duration: {time} seconds")
        temp_result = []

        # 提取固定注視次數特徵
        temp_result.append(fixation_count(p, time))
        # 提取視線角度累積變化量特徵
        temp_result.append(accumulate_gaze_angle(p, time))

        result_data.append(temp_result)  # 將特徵加入結果列表

    return result_data

# 將數據分段
def split_phases(df:pd.DataFrame) -> dict:
    """
    將數據按時間分段。

    參數:
        df (pd.DataFrame): 包含眼動追蹤數據的數據框。

    返回:
        dict: 分段後的數據字典，每個鍵對應一個階段。
    """
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])  # 將時間戳轉換為 datetime 格式
    start_time = df['Timestamp'].iloc[0]  # 獲取起始時間
    df['elapsed_time'] = (df['Timestamp'] - start_time).dt.total_seconds()  # 計算經過時間

    # 定義階段劃分邏輯
    def assign_phase(row):
        if row['elapsed_time'] <= 60:
            return 'Baseline'
        elif 60 < row['elapsed_time'] <= 90:
            return 'High-rise Early'
        else:
            return 'High-rise Late'

    df['phase'] = df.apply(assign_phase, axis=1)  # 添加階段標籤

    # 按階段分組
    phase_dict = {}
    for phase_name, phase_df in df.groupby('phase'):
        phase_dict[phase_name] = phase_df.reset_index(drop=True)

    return phase_dict

# 按時間窗口分段數據
def split_by_time_window(df:pd.DataFrame, window_size:int) -> list:
    """
    將數據按固定時間窗口大小分段。

    參數:
        df (pd.DataFrame): 包含眼動追蹤數據的數據框。
        window_size (int): 時間窗口大小（秒）。

    返回:
        list: 包含分段後數據框的列表。
    """
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])  # 將時間戳轉換為 datetime 格式
    start_time = df['Timestamp'].iloc[0]  # 獲取起始時間
    df['elapsed_time'] = (df['Timestamp'] - start_time).dt.total_seconds()  # 計算經過時間
    
    # 計算總時長
    total_duration = df['elapsed_time'].iloc[-1]
    
    # 計算窗口數量
    num_windows = int(np.ceil(total_duration / window_size))
    
    windows = []
    for i in range(num_windows):
        start = i * window_size
        end = (i + 1) * window_size
        window_df = df[(df['elapsed_time'] >= start) & (df['elapsed_time'] < end)].reset_index(drop=True)
        if not window_df.empty:
            windows.append(window_df)
    
    return windows

# 計算每個時間窗口的特徵
def calculate_window_features(windows:list):
    """
    計算每個時間窗口的特徵。

    參數:
        windows (list): 包含分段後數據框的列表。

    返回:
        list: 每個窗口的特徵列表。
    """
    all_features = []
    
    for i, window_df in enumerate(windows):
        print(f"Processing window {i+1}/{len(windows)}")
        
        # 計算窗口持續時間
        time = (pd.to_datetime(window_df['Timestamp'].iloc[-1]) - 
                pd.to_datetime(window_df['Timestamp'].iloc[0])).total_seconds()
        
        if time <= 0:
            print(f"Warning: Invalid time duration for window {i+1}")
            continue
            
        features = []
        
        # 提取固定注視次數特徵
        features.append(fixation_count(window_df, time))
        
        # 提取視線角度累積變化量特徵
        features.append(accumulate_gaze_angle(window_df, time))
        
        all_features.append(features)
    
    return all_features

# 主程式
if __name__ == "__main__":
    try:
        # 假設 Eye 類能夠載入您的數據
        from Eye import Eye
        with Eye() as eye:
            print("Loading eye tracking data...")
            df = eye.load_focus()
            
            # 確保 FocusNormal 欄位的格式正確
            if 'FocusNormal' in df.columns:
                # 如果是字串格式，嘗試轉換為字典
                if isinstance(df['FocusNormal'].iloc[0], str):
                    try:
                        df['FocusNormal'] = df['FocusNormal'].apply(
                            lambda x: json.loads(x.replace("'", '"')) if isinstance(x, str) else x
                        )
                    except Exception as e:
                        print(f"Warning: Could not parse FocusNormal as JSON: {e}")
            else:
                print("Error: Required column 'FocusNormal' not found in data")
                exit(1)
            
            # 設置時間窗口大小（秒）
            window_size = 30
            
            # 按時間窗口分段數據
            print(f"Splitting data into {window_size}-second windows...")
            windows = split_by_time_window(df, window_size)
            print(f"Created {len(windows)} time windows")
            
            # 計算每個時間窗口的特徵
            print("Calculating features for each time window...")
            window_features = calculate_window_features(windows)
            print(f"Extracted features for {len(window_features)} windows")
            
            # 分階段處理
            print("Splitting data into phases...")
            phases = split_phases(df)
            
            # 處理每個階段
            phase_features = {}
            for phase_name, phase_df in phases.items():
                print(f"Processing phase: {phase_name}")
                
                # 計算階段持續時間
                time = (pd.to_datetime(phase_df['Timestamp'].iloc[-1]) - 
                       pd.to_datetime(phase_df['Timestamp'].iloc[0])).total_seconds()
                
                # 計算階段特徵
                features = []
                features.append(fixation_count(phase_df, time))
                features.append(accumulate_gaze_angle(phase_df, time))
                
                phase_features[phase_name] = features
                
                print(f"Phase {phase_name} calculation completed.")
                print(f"Duration: {time} seconds")
                print(f"Features: {features}")
            
            # 輸出結果摘要
            print("\nResults Summary:")
            print("================")
            print("Phase Features:")
            for phase_name, features in phase_features.items():
                print(f"{phase_name}: Fixation Count = {features[0]:.4f}, Gaze Angle Change = {features[1]:.4f}")
            
            print("\nWindow Features (first 5):")
            for i, features in enumerate(window_features[:5]):
                print(f"Window {i+1}: Fixation Count = {features[0]:.4f}, Gaze Angle Change = {features[1]:.4f}")
            
            print("All phases processed successfully.")
            
    except Exception as e:
        print(f"Error in main execution: {e}")
        print(traceback.format_exc())
