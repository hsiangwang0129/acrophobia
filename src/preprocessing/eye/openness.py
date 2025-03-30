import numpy as np
import pandas as pd
from datetime import datetime
import numba
import traceback

# 統一眨眼閾值常數
BLINK_THRESHOLD = 0.5
PERCLOS_THRESHOLD = 0.3

@numba.jit(nopython=True)
def _blink_rate_calc(values, timestamps):
    """
    計算眨眼頻率的核心計算部分（純 numba 實現）
    
    Parameters:
        values: numpy 數組格式的眼睛開合度值
        timestamps: numpy 數組格式的時間戳（秒）
    
    Returns:
        每秒眨眼次數
    """
    blink_count = 0
    
    # 檢查數據有效性
    if len(values) < 2 or len(timestamps) < 2:
        return 0
    
    # 計算總時間（秒）
    start_time = timestamps[0]
    end_time = timestamps[-1]
    total_time = end_time - start_time
    
    i = 0
    while i < len(values) - 1:
        isBlink = False
        # 找到開始眨眼處
        if values[i] == 1 and values[i+1] < 1:
            j = 1
            # 找出眨眼區間，且該區間內openness有小於BLINK_THRESHOLD，才認定為眨眼
            while i+j < len(values) and values[i+j] != 1:
                if values[i+j] <= BLINK_THRESHOLD:
                    isBlink = True
                j += 1
            
            if isBlink:
                blink_count += 1
            
            # 跳過已處理的眨眼區間
            i += j
        else:
            i += 1
    
    # 返回每秒眨眼次數
    return blink_count / total_time if total_time > 0 else 0

def blink_rate(df, eye_column):
    """
    計算眨眼頻率（每秒眨眼次數）
    
    Parameters:
        df: 包含眼睛開合度和時間戳的DataFrame
        eye_column: 眼睛開合度列名
    
    Returns:
        每秒眨眼次數
    """
    try:
        # 檢查數據有效性
        if df.empty or len(df) < 2:
            return 0
        
        # 提取開合度值和時間戳
        values = df[eye_column].values
        
        # 將時間戳轉換為秒級數值
        if 'Timestamp' in df.columns:
            timestamps = np.array([pd.to_datetime(ts).timestamp() for ts in df['Timestamp']])
        else:
            timestamps = np.array([pd.to_datetime(ts).timestamp() for ts in df.index])
        
        # 調用 numba 加速的核心計算函數
        return _blink_rate_calc(values, timestamps)
    except Exception as e:
        # 發生錯誤時返回0
        print(f"Error in blink_rate: {e}")
        print(traceback.format_exc())
        return 0

@numba.jit(nopython=True)
def _compute_blink_metrics(values, timestamps):
    """
    純計算部分：計算眨眼持續時間、間隔和PERCLOS
    
    Parameters:
        values: 眼睛開合度值數組
        timestamps: 對應的時間戳數組（毫秒）
    
    Returns:
        眨眼持續時間列表、眨眼開始時間列表、PERCLOS值
    """
    duration = []
    blink_start_times = []
    PERCLOS_time = 0
    total_time = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0
    
    i = 0
    while i < len(values) - 1:
        # 找出眨眼開始時間
        if values[i] == 1 and values[i+1] < 1:
            start_idx = i + 1
            start_time = timestamps[start_idx]
            j = 1
            isBlink = False
            
            # 找出眨眼結束時間
            while i+j < len(values) and values[i+j] != 1:
                if values[i+j] <= BLINK_THRESHOLD:  # 使用常數閾值
                    isBlink = True
                j += 1
            
            if i+j < len(values):
                end_idx = i + j
                end_time = timestamps[end_idx]
                time_diff = end_time - start_time
                
                if isBlink:
                    duration.append(time_diff)
                    blink_start_times.append(start_time)
                
                # 跳過已處理的眨眼區間
                i += j
            else:
                i += 1
        
        # PERCLOS 計算（眼睛閉合超過閾值的時間比例）
        elif i < len(values) - 1 and values[i] > PERCLOS_THRESHOLD and values[i+1] <= PERCLOS_THRESHOLD:
            perclos_start_idx = i + 1
            perclos_start = timestamps[perclos_start_idx]
            k = 1
            
            while i+k < len(values) and values[i+k] <= PERCLOS_THRESHOLD:
                k += 1
            
            if i+k < len(values):
                perclos_end_idx = i + k
                perclos_end = timestamps[perclos_end_idx]
                time_diff = perclos_end - perclos_start
                PERCLOS_time += time_diff
                
                # 跳過已處理的閉眼區間
                i += k
            else:
                i += 1
        else:
            i += 1
    
    # 計算PERCLOS值（百分比）
    PERCLOS = (PERCLOS_time / total_time) * 100 if total_time > 0 else 0
    
    return duration, blink_start_times, PERCLOS


@numba.jit(nopython=True)
def _compute_intervals(start_times):
    """
    計算眨眼間隔
    
    Parameters:
        start_times: 眨眼開始時間列表
        
    Returns:
        眨眼間隔數組
    """
    if len(start_times) <= 1:
        return np.array([0.0])
    
    intervals = np.zeros(len(start_times) - 1)
    for i in range(1, len(start_times)):
        intervals[i-1] = start_times[i] - start_times[i-1]
    
    return intervals


def blink_duration(df, eye_column):
    """
    計算眨眼持續時間和間隔的統計數據，以及PERCLOS值
    
    Parameters:
        df: 包含眼睛開合度和時間戳的DataFrame
        eye_column: 眼睛開合度列名
    
    Returns:
        (眨眼持續時間平均值, 眨眼持續時間標準差, 眨眼間隔平均值, 眨眼間隔標準差, PERCLOS值)
    """
    try:
        # 檢查數據有效性
        if df.empty or len(df) < 2:
            return (0, 0, 0, 0, 0)
        
        # 提取開合度值和時間戳
        values = df[eye_column].values
        
        # 將時間戳轉換為毫秒級數值
        if 'Timestamp' in df.columns:
            timestamps = np.array([pd.to_datetime(ts).timestamp() * 1000 for ts in df['Timestamp']])
        else:
            timestamps = np.array([pd.to_datetime(ts).timestamp() * 1000 for ts in df.index])
        
        # 檢查預處理後的數據有效性
        if len(values) < 2 or len(timestamps) < 2:
            return (0, 0, 0, 0, 0)
        
        # 計算眨眼指標
        duration, start_times, PERCLOS = _compute_blink_metrics(values, timestamps)
        
        # 如果沒有檢測到眨眼，返回只有PERCLOS的結果
        if len(duration) == 0:
            return (0, 0, 0, 0, PERCLOS)
        
        # 計算眨眼間隔
        duration_array = np.array(duration)
        intervals = _compute_intervals(np.array(start_times))
        
        # 返回統計數據
        return (
            float(np.mean(duration_array)),
            float(np.std(duration_array)) if len(duration_array) > 1 else 0,
            float(np.mean(intervals)),
            float(np.std(intervals)) if len(intervals) > 1 else 0,
            float(PERCLOS)
        )
    except Exception as e:
        # 發生錯誤時返回全0並記錄錯誤
        print(f"Error in blink_duration: {e}")
        print(traceback.format_exc())
        return (0, 0, 0, 0, 0)


if __name__ == "__main__":
    try:
        from Eye import Eye
        with Eye() as eye:
            df = eye.load_openness()
            
            # 計算眨眼頻率
            df['LeftBlinkRate'] = blink_rate(df, 'LeftEyeOpenness')
            df['RightBlinkRate'] = blink_rate(df, 'RightEyeOpenness')
            
            # 計算左眼眨眼持續時間等指標
            left_metrics = blink_duration(df, 'LeftEyeOpenness')
            df['LeftBlinkDurationMean'] = left_metrics[0]
            df['LeftBlinkDurationStd'] = left_metrics[1]
            df['LeftBlinkIntervalMean'] = left_metrics[2]
            df['LeftBlinkIntervalStd'] = left_metrics[3]
            df['LeftPERCLOS'] = left_metrics[4]
            
            # 計算右眼眨眼持續時間等指標
            right_metrics = blink_duration(df, 'RightEyeOpenness')
            df['RightBlinkDurationMean'] = right_metrics[0]
            df['RightBlinkDurationStd'] = right_metrics[1]
            df['RightBlinkIntervalMean'] = right_metrics[2]
            df['RightBlinkIntervalStd'] = right_metrics[3]
            df['RightPERCLOS'] = right_metrics[4]
            
            print("計算完成", df)
    except Exception as e:
        print(f"Error in main execution: {e}")
        print(traceback.format_exc())
