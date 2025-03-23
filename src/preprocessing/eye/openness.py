import numpy as np
import pandas as pd
from datetime import datetime
import numba

@numba.jit(nopython=True)
def blink_rate(openness: list):
    """
    計算眨眼頻率（每秒眨眼次數）
    
    Parameters:
        openness: 眼睛開合度數據列表，每個元素為 [開合度值, 時間戳]
    
    Returns:
        每秒眨眼次數
    """
    blink_count = 0
    openness = np.array(openness)
    
    # 計算總時間（秒）
    start_time = pd.to_datetime(openness[0][1]).timestamp()
    end_time = pd.to_datetime(openness[-1][1]).timestamp()
    total_time = end_time - start_time
    
    i = 0
    while i < len(openness) - 1:
        isBlink = False
        # 找到開始眨眼處
        if openness[i][0] == 1 and openness[i+1][0] < 1:
            j = 1
            # 找出眨眼區間，且該區間內openness有小於0.5，才認定為眨眼
            while i+j < len(openness) and openness[i+j][0] != 1:
                if openness[i+j][0] <= 0.5:
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


# 第一步：預處理函數，將時間戳轉換為數值型時間戳數組
def preprocess_timestamps(openness):
    """將原始數據中的時間戳轉換為數值型時間戳（毫秒）"""
    timestamps = []
    values = []
    
    for item in openness:
        values.append(item[0])
        # 將時間戳轉換為毫秒級數值
        timestamp = pd.to_datetime(item[1]).timestamp() * 1000
        timestamps.append(timestamp)
    
    return np.array(values), np.array(timestamps)


# 第二步：純計算部分，使用 nopython=True
@numba.jit(nopython=True)
def _compute_blink_metrics(values, timestamps):
    """
    純計算部分：計算眨眼持續時間、間隔和PERCLOS
    
    Parameters:
        values: 眼睛開合度值數組
        timestamps: 對應的時間戳數組（毫秒）
    
    Returns:
        眨眼持續時間列表、眨眼開始時間列表、PERCLOS時間
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
                if values[i+j] <= 0.5:
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
        
        # PERCLOS 計算（眼睛閉合超過70%的時間比例）
        elif i < len(values) - 1 and values[i] >= 0.3 and values[i+1] <= 0.3:
            perclos_start_idx = i + 1
            perclos_start = timestamps[perclos_start_idx]
            k = 1
            
            while i+k < len(values) and values[i+k] <= 0.3:
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


# 第三步：計算眨眼間隔的純計算部分
@numba.jit(nopython=True)
def _compute_intervals(start_times):
    """計算眨眼間隔"""
    if len(start_times) <= 1:
        return np.array([0.0])
    
    intervals = np.zeros(len(start_times) - 1)
    for i in range(1, len(start_times)):
        intervals[i-1] = start_times[i] - start_times[i-1]
    
    return intervals


# 第四步：主函數，組合上述步驟
def blink_duration(openness: list):
    """
    計算眨眼持續時間和間隔的統計數據，以及PERCLOS值
    
    Parameters:
        openness: 眼睛開合度數據列表，每個元素為 [開合度值, 時間戳]
    
    Returns:
        (眨眼持續時間平均值, 眨眼持續時間標準差, 眨眼間隔平均值, 眨眼間隔標準差, PERCLOS值)
    """
    try:
        # 預處理數據
        values, timestamps = preprocess_timestamps(openness)
        
        # 計算眨眼指標
        duration, start_times, PERCLOS = _compute_blink_metrics(values, timestamps)
        
        # 如果沒有檢測到眨眼，返回全0
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
        # 發生錯誤時返回全0
        print(f"Error in blink_duration: {e}")
        return (0, 0, 0, 0, 0)
