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
    """
    blink_count = 0
    
    if len(values) < 2 or len(timestamps) < 2:
        return 0
    
    start_time = timestamps[0]
    end_time = timestamps[-1]
    total_time = end_time - start_time
    
    i = 0
    while i < len(values) - 1:
        isBlink = False
        if values[i] == 1 and values[i+1] < 1:
            j = 1
            while i+j < len(values) and values[i+j] != 1:
                if values[i+j] <= BLINK_THRESHOLD:
                    isBlink = True
                j += 1
            
            if isBlink:
                blink_count += 1
            
            i += j
        else:
            i += 1
    
    return blink_count / total_time if total_time > 0 else 0

def blink_rate(df, eye_column):
    """
    計算眨眼頻率（每秒眨眼次數）
    """
    try:
        if df.empty or len(df) < 2:
            return 0
        
        values = df[eye_column].values
        
        if 'Timestamp' in df.columns:
            timestamps = np.array([pd.to_datetime(ts).timestamp() for ts in df['Timestamp']])
        else:
            timestamps = np.array([pd.to_datetime(ts).timestamp() for ts in df.index])
        
        return _blink_rate_calc(values, timestamps)
    except Exception as e:
        print(f"Error in blink_rate: {e}")
        print(traceback.format_exc())
        return 0

@numba.jit(nopython=True)
def _compute_blink_metrics(values, timestamps):
    """
    計算眨眼持續時間、間隔和PERCLOS
    """
    duration = []
    blink_start_times = []
    PERCLOS_time = 0
    total_time = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0
    
    i = 0
    while i < len(values) - 1:
        if values[i] == 1 and values[i+1] < 1:
            start_idx = i + 1
            start_time = timestamps[start_idx]
            j = 1
            isBlink = False
            
            while i+j < len(values) and values[i+j] != 1:
                if values[i+j] <= BLINK_THRESHOLD:
                    isBlink = True
                j += 1
            
            if i+j < len(values):
                end_idx = i + j
                end_time = timestamps[end_idx]
                time_diff = end_time - start_time
                
                if isBlink:
                    duration.append(time_diff)
                    blink_start_times.append(start_time)
                
                i += j
            else:
                i += 1
        
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
                
                i += k
            else:
                i += 1
        else:
            i += 1
    
    PERCLOS = (PERCLOS_time / total_time) * 100 if total_time > 0 else 0
    
    return duration, blink_start_times, PERCLOS

@numba.jit(nopython=True)
def _compute_intervals(start_times):
    """
    計算眨眼間隔
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
    """
    try:
        if df.empty or len(df) < 2:
            return (0, 0, 0, 0, 0)
        
        values = df[eye_column].values
        if 'Timestamp' in df.columns:
            timestamps = np.array([pd.to_datetime(ts).timestamp() * 1000 for ts in df['Timestamp']])
        else:
            timestamps = np.array([pd.to_datetime(ts).timestamp() * 1000 for ts in df.index])
        
        if len(values) < 2 or len(timestamps) < 2:
            return (0, 0, 0, 0, 0)
        
        duration, start_times, PERCLOS = _compute_blink_metrics(values, timestamps)
        
        if len(duration) == 0:
            return (0, 0, 0, 0, PERCLOS)
        
        duration_array = np.array(duration)
        intervals = _compute_intervals(np.array(start_times))
        
        return (
            float(np.mean(duration_array)),
            float(np.std(duration_array)) if len(duration_array) > 1 else 0,
            float(np.mean(intervals)),
            float(np.std(intervals)) if len(intervals) > 1 else 0,
            float(PERCLOS)
        )
    except Exception as e:
        print(f"Error in blink_duration: {e}")
        print(traceback.format_exc())
        return (0, 0, 0, 0, 0)

def split_by_time_window(df, window_size):
    """
    根據時間窗口對數據進行分段
    """
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    start_time = df['Timestamp'].iloc[0]
    df['elapsed_time'] = (df['Timestamp'] - start_time).dt.total_seconds()

    df['time_window'] = (df['elapsed_time'] // window_size).astype(int)

    window_dict = {}
    for window, window_df in df.groupby('time_window'):
        window_dict[window] = window_df.reset_index(drop=True)
    
    return window_dict

def split_phases(df):
    """
    將數據分為三個階段：Baseline、High-rise Early 和 High-rise Late
    """
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    start_time = df['Timestamp'].iloc[0]
    df['elapsed_time'] = (df['Timestamp'] - start_time).dt.total_seconds()

    def assign_phase(row):
        if row['elapsed_time'] <= 60:
            return 'Baseline'
        elif 60 < row['elapsed_time'] <= 90:
            return 'High-rise Early'
        else:
            return 'High-rise Late'

    df['phase'] = df.apply(assign_phase, axis=1)

    phase_dict = {}
    for phase_name, phase_df in df.groupby('phase'):
        phase_dict[phase_name] = phase_df.reset_index(drop=True)

    return phase_dict

if __name__ == "__main__":
    try:
        from Eye import Eye
        with Eye() as eye:
            df = eye.load_openness()
            
            # 設置時間窗口大小（秒）
            window_size = 30
            
            # 按時間窗口分段數據
            windows = split_by_time_window(df, window_size)
            
            # 分階段處理
            phases = split_phases(df)
            
            # 處理每個階段
            for phase_name, phase_df in phases.items():
                print(f"Processing phase: {phase_name}")
                
                # 計算眨眼頻率
                phase_df['LeftBlinkRate'] = blink_rate(phase_df, 'LeftEyeOpenness')
                phase_df['RightBlinkRate'] = blink_rate(phase_df, 'RightEyeOpenness')
                
                print(f"Phase {phase_name} calculation completed.")
                # print(phase_df.head())
            
            print("All phases processed successfully.")
    except Exception as e:
        print(f"Error in main execution: {e}")
        print(traceback.format_exc())
