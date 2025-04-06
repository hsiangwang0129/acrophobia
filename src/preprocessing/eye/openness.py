import numpy as np  # 用於數值計算
import pandas as pd  # 用於數據處理和分析
from datetime import datetime  # 處理日期和時間
import numba  # 用於加速數值運算
import traceback  # 用於捕捉和打印錯誤堆疊
from typing import List, Tuple, Dict  # 用於類型標註

# 統一眨眼閾值常數
BLINK_THRESHOLD: float = 0.5  # 定義眨眼的閾值，低於此值表示眼睛閉合
PERCLOS_THRESHOLD: float = 0.3  # 定義 PERCLOS 的閾值，低於此值表示眼睛閉合時間計入 PERCLOS

@numba.jit(nopython=True)
def _blink_rate_calc(values: np.ndarray, timestamps: np.ndarray) -> float:
    """
    計算眨眼頻率的核心計算部分（純 numba 實現）

    參數:
        values (np.ndarray): 眼睛開合度數據的數組。
        timestamps (np.ndarray): 時間戳數據的數組。

    返回:
        float: 每秒眨眼次數。
    """
    blink_count: int = 0  # 初始化眨眼次數計數器
    
    # 如果數據不足，直接返回 0
    if len(values) < 2 or len(timestamps) < 2:
        return 0.0
    
    start_time: float = timestamps[0]  # 起始時間
    end_time: float = timestamps[-1]  # 結束時間
    total_time: float = end_time - start_time  # 總時長
    
    i: int = 0
    while i < len(values) - 1:
        isBlink: bool = False  # 假設當前不是眨眼
        # 檢測眨眼的開始
        if values[i] == 1 and values[i+1] < 1:
            j: int = 1
            # 檢測眨眼持續時間
            while i+j < len(values) and values[i+j] != 1:
                if values[i+j] <= BLINK_THRESHOLD:  # 判定是否符合眨眼條件
                    isBlink = True
                j += 1
            
            # 如果是眨眼，計數器加 1
            if isBlink:
                blink_count += 1
            
            # 跳過已處理的數據
            i += j
        else:
            i += 1  # 繼續下一個數據
    
    # 返回每秒眨眼次數
    return blink_count / total_time if total_time > 0 else 0.0


def blink_rate(df: pd.DataFrame, eye_column: str) -> float:
    """
    計算眨眼頻率（每秒眨眼次數）

    參數:
        df (pd.DataFrame): 包含眼睛開合度數據的數據框。
        eye_column (str): 眼睛開合度的欄位名稱。

    返回:
        float: 每秒眨眼次數。
    """
    try:
        # 如果數據不足，直接返回 0
        if df.empty or len(df) < 2:
            return 0.0
        
        values: np.ndarray = df[eye_column].values  # 提取眼睛開合度數據
        
        # 提取時間戳數據
        if 'Timestamp' in df.columns:
            timestamps: np.ndarray = np.array([pd.to_datetime(ts).timestamp() for ts in df['Timestamp']])
        else:
            timestamps: np.ndarray = np.array([pd.to_datetime(ts).timestamp() for ts in df.index])
        
        # 調用核心計算函數
        return _blink_rate_calc(values, timestamps)
    except Exception as e:
        print(f"Error in blink_rate: {e}")
        print(traceback.format_exc())
        return 0.0  # 出現錯誤時返回 0


@numba.jit(nopython=True)
def _compute_blink_metrics(values: np.ndarray, timestamps: np.ndarray) -> Tuple[List[float], List[float], float]:
    """
    計算眨眼持續時間、間隔和 PERCLOS。

    參數:
        values (np.ndarray): 眼睛開合度數據的數組。
        timestamps (np.ndarray): 時間戳數據的數組。

    返回:
        Tuple[List[float], List[float], float]: 包括眨眼持續時間列表、眨眼開始時間列表和 PERCLOS 值。
    """
    duration: List[float] = []  # 存儲眨眼持續時間
    blink_start_times: List[float] = []  # 存儲眨眼開始時間
    PERCLOS_time: float = 0.0  # 初始化 PERCLOS 時間累積
    total_time: float = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0.0  # 總時長
    
    i: int = 0
    while i < len(values) - 1:
        # 檢測眨眼的開始
        if values[i] == 1 and values[i+1] < 1:
            start_idx: int = i + 1
            start_time: float = timestamps[start_idx]
            j: int = 1
            isBlink: bool = False
            
            # 檢測眨眼持續時間
            while i+j < len(values) and values[i+j] != 1:
                if values[i+j] <= BLINK_THRESHOLD:  # 判定是否符合眨眼條件
                    isBlink = True
                j += 1
            
            # 如果眨眼結束
            if i+j < len(values):
                end_idx: int = i + j
                end_time: float = timestamps[end_idx]
                time_diff: float = end_time - start_time
                
                if isBlink:
                    duration.append(time_diff)  # 記錄眨眼持續時間
                    blink_start_times.append(start_time)  # 記錄眨眼開始時間
                
                i += j
            else:
                i += 1
        
        # 檢測 PERCLOS 的開始
        elif i < len(values) - 1 and values[i] > PERCLOS_THRESHOLD and values[i+1] <= PERCLOS_THRESHOLD:
            perclos_start_idx: int = i + 1
            perclos_start: float = timestamps[perclos_start_idx]
            k: int = 1
            
            # 檢測 PERCLOS 持續時間
            while i+k < len(values) and values[i+k] <= PERCLOS_THRESHOLD:
                k += 1
            
            # 如果 PERCLOS 結束
            if i+k < len(values):
                perclos_end_idx: int = i + k
                perclos_end: float = timestamps[perclos_end_idx]
                time_diff: float = perclos_end - perclos_start
                PERCLOS_time += time_diff
                
                i += k
            else:
                i += 1
        else:
            i += 1
    
    # 計算 PERCLOS 值
    PERCLOS: float = (PERCLOS_time / total_time) * 100 if total_time > 0 else 0.0
    
    return duration, blink_start_times, PERCLOS


@numba.jit(nopython=True)
def _compute_intervals(start_times: np.ndarray) -> np.ndarray:
    """
    計算眨眼間隔。

    參數:
        start_times (np.ndarray): 眨眼開始時間的數組。

    返回:
        np.ndarray: 每次眨眼之間的間隔數組。
    """
    # 如果只有一個眨眼記錄，返回空間隔
    if len(start_times) <= 1:
        return np.array([0.0])
    
    intervals = np.zeros(len(start_times) - 1)  # 初始化間隔數組
    for i in range(1, len(start_times)):
        intervals[i-1] = start_times[i] - start_times[i-1]  # 計算相鄰眨眼的間隔
    
    return intervals


def blink_duration(df: pd.DataFrame, eye_column: str) -> Tuple[float, float, float, float, float]:
    """
    計算眨眼持續時間、間隔的統計數據，以及 PERCLOS 值。

    參數:
        df (pd.DataFrame): 包含眼睛開合度數據的數據框。
        eye_column (str): 眼睛開合度的欄位名稱。

    返回:
        Tuple[float, float, float, float, float]: 
            - 平均眨眼持續時間
            - 持續時間的標準差
            - 平均眨眼間隔
            - 間隔的標準差
            - PERCLOS 值
    """
    try:
        # 如果數據不足，返回全 0
        if df.empty or len(df) < 2:
            return (0.0, 0.0, 0.0, 0.0, 0.0)
        
        values: np.ndarray = df[eye_column].values  # 提取眼睛開合度數據
        if 'Timestamp' in df.columns:
            timestamps: np.ndarray = np.array([pd.to_datetime(ts).timestamp() * 1000 for ts in df['Timestamp']])
        else:
            timestamps: np.ndarray = np.array([pd.to_datetime(ts).timestamp() * 1000 for ts in df.index])
        
        # 如果數據不足，返回全 0
        if len(values) < 2 or len(timestamps) < 2:
            return (0.0, 0.0, 0.0, 0.0, 0.0)
        
        # 計算眨眼持續時間、開始時間和 PERCLOS
        duration, start_times, PERCLOS = _compute_blink_metrics(values, timestamps)
        
        # 如果沒有眨眼記錄，返回 PERCLOS
        if len(duration) == 0:
            return (0.0, 0.0, 0.0, 0.0, PERCLOS)
        
        duration_array: np.ndarray = np.array(duration)
        intervals: np.ndarray = _compute_intervals(np.array(start_times))
        
        return (
            float(np.mean(duration_array)),  # 平均眨眼持續時間
            float(np.std(duration_array)) if len(duration_array) > 1 else 0.0,  # 持續時間的標準差
            float(np.mean(intervals)),  # 平均眨眼間隔
            float(np.std(intervals)) if len(intervals) > 1 else 0.0,  # 間隔的標準差
            float(PERCLOS)  # PERCLOS 值
        )
    except Exception as e:
        print(f"Error in blink_duration: {e}")
        print(traceback.format_exc())
        return (0.0, 0.0, 0.0, 0.0, 0.0)


def split_by_time_window(df: pd.DataFrame, window_size: int) -> Dict[int, pd.DataFrame]:
    """
    根據時間窗口對數據進行分段。

    參數:
        df (pd.DataFrame): 包含眼睛開合度數據的數據框。
        window_size (int): 時間窗口大小（以秒為單位）。

    返回:
        Dict[int, pd.DataFrame]: 每個窗口的數據字典，鍵為窗口編號，值為對應的數據框。
    """
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])  # 確保時間戳為 datetime 格式
    start_time: datetime = df['Timestamp'].iloc[0]  # 起始時間
    df['elapsed_time'] = (df['Timestamp'] - start_time).dt.total_seconds()  # 計算經過時間

    df['time_window'] = (df['elapsed_time'] // window_size).astype(int)  # 計算窗口編號

    window_dict: Dict[int, pd.DataFrame] = {}
    for window, window_df in df.groupby('time_window'):  # 按窗口編號分組
        window_dict[window] = window_df.reset_index(drop=True)  # 存儲分段數據
    
    return window_dict


def split_phases(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    將數據分為三個階段：Baseline、High-rise Early 和 High-rise Late。

    參數:
        df (pd.DataFrame): 包含眼睛開合度數據的數據框。

    返回:
        Dict[str, pd.DataFrame]: 每個階段的數據字典，鍵為階段名稱，值為對應的數據框。
    """
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])  # 確保時間戳為 datetime 格式
    start_time: datetime = df['Timestamp'].iloc[0]  # 起始時間
    df['elapsed_time'] = (df['Timestamp'] - start_time).dt.total_seconds()  # 計算經過時間

    # 根據經過時間分配階段標籤
    def assign_phase(row: pd.Series) -> str:
        if row['elapsed_time'] <= 60:
            return 'Baseline'
        elif 60 < row['elapsed_time'] <= 90:
            return 'High-rise Early'
        else:
            return 'High-rise Late'

    df['phase'] = df.apply(assign_phase, axis=1)  # 添加階段標籤

    phase_dict: Dict[str, pd.DataFrame] = {}
    for phase_name, phase_df in df.groupby('phase'):  # 按階段分組
        phase_dict[phase_name] = phase_df.reset_index(drop=True)  # 存儲分段數據
    
    return phase_dict


if __name__ == "__main__":
    try:
        from Eye import Eye  # 假設 Eye 類能夠載入眼動追蹤數據
        with Eye() as eye:
            df: pd.DataFrame = eye.load_openness()  # 載入數據
            
            # 設置時間窗口大小（秒）
            window_size: int = 30
            
            # 按時間窗口分段數據
            windows: Dict[int, pd.DataFrame] = split_by_time_window(df, window_size)
            
            # 分階段處理
            phases: Dict[str, pd.DataFrame] = split_phases(df)
            
            # 處理每個階段
            for phase_name, phase_df in phases.items():
                print(f"Processing phase: {phase_name}")
                
                # 計算眨眼頻率
                phase_df['LeftBlinkRate'] = blink_rate(phase_df, 'LeftEyeOpenness')
                phase_df['RightBlinkRate'] = blink_rate(phase_df, 'RightEyeOpenness')
                
                # 打印結果（可選）
                print(f"Phase {phase_name} calculation completed.")
                # print(phase_df.head())  # 可選，檢查數據
                
            print("All phases processed successfully.")
    except Exception as e:
        print(f"Error in main execution: {e}")
        print(traceback.format_exc())

