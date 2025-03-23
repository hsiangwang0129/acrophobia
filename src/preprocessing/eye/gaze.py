from .Eye import Eye
import numpy as np
import math
import pandas as pd
import torch
import json
def fixation_count(data, time):  
    direction_X = []
    direction_Y = []
    direction_Z = []
    for direction in data:
        if direction['x'] == 0.0 :
            continue
        direction_X.append(direction['x'])
        direction_Y.append(direction['y'])
        direction_Z.append(direction['z'])
        
    direction_X = np.array(direction_X)
    direction_Y = np.array(direction_Y)
    direction_Z = np.array(direction_Z)
    
    fixation_count = 0
    for i in range(5, len(direction_X)):
        isFixation = True
        for j in range(0, 5):
            vector_dot = direction_X[i]*direction_X[i-j] + direction_Y[i]*direction_Y[i-j] + direction_Z[i]*direction_Z[i-j]
            try:
                angle = float(math.acos(vector_dot))*180/math.pi
            except:
                angle = float(math.acos(1))*180/math.pi
            if(angle >= 3):
                isFixation = False
        if(isFixation):
            fixation_count += 1
    print("fixation count", fixation_count/time)
    return fixation_count/time

def calculate_angle(v1, v2):
    dot_product = torch.dot(v1, v2)
    length_v1 = torch.norm(v1)
    length_v2 = torch.norm(v2)
    cosine_similarity = dot_product / (length_v1 * length_v2)
    radians = torch.acos(cosine_similarity)
    degrees = radians * 180 / math.pi
    return degrees

def accumulate_on_series(series):
    accumulate = 0
    for i in range(len(series)-1):
        accumulate += abs(series[i] - series[i+1])
    return accumulate

def accumulate_gaze_angle(data, time):
    baseVector = torch.tensor([0., 0., 1.]).float() 
    angles = []
    for direction in data:
        if direction['x'] == 0.0 :
            continue
        gazeDirectionVector = torch.tensor([direction['x'], direction['y'], direction['z']]).float()
        angle = calculate_angle(gazeDirectionVector, baseVector)
        angles.append(angle)
    output = accumulate_on_series(angles)
    print("accumulate_on_series",output)
    output = output.float().item()
    print("accumulate_gaze_angle output", output)
    return output/time


def feature_extract(df):
    # game = eachArea(path2) #prepocessing game.json
    # phase = cut_by_timer(data, game[['start_time', 'end_time']]) #return the df that divide by third round
    result_data = []
    phases = split_phases(df)
    for _, p in phases.items():
        time = (pd.to_datetime(p['Timestamp'][len(p)-1]) - pd.to_datetime(p['Timestamp'][0])).total_seconds()
        print(time)
        temp_result = []
        # temp_result.append(blink_duration(p[['LeftEyeOpenness', 'timeStamp']]))
        # temp_result.append((pupilDiameter(p['LeftEyePupilDiameter'])))
        # temp_result = list(itertools.chain.from_iterable(temp_result))
        # temp_result.append(blink_rate(p[['LeftEyeOpenness', 'timeStamp']]))
        temp_result.append(fixation_count(p['CombineEyeGazeVector'], time))
        temp_result.append(accumulate_gaze_angle(p['CombineEyeGazeVector'], time))
        # temp_result.append(calculatePathLength(p['LeftEyePosition'], time))
        result_data.append(temp_result)

    return result_data

def split_phases(df):
    
    # 2. timestamp convert to datetime format
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])

    # 3. calculate elapsed_time
    start_time = df['Timestamp'].iloc[0]
    df['elapsed_time'] = (df['Timestamp'] - start_time).dt.total_seconds()

    # 4. 定義 phase
    def assign_phase(row):
        if row['elapsed_time'] <= 60:
            return 'Baseline'
        elif 60 < row['elapsed_time'] <= 90:
            return 'High-rise Early'
        else:
            return 'High-rise Late'

    df['phase'] = df.apply(assign_phase, axis=1)

    # 5. 按 phase 分段
    phase_dict = {}
    for phase_name, phase_df in df.groupby('phase'):
        phase_dict[phase_name] = phase_df.reset_index(drop=True)

    return phase_dict

newEye = Eye()
df = newEye.load_gaze()
print(df.head(50))
# phases = split_phases(df)
# print("len of phases",len(phases))
# print(phases['Baseline'])
result = feature_extract(df)
print(result)