import numpy as np
import pandas as pd
import json
import csv
import sys
import time 
from datetime import datetime
from datetime import timedelta
import glob
import math
import matplotlib.pyplot as plt
from sklearn.utils import resample
from scipy.ndimage.filters import gaussian_filter
import seaborn as sns
import itertools
import torch
from pathlib import Path

def blink_duration(openness):
    openness = np.array(openness)
    duration = []
    start_time = []
    PERCLOS_time = 0
    for i in range(len(openness)):
        try:
            #找出眨眼開始時間
            if(openness[i][0] == 1 and openness[i+1][0] < 1):
                start = pd.to_datetime(openness[i+1][1])
                j = 1
                isBlink = False
                #找出眨眼結束時間
                while(openness[i+j][0] != 1):
                    if(openness[i+j][0] < 0.5):
                        isBlink = True
                    j += 1
                end = pd.to_datetime(openness[i+j][1])
                time = (end - start).total_seconds()*1000
                if(isBlink):
                    duration.append(time)
                    start_time.append(start)
                i += j
            # PERCLOS
            if(openness[i][0] >= 0.3 and openness[i+1][0] <= 0.3):
                perclos_start = pd.to_datetime(openness[i+1][1])
                k = 1
                isclose70 = True
                while(openness[i+k][0] <= 0.3):
                    k += 1
                isclose70 = False
                perclos_end = pd.to_datetime(openness[i+k][1])
                time = (perclos_end - perclos_start).total_seconds()*1000
                PERCLOS_time += time
        except:
            continue
            
    if(len(duration) == 0):
        return (0,0,0,0,0)
    else:
        duration = np.array(duration)
        start_time = np.array(start_time)
        time_difference = start_time[1:] - start_time[:-1]
        interval = []
        if(len(time_difference)> 1):
            for time in time_difference:
                interval.append(time.total_seconds() * 1000)
        else:
            interval = duration
        interval = np.array(interval)
        # perclos
        try:
            total_time = (pd.to_datetime(openness[len(openness)-1][1]) - pd.to_datetime(openness[0][1])).total_seconds()*1000
            PERCLOS = (PERCLOS_time/total_time)*100
        except:
            print("PERCLOS error")
            PERCLOS = 0
        # print(total_time, PERCLOS)
        return duration.mean(), duration.std(), interval.mean(), interval.std()

def pupilDiameter(diameter):
    while diameter[0] == -1.0:
        diameter.pop(0)
    diameter = np.array(diameter)
    delta = abs((diameter[1:] - diameter[:-1]))
    pcps = []
    for i in range(len(diameter)):
        pcps.append(abs(diameter[i]-diameter.mean()))
    pcps = np.array(pcps)
    # print(pcps.mean())
    return diameter.mean(), diameter.std(), delta.mean(), delta.std(), pcps.mean()

def blink_rate(openness):
    blink_count = 0
    openness = np.array(openness)
    total_time = (pd.to_datetime(openness[len(openness)-1][1]) - pd.to_datetime(openness[0][1])).total_seconds()
    for i in range(0, len(openness)):
        isBlink = False
        try:
            #找到開始眨眼處
            if (openness[i][0] == 1 and openness[i+1][0] < 1):
                j = 1
                #找出眨眼區間, 且該區間內openness有小於某數, 才認定為眨眼
                while(openness[i+j][0] != 1):
                    if(openness[i+j][0] <= 0.5):
                        isBlink = True
                    j+=1
                if(isBlink):
                    blink_count += 1
                i += j
        except:
            continue
    return blink_count/total_time    #待資料筆數等長時加入blink_count

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
                angle = float(math.acos(vector_dot))*180/3.14
            except:
                angle = float(math.acos(1))*180/3.14
            if(angle >= 3):
                isFixation = False
        if(isFixation):
            fixation_count += 1
            
    return fixation_count/time

def accumulate_on_series(series):
    accumulate = 0
    for i in range(len(series)-1):
        accumulate += abs(series[i] - series[i+1])
    return accumulate

def calculate_angle(v1, v2):
    dot_product = torch.dot(v1, v2)
    length_v1 = torch.norm(v1)
    length_v2 = torch.norm(v2)
    cosine_similarity = dot_product / (length_v1 * length_v2)
    radians = torch.acos(cosine_similarity)
    degrees = radians * 180 / math.pi
    return degrees

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
    output = output.float().item()
    return output/time

def calculateEuclideanDistance(a, b):
        return np.linalg.norm(a-b)

def calculatePathLength(data, time):
    x_series = []
    y_series = []
    z_series = []
    pathLength = 0
    try:
        for path in data:
            if path['x'] == 0.0 :
                continue
            x_series.append(path['x'])
            y_series.append(path['y'])
            z_series.append(path['z'])
        for i in range(len(x_series)-1):
            nowPos = np.array((x_series[i], y_series[i], z_series[i]))
            nextPos = np.array((x_series[i+1], y_series[i+1], z_series[i+1]))
            pathLength += calculateEuclideanDistance(nowPos, nextPos)
    except:
        print("error while calculating path length")
    return pathLength/time

def feature_extract(data):
    result_data = []
    time = (pd.to_datetime(data['timeStamp'][len(data)-1]) - pd.to_datetime(data['timeStamp'][0])).total_seconds()
    print(time)
    temp_result = []
    temp_result.append(blink_duration(data[['LeftEyeOpenness', 'timeStamp']]))
    temp_result.append((pupilDiameter(data['LeftEyePupilDiameter'])))
    temp_result = list(itertools.chain.from_iterable(temp_result))
    temp_result.append(blink_rate(data[['LeftEyeOpenness', 'timeStamp']]))
    temp_result.append(fixation_count(data['LeftGazeDirection'], time))
    temp_result.append(accumulate_gaze_angle(data['LeftGazeDirection'], time))
    temp_result.append(calculatePathLength(data['LeftEyePosition'], time))
    # temp_result.append(0)
    # result_data.append(temp_result)
    return temp_result

def cut_by_timer(data, timer_data):
    phase = [[] for i in range(4)]
    # print(phase)
    for tv in range(len(data)):
        time = pd.to_datetime(data['Timestamp'][tv])
        # print(time, timer_data[0])
        # print(data.loc[tv].values.flatten().tolist())
        if time < timer_data[0]: #無干擾
            phase[0].append(data.loc[tv].values.flatten().tolist())
        elif time < timer_data[1] and time >= timer_data[0]: #有干擾
            phase[1].append(data.loc[tv].values.flatten().tolist())
        # elif time < timer_data[2] and time >= timer_data[1]: #無干擾
        #     phase[2].append(data.loc[tv].values.flatten().tolist())
        elif time < timer_data[3] and time >= timer_data[2]: #有干擾
            phase[2].append(data.loc[tv].values.flatten().tolist())
        # elif time < timer_data[4] and time >= timer_data[3]: #無干擾
        #     phase[4].append(data.loc[tv].values.flatten().tolist())
        elif time < timer_data[5] and time >= timer_data[4]: #有干擾
            phase[3].append(data.loc[tv].values.flatten().tolist())
    for i in range(len(phase)):
        phase[i] = pd.DataFrame(phase[i],columns=['Timestamp', 'pupil_diameter_mm', 'eye_openness', 'gaze', 'gaze_point'])
    print(len(phase))
    print(len(phase[0]),len(phase[1]),len(phase[2]),len(phase[3]))
    return phase

def timer_process(path):
    timer_data = []
    strData = open(path,"r",encoding='utf-8').read()
    time = []
    time.extend(json.loads(strData)['Data'])
    for t in time:
        am_pm_list = t.split(' ')
        if am_pm_list[1] == '下午':
            new_t = datetime.strptime(t , '%Y/%m/%d 下午 %H:%M:%S')
            new_t += timedelta(hours=12)
            timer_data.append(new_t)
        else:
            new_t = datetime.strptime(t , '%Y/%m/%d 上午 %H:%M:%S')
            timer_data.append(new_t)
    # print(timer_data)
    return timer_data

def read_data(path):
    strData = open(path, 'r', encoding='utf-8').read()
    listData = strData.split('\n')
    listData = listData[:len(listData)-1]
    eye_data = []
    for data in listData:
        temp = []
        temp.append(json.loads(data)['timeStamp'].replace('+08:00', ''))
        temp.append(json.loads(data)['LeftEyePupilDiameter'])
        temp.append(json.loads(data)['LeftEyeOpenness'])
        temp.append(json.loads(data)['LeftGazeDirection'])
        temp.append(json.loads(data)['LeftEyePosition'])
        if temp[1] == -1.0 :
            continue
        if temp[3]['x'] == 0.0 :
            continue
        if temp[4]['x'] == 0.0 :
            continue
        eye_data.append(temp)
    eye_data.pop(0)
    df = []
    df = pd.DataFrame(eye_data,columns=['timeStamp', 'LeftEyePupilDiameter', 'LeftEyeOpenness', 'LeftGazeDirection', 'LeftEyePosition'])
    # print(df)
    return df

def data_prepro(path):
    data = read_data(path)
    # timer_data = timer_process(path2)
    # phase_data = cut_by_timer(data, timer_data)
    total_features = feature_extract(data)
    # blink_duration.mean(), blink_duration.std(), blink_interval.mean(), blink_interval.std(), PERCLOS, diameter.mean(), diameter.std(), delta.mean(), delta.std(), pcps.mean(),
    # blink_count/total_time, fixation_count/len(data), gaze_angle_accumulate/time, path/time
    # for i in total_features:
    #     print(i)
    return total_features

def main():
    # path = 'C:/Users/Jackson/OneDrive/LAB/資料/壓力訓練/測試資料/RestingState/202401150356_test_FPre/test_FPre_Eye.json'
    # path = 'C:/Users/Jackson/OneDrive/LAB/資料/壓力訓練/測試資料/RestingState/202403150520_N06_DPost/N06_DPost_Eye.json'
    path = 'C:/Users/Jackson/OneDrive/LAB/資料/壓力訓練/正式收案/RestingState'
    data_path = Path(path)
    data_path = [x for x in data_path.iterdir() if x.is_dir()]
    all_data = []
    for fd in data_path:
        # print(fd)
        eye = Path(list(fd.rglob('*Eye.json'))[0], lines=True)
        print(eye)
        temp = data_prepro(eye)
        # print(temp)
        all_data.append(temp)
    # temp = data_prepro(path)
    # print(all_data)
    final_data = pd.DataFrame(all_data, columns = ['blink_duration_m', 'blink_duration_std',
                                                    'blink_interval_m', 'blink_interval_std',
                                                    'diam_mean', 'diam_std', 'diam_delta_mean', 'diam_delta_std', 'pcps_mean',
                                                    'blink_rate', 'fixation_count/time', 'gaze_angle/time', 'path/time'])
    final_data.to_csv('C:/Users/Jackson/Downloads/result.csv')

def grouping_prepro(start , end):
    total_case = []
    total_label = []
    for i in range(start,end+1):
        if i==1 or i==3 or i==5 or i==6 or i==10 or i==11 or i==16 or i==17:
            pass
        # elif i==8:
        else:
            path1 = 'C:/Users/Jackson/OneDrive/LAB/VR_Driving/output/war_jeep_read/N%d/_N%d_LeftEyeTrackData.json'%(i,i)
            path2 = 'C:/Users/Jackson/OneDrive/LAB/VR_Driving/output/war_jeep_read/N%d/_N%d_TimerData.json'%(i,i)
            print("i = ", i)
            temp = data_prepro(path1)
            total_case.extend(temp)    #2維  每個一維是一段features
            condition = [0,1,1,1]
            total_label.extend(condition)
    # print(total_case)
    # # print(condition)
    # print(total_label)
    # print(len(total_case))
    # print(len(total_label))
  
    return  total_case , total_label

if __name__ == '__main__':
    main()







