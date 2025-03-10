from biosppy.signals import ecg
from pyentrp import entropy as ent
import pywt
from scipy import interpolate
from scipy.signal import welch
import matplotlib.pyplot as plt
import time 
from datetime import datetime
from datetime import timedelta
# from ecg_prepros import ECG_Analysis
from pathlib import Path
from hrvanalysis import get_sampen
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import json
import csv
import sys

def time_domain(rri):
    if not len(rri) == 0:
        diff = np.diff(rri,1)             #前後差值  A(n)-A(n-1)   ex (rri,2) >> A(n)-A(n-2) 
        rmssd = np.sqrt(sum(diff ** 2)/(len(rri)-1))                  #相鄰正常心跳間期差值平方和的均方根
        sdnn = np.std(rri, ddof=1)  # make it calculates N-1          #standard deviation of the NN interval
        nn50 = (sum(abs(np.diff(rri)) > 50))                          #心電圖中所有每對相鄰正常心跳時間間隔，差距超過50毫秒的數目
        pnn50 =  nn50 / len(rri) * 100                                #相鄰正常心跳間期差值超過 50 毫秒的百分比
        nn20 = (sum(abs(np.diff(rri)) > 20))                          #心電圖中所有每對相鄰正常心跳時間間隔，差距超過20毫秒的數目
        pnn20 = nn20/ len(rri) * 100                                  #相鄰正常心跳間期差值超過 20 毫秒的百分比
        mrri = np.mean(rri)                                           #mean of rr interval
        mhr = np.mean(60 / (rri / 1000.0))                            #mean of heart rates

        sdsd = np.std(diff, ddof=1)                                   # standard deviation of successive differences (SDSD)
        sdhr = np.std(60 / (rri / 1000.0), ddof=1)                    # standard deviation of heart rate (SDHR)
        # print("nn50 nn20 pnn20", nn50, nn20, pnn20)
        # print("sdsd sdhr", sdsd, sdhr)
    else:
        rmssd = 0
        sdnn = 0    
        nn50 = 0         
        pnn50 = 0  
        nn20 = 0    
        pnn20 = 0   
        mrri = 0    
        mhr = 0  

    return  mrri, sdnn, sdsd, nn50, pnn50, nn20, pnn20, rmssd, mhr, sdhr

def frequency_domain(rri,ulf_band=(0, 0.0033),vlf_band=(0, 0.04),lf_band=(0.04, 0.15), hf_band=(0.15, 0.4),
                     fs = 4.0, interp_method = 'cubic',**kwargs):
    try:
        if not len(rri) == 0:
            alltime = np.cumsum(rri) / 1000.0 #時間差值 每個RR間隔相對於第一個RR間隔的時間偏移
            # print(alltime)
            time = alltime - alltime[0]
            # print(time)
            #interpolate:'cubic','linear'
            if interp_method is not None:
                time_resolution = 1 / float(fs)
                time_interp = np.arange(0, time[-1] + time_resolution, time_resolution)
                if interp_method == 'cubic':
                    tck = interpolate.splrep(time, rri, s=0)
                    rri = interpolate.splev(time_interp, tck, der=0)
                elif interp_method == 'linear':
                    rri = np.interp(time_interp, time, rri)
            #psd:'welch'
            fxx, pxx = welch(x = rri, fs = fs, **kwargs)

            #壓力上升==增加交感減少副交感的控調==LF增加==hf減少
            ulf_ind = np.logical_and(fxx>=ulf_band[0],fxx<ulf_band[1])
            # print(ulf_ind)
            vlf_ind = np.logical_and(fxx>=vlf_band[0],fxx<vlf_band[1])
            # print(vlf_ind)
            lf_ind = np.logical_and(fxx>=lf_band[0],fxx<lf_band[1])         
            hf_ind = np.logical_and(fxx>=hf_band[0],fxx<hf_band[1])         
            ulf_p = np.trapz(y=pxx[ulf_ind],x=fxx[ulf_ind])
            # print(ulf_p)
            vlf_p = np.trapz(y=pxx[vlf_ind],x=fxx[vlf_ind])                 #very low-frequency   極低頻範圍正常心跳間期之變異   ≤0.04Hz
            # print(vlf_p)
            lf_p = np.trapz(y=pxx[lf_ind],x=fxx[lf_ind])                    #低頻範圍功率          低頻範圍正常心跳間期之變異數    代表交感與副交感神經活性
            # print(lf_p)
            hf_p = np.trapz(y=pxx[hf_ind],x=fxx[hf_ind])                    #高頻範圍功率          高頻範圍正常心跳間期之變異數    代表副交感神經活性
            total_p = ulf_p + vlf_p + lf_p + hf_p                                   #總功率     全部正常心跳間期之變異數高頻、低頻、極低頻的總和
            lf_hf = lf_p/hf_p                                               #低、高頻功率的比值    代表自律神經 活性平衡    就是CSI
            # print(lf_hf)
            lfnu = (lf_p / (total_p - vlf_p)) * 100                         #標準化低頻功率        交感神經活性 定量指標
            hfnu = (hf_p / (total_p - vlf_p)) * 100                         #標準化高頻功率        副交感神經活性 定量指標

            cvi = hf_p/(lf_p+hf_p)  #cvi
            # print(cvi)

        else:
            lf_ind = 0
            hf_ind = 0
            ulf_p = 0
            vlf_p = 0
            lf_p = 0
            hf_p = 0
            total_p = 0
            lf_hf = 0
            lfnu = 0
            hfnu = 0
            cvi = 0

        return ulf_p, vlf_p, lf_p, hf_p, lf_hf, lfnu, hfnu, total_p, cvi
    
    except:
        lf_ind = 0
        hf_ind = 0
        ulf_p = 0
        vlf_p = 0
        lf_p = 0
        hf_p = 0
        total_p = 0
        lf_hf = 0
        lfnu = 0
        hfnu = 0
        cvi = 0

        return ulf_p, vlf_p, lf_p, hf_p, lf_hf, lfnu, hfnu, total_p, cvi

def non_linear(rri):
    sd1, sd2 = _poincare(rri)
    #return dict(zip(['sd1', 'sd2'], [sd1, sd2]))
    # print("sd1 sd2", sd1, sd2)
    sampen = get_sampen(rri)
    # print(sampen)
    return sd1, sd2, sampen['sampen']

def _poincare(rri):
    diff_rri = np.diff(rri)
    sd1 = np.sqrt(np.std(diff_rri, ddof=1) ** 2 * 0.5)
    sd2 = np.sqrt(2 * np.std(rri, ddof=1) ** 2 - 0.5 * np.std(diff_rri,
                                                              ddof=1) ** 2)
    return sd1, sd2

def Cleandata(data):
    filtered_list = [x for x in data if x >= 1]
    return filtered_list

def CleanRRdata(data):
    filtered_list = [x for x in data if x >= 100]
    return filtered_list

def read_data(path):
    '''
    根據資料格式(csv or json or txt)讀進來
    '''
    #先透過open將檔案以string的方式讀取，
    #接著用split'\n'來切割為list 每個element皆為正確json格式過並濾掉錯誤的最後一行
    #透過迴圈抓出json中需要的attribute再使用另外兩個對應的list接起來
    time = []
    heart = []
    rr = []
    stresslv = []
    spo2 = []
    resp = []
    strData = open(path,"r",encoding='utf-8').read()
    listData = strData.split("\n")
    listData = listData[:len(listData)-1]
    #print(listData)   #一維  [{} ... {}]
    all_data = []
    for data in listData:
        temp = []
        if(json.loads(data)['time'] == "null"):
            continue
        temp.append(json.loads(data)['time'])
        temp.append(json.loads(data)['heartRate'])
        temp.append(json.loads(data)['heartRateVariability'])
        temp.append(json.loads(data)['stressLevel'])
        temp.append(json.loads(data)['SPO2'])
        temp.append(json.loads(data)['respiration'])
        all_data.append(temp)
    df = []
    df = pd.DataFrame(all_data,columns=['time', 'heartRate', 'heartRateVariability', 'stressLevel', 'SPO2', 'respiration'])
    # heart = Cleandata(heart)
    # stresslv = Cleandata(stresslv)
    # spo2 = Cleandata(spo2)
    # resp = Cleandata(resp)
    return df

def calculate_features(p):
    '''
    calculate features
    '''
    p = np.array(p)
    # print(p)
    mrri, sdnn, sdsd, nn50, pnn50, nn20, pnn20, rmssd, mhr, sdhr = time_domain(p)
    ulf_p, vlf_p, lf_p, hf_p, lf_hf, lfnu, hfnu, total_p, cvi = frequency_domain(p, ulf_band=(0, 0.0033), vlf_band=(0, 0.04),lf_band=(0.04, 0.15), hf_band=(0.15, 0.4),
                    fs = 4.0, interp_method = 'cubic',detrend='linear')
    sd1, sd2, sampen = non_linear(p)
    features = []
    # time domain
    features.append(mrri)
    features.append(sdnn)
    features.append(sdsd)
    features.append(nn50)
    features.append(pnn50)
    features.append(nn20)
    features.append(pnn20)
    features.append(rmssd)
    features.append(mhr)
    features.append(sdhr)
    # frequency domain
    features.append(ulf_p)
    features.append(vlf_p) 
    features.append(lf_p)
    features.append(hf_p)
    features.append(lf_hf)
    features.append(lfnu)
    features.append(hfnu)
    features.append(total_p)
    # nonlinear domain
    features.append(cvi)
    features.append(sd1)
    features.append(sd2)
    features.append(sampen)
    
    return features

def av_std_var(data):
    average = np.mean(data)
    std_deviation = np.std(data)
    variance = np.var(data)
    try:
        growth_rate = (data[4999] - data[0]) / data[0]
    except:
        growth_rate = 0
    return average, std_deviation, variance, growth_rate

def cal_time(time):
    date_str1 = time[0]
    date_str2 = time[len(time)-1]
    date_format1 = "%Y-%m-%d %H:%M:%S:%f"
    date_format2 = "%Y-%m-%d %H-%M-%S-%f"
    try:
        date1 = datetime.strptime(date_str1, date_format1)
        date2 = datetime.strptime(date_str2, date_format1)
    except:
        date1 = datetime.strptime(date_str1, date_format2)
        date2 = datetime.strptime(date_str2, date_format2)
    seconds_difference = (date2 - date1).total_seconds()
    return seconds_difference

def cal_real_rr(rr):
    realrr = []
    realrr.append(rr[0])
    i = 0
    for r in rr:
        if realrr[i] != r:
            realrr.append(r)
            i += 1
    return realrr

def showgraph(data):
    for i in range(4):
        first = data[0][i]
        second = data[1][i]
        third = data[2][i]
        first = np.array(first)
        second = np.array(second)
        third = np.array(third)
        # 將列表插值為相同長度
        # max_length = max(len(first), len(second), len(third))
        # time = np.linspace(1, len(first), max_length)  # 假設時間是均勻分布的
        # first_interp = np.interp(time, range(1, len(first) + 1), first)
        # second_interp = np.interp(time, range(1, len(second) + 1), second)
        # third_interp = np.interp(time, range(1, len(third) + 1), third)
        # plt.plot(time, first_interp, label='First')
        # plt.plot(time, second_interp, label='Second')
        # plt.plot(time, third_interp, label='Third')
        
        min_length = min(len(first), len(second), len(third))
        # min_length = min(len(first), len(second))
        first_sample_step = len(first) // min_length
        second_sample_step = len(second) // min_length
        third_sample_step = len(third) // min_length
        first_sampled = [np.mean(first[i:i+first_sample_step]) for i in range(0, len(first), first_sample_step)]
        second_sampled = [np.mean(second[i:i+second_sample_step]) for i in range(0, len(second), second_sample_step)]
        third_sampled = [np.mean(third[i:i+third_sample_step]) for i in range(0, len(third), third_sample_step)]
        first_mean = np.mean(first_sampled)
        second_mean = np.mean(second_sampled)
        third_mean = np.mean(third_sampled)
        first_std = np.std(first_sampled)
        second_std = np.std(second_sampled)
        third_std = np.std(third_sampled)
        first_var = np.var(first_sampled)
        second_var = np.var(second_sampled)
        third_var = np.var(third_sampled)

        plt.plot(first_sampled, label='First')
        plt.plot(second_sampled, label='Second')
        plt.plot(third_sampled, label='Third')

        plt.xlabel('Number of Data')
        if(i == 0):
            plt.ylabel('Heart Rate')
            plt.title('Heart Rate Data')
        if(i == 1):
            plt.ylabel('stresslv')
            plt.title('stresslv Data')
        if(i == 2):
            plt.ylabel('spo2')
            plt.title('spo2 Data')
        if(i == 3):
            plt.ylabel('resp')
            plt.title('resp Data')
        plt.legend()
        plt.show()

        # plt.figure(figsize=(8, 6))
        # plt.boxplot([first_sampled, second_sampled, third_sampled], labels=['First', 'Second', 'Third'])
        # plt.scatter([1, 2, 3], [first_mean, second_mean, third_mean], color='red', label='Mean', zorder=3)
        # plt.errorbar([1, 2, 3], [first_mean, second_mean, third_mean], yerr=[first_std, second_std, third_std],
        #             fmt='o', color='black', label='Standard Deviation', zorder=2)
        # plt.xlabel('Area')
        # if(i == 0):
        #     plt.ylabel('Heart Rate')
        #     plt.title('Heart Rate Data')
        # if(i == 1):
        #     plt.ylabel('stresslv')
        #     plt.title('stresslv Data')
        # if(i == 2):
        #     plt.ylabel('spo2')
        #     plt.title('spo2 Data')
        # if(i == 3):
        #     plt.ylabel('resp')
        #     plt.title('resp Data')
        # plt.legend()
        # plt.grid(True)
        # plt.show()


def eachArea(path2):
    strData = open(path2, 'r', encoding='utf-8').read()
    listData = strData.split('\n')
    listData = listData[:len(listData)-1]
    game = []
    for data in listData:
        temp = []
        temp.append(json.loads(data)['start_time'].replace('+08:00', ''))
        temp.append(json.loads(data)['end_time'].replace('+08:00', ''))
        temp.append(json.loads(data)['test_total'])
        temp.append(json.loads(data)['test_correct'])
        temp.append(json.loads(data)['collision_count'])
        temp.append(json.loads(data)['answer_total'])
        temp.append(json.loads(data)['shouldCorrect'])
        game.append(temp)
    df = []
    df = pd.DataFrame(game,columns=['start_time', 'end_time', 'test_total', 'test_correct', 'collision_count', 'answer_total', 'shouldCorrect'])
    return df

def cut_by_timer(data, timer_data):
    phase = [[] for i in range(3)]
    timer_data = np.array(timer_data)
    # print(phase)
    date_format1 = "%Y-%m-%d %H:%M:%S:%f"
    date_format2 = "%Y-%m-%d %H-%M-%S-%f"
    for tv in range(len(data)):
        try:
            time = datetime.strptime(data['time'][tv], date_format1)
        except:
            time = datetime.strptime(data['time'][tv], date_format2)
        # time = pd.to_datetime(data['time'][tv])
        if time < pd.to_datetime(timer_data[0][1]): #第一圈
            phase[0].append(data.loc[tv].values.flatten().tolist())
        elif time < pd.to_datetime(timer_data[1][1]) and time >= pd.to_datetime(timer_data[1][0]): #第二圈
            phase[1].append(data.loc[tv].values.flatten().tolist())
        elif time < pd.to_datetime(timer_data[2][1]) and time >= pd.to_datetime(timer_data[2][0]): #第三圈
            phase[2].append(data.loc[tv].values.flatten().tolist())
    # for i in range(len(phase)):
    #     if len(phase[i]) > 15000:
    #         phase[i] = phase[i][:15000]
    #     phase[i] = pd.DataFrame(phase[i],columns=['time', 'heartRate', 'heartRateVariability', 'stressLevel', 'SPO2', 'respiration'])
    for i in range(len(phase)):
        phase[i] = pd.DataFrame(phase[i],columns=['time', 'heartRate', 'heartRateVariability', 'stressLevel', 'SPO2', 'respiration'])
    # print(len(phase))
    # print(len(phase[0]),len(phase[1]),len(phase[2]))
    return phase

def data_prepro(path1, path2):
    # print('json data process')
    game = eachArea(path2)
    data = read_data(path1)
    # time , heart, rr, stresslv, spo2, resp = read_data(path1)
    phase = cut_by_timer(data, game[['start_time', 'end_time']])
    game = np.array(game)
    result_data = []
    toshow = [[] for i in range(3)]
    i = 0
    for p in phase:
        time = np.array(p['time'])
        heart = np.array(p['heartRate'])
        rr = np.array(p['heartRateVariability'])
        stresslv = np.array(p['stressLevel'])
        spo2 = np.array(p['SPO2'])
        resp = np.array(p['respiration'])
        heart = Cleandata(heart)
        stresslv = Cleandata(stresslv)
        spo2 = Cleandata(spo2)
        resp = Cleandata(resp)
        rr = CleanRRdata(rr)
        print(len(rr))
        realrr = cal_real_rr(rr)
        print(len(realrr))
        total_time = cal_time(time)
        print(total_time)
        heart_mean, heart_std, heart_var, heart_growth = av_std_var(heart)
        # rr_mean, rr_std, rr_var = av_std_var(rr)
        stresslv_mean, stresslv_std, stresslv_var, stresslv_growth = av_std_var(stresslv)
        spo2_mean, spo2_std, spo2_var, spo2_growth = av_std_var(spo2)
        resp_mean, resp_std, resp_var, resp_growth = av_std_var(resp)
        toshow[i].append(heart)
        toshow[i].append(stresslv)
        toshow[i].append(spo2)
        toshow[i].append(resp)
# ['start_time', 'end_time', 'test_total', 'test_correct', 'collision_count']"answer_total":5,"shouldCorrect":5
        should_correct = game[i][6]
        answer_total = game[i][5]
        test_correct = game[i][3]
        collision_count = game[i][4]
        try:
            answer_freq = answer_total/should_correct
        except:
            answer_freq = 0
        try:
            correct = test_correct/answer_total
        except:
            correct = 0
        try:
            wrong = (answer_total-test_correct)/answer_total
        except:
            wrong = 0
        try:
            score = test_correct/should_correct
        except:
            score = 0
        i = i + 1
        print("hr mean std var growth :", heart_mean, heart_std, heart_var, heart_growth)
        # print("rr mean std var :", rr_mean, rr_std, stresslv_var)
        print("stresslv mean std var growth :", stresslv_mean, stresslv_std, stresslv_var, stresslv_growth)
        print("spo2 mean std var growth :", spo2_mean, spo2_std, spo2_var, spo2_growth)
        print("resp mean std var growth :", resp_mean, resp_std, resp_var, resp_growth)
        total_features = calculate_features(realrr)
        total_features.append(heart_mean)
        total_features.append(heart_std)
        total_features.append(heart_var)
        total_features.append(heart_growth)
        total_features.append(stresslv_mean)
        total_features.append(stresslv_std)
        total_features.append(stresslv_var)
        total_features.append(stresslv_growth)
        total_features.append(spo2_mean)
        total_features.append(spo2_std)
        total_features.append(spo2_var)
        total_features.append(spo2_growth)
        total_features.append(resp_mean)
        total_features.append(resp_std)
        total_features.append(resp_var)
        total_features.append(resp_growth)
        total_features.append(answer_freq)
        total_features.append(correct)
        total_features.append(wrong)
        total_features.append(score)
        total_features.append(collision_count)
        print(total_features)
        result_data.append(total_features)
    # showgraph(toshow)
    return result_data

def csv_output(path1 , path2):
    '''
    輸出 featureCSV
    '''
    total_features = data_prepro(path1 , path2)
    #print(total_features)

    with open('jeep_hrv.csv', 'w', newline='') as csvfile:
        # 建立 CSV 檔寫入器
        writer = csv.writer(csvfile)
        writer.writerow(['MEAN_RR', 'RMSSD', 'pNN50', 'VLF', 'LF', 'HF', 'LF_HF', 'LF_NU'])
        for c in total_features:
            writer.writerow(c)
    csvfile.close()

# path = 'C:/Users/Jackson/OneDrive/LAB/資料/壓力訓練/測試資料/RestingState/202401160404_test_FPre/test_FPre_garmin.json'
# path1 = 'C:/Users/Jackson/OneDrive/LAB/資料/壓力訓練/測試資料/WarCar/202403150424_N06_D_Stroop/N06_D_Stroop_garmin.json'
# path2 = 'C:/Users/Jackson/OneDrive/LAB/資料/壓力訓練/測試資料/WarCar/202403150424_N06_D_Stroop/N06_D_Stroop_Game.json'
# path1 = 'C:/Users/Jackson/OneDrive/LAB/資料/壓力訓練/測試資料/WarCar/202403150456_N06_D_CPT/N06_D_CPT_garmin.json'
# path2 = 'C:/Users/Jackson/OneDrive/LAB/資料/壓力訓練/測試資料/WarCar/202403150456_N06_D_CPT/N06_D_CPT_Game.json'
# path = 'C:/Users/Jackson/OneDrive/LAB/資料/壓力訓練/正式收案/WarCar'
# path = 'C:/Users/Jackson/OneDrive/LAB/資料/壓力訓練/正式收案/分開算'
# path = 'C:/Users/Jackson/OneDrive/LAB/資料/壓力訓練/正式收案/WarCar/General第二次/CPT'
# path = 'C:/Users/Jackson/OneDrive/LAB/資料/壓力訓練/正式收案/WarCar/General第二次/Stroop'
# path = 'C:/Users/Jackson/OneDrive/LAB/資料/壓力訓練/正式收案/WarCar/Sport第二次/CPT'
path = 'C:/Users/Jackson/OneDrive/LAB/資料/壓力訓練/正式收案/WarCar/Sport第二次/Stroop'
data_path = Path(path)
cpt = [x for x in data_path.rglob('*CPT') if x.is_dir()]
stroop = [x for x in data_path.rglob('*Stroop') if x.is_dir()]
all_data = []
for fd in cpt:
    print(fd)
    garmin = Path(list(fd.rglob('*garmin.json'))[0], lines=True)
    game = Path(list(fd.rglob('*Game.json'))[0], lines=True)
    temp = data_prepro(garmin, game)
    for t in temp:
        all_data.append(t)
all_data.append(['mrri', 'sdnn','sdsd', 'nn50','pnn50', 'nn20', 'pnn20', 'rmssd', 'mhr','sdhr',
                                            'ulf_p', 'vlf_p','lf_p', 'hf_p', 'lf_hf', 'lfnu', 'hfnu', 'total_p', 'cvi', 
                                            'sd1', 'sd2', 'sampen', 'heart_mean', 'heart_std', 'heart_var', 'heart_growth', 'stresslv_mean', 'stresslv_std', 'stresslv_var', 'stresslv_growth'
                                            , 'spo2_mean', 'spo2_std', 'spo2_var', 'spo2_growth', 'resp_mean', 'resp_std', 'resp_var', 'resp_growth'
                                            , 'answer_freq', 'correct', 'wrong', 'score', 'collision_count'])
for fd in stroop:
    print(fd)
    garmin = Path(list(fd.rglob('*garmin.json'))[0], lines=True)
    game = Path(list(fd.rglob('*Game.json'))[0], lines=True)
    temp = data_prepro(garmin, game)
    for t in temp:
        all_data.append(t)

final_data = pd.DataFrame(all_data, columns = ['mrri', 'sdnn','sdsd', 'nn50','pnn50', 'nn20', 'pnn20', 'rmssd', 'mhr','sdhr',
                                            'ulf_p', 'vlf_p','lf_p', 'hf_p', 'lf_hf', 'lfnu', 'hfnu', 'total_p', 'cvi', 
                                            'sd1', 'sd2', 'sampen', 'heart_mean', 'heart_std', 'heart_var', 'heart_growth', 'stresslv_mean', 'stresslv_std', 'stresslv_var', 'stresslv_growth'
                                            , 'spo2_mean', 'spo2_std', 'spo2_var', 'spo2_growth', 'resp_mean', 'resp_std', 'resp_var', 'resp_growth'
                                            , 'answer_freq', 'correct', 'wrong', 'score', 'collision_count'])

final_data.to_csv('C:/Users/Jackson/Downloads/result.csv')

