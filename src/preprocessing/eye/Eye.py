import pandas as pd
import json
class Eye:
    def __init__(self):
        
        pass
    
    def __enter__(self):
        return self
    
    def __exit__(self, *exc_info):
        self.close()
        
    def close(self):
        pass

    def load_gaze(self) -> pd.DataFrame:
        path = '/Users/shawn/Desktop/acrophobia/elevatordata/54273/acrophobiapico_elevator_20241108033202_EyeCombinedData.json'
        strData = open(path, 'r', encoding='utf-8-sig').read()
        listData = strData.split('\n')
        listData = listData[:len(listData)-1]
        eye_data = []
        for data in listData:
            temp = []
            temp.append(json.loads(data)['Timestamp'].replace('+08:00', ''))
            temp.append(json.loads(data)['CombineEyeGazePoint'])
            temp.append(json.loads(data)['CombineEyeGazeVector'])
            if temp[1]['x'] == 0.0 :
                continue
            if temp[2]['x'] == 0.0 :
                continue
            eye_data.append(temp)
        df = []
        df = pd.DataFrame(eye_data,columns=['Timestamp', 'CombineEyeGazePoint', 'CombineEyeGazeVector'])
        return df
    
    def load_position(self) -> pd.DataFrame:
        path = '/Users/shawn/Desktop/acrophobia/elevatordata/54273/acrophobiapico_elevator_20241108033202_EyeLeftRightData.json'
        strData = open(path, 'r', encoding='utf-8-sig').read()
        listData = strData.split('\n')
        listData = listData[:len(listData)-1]
        eye_data = []
        for data in listData:
            temp = []
            temp.append(json.loads(data)['Timestamp'].replace('+08:00', ''))
            temp.append(json.loads(data)['LeftEyePositionGuide'])
            temp.append(json.loads(data)['LeftEyeOpenness'])
            temp.append(json.loads(data)['RightEyePositionGuide'])
            temp.append(json.loads(data)['RightEyeOpenness'])
            if temp[1]['x'] == 0.0 :
                continue
            if temp[3]['x'] == 0.0 :
                continue
            eye_data.append(temp)
        df = []
        df = pd.DataFrame(eye_data,columns=['Timestamp', 'LeftEyePositionGuide', 'LeftEyeOpenness','RightEyePositionGuide','RightEyeOpenness'])
        print(df.head(50))
        return df
        
    def load_focus(self) -> pd.DataFrame:
        path = rf'elevatordata\54273\acrophobiapico_elevator_20241108033202_EyeFocusData.json'
        strData = open(path, 'r', encoding='utf-8-sig').read()
        listData = strData.split('\n')
        listData = listData[:len(listData)-1]
        eye_data = []
        for data in listData:
            temp = []
            temp.append(json.loads(data)['Timestamp'].replace('+08:00', ''))
            temp.append(json.loads(data)['FocusName'])
            temp.append(json.loads(data)['FocusPoint'])
            temp.append(json.loads(data)['FocusNormal'])
            temp.append(json.loads(data)['FocusDistance'])
            eye_data.append(temp)
        df = []
        df = pd.DataFrame(eye_data,columns=['Timestamp', 'FocusName', 'FocusPoint','FocusNormal','FocusDistance'])
        print(df.head(50))
        return df
    
    def load_openness(self) -> pd.DataFrame:
        path = rf"elevatordata\54273\acrophobiapico_elevator_20241108033202_EyeLeftRightData.json"
        str_data = open(path, 'r', encoding='utf-8-sig').read()
        listData = str_data.split('\n')
        listData = listData[:len(listData)-1]
        eye_data = []
        for data in listData:
            json_data = json.loads(data)
            temp = []
            temp.append(json_data['Timestamp'].replace('+08:00',''))
            temp.append(json_data['LeftEyeOpenness'])
            temp.append(json_data['RightEyeOpenness'])
            eye_data.append(temp)
        df = pd.DataFrame(eye_data, columns=['Timestamp','LeftEyeOpenness','RightEyeOpenness'])
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        print(df.head(50))
        return df
    
    def preprocessing(self):
        pass

if __name__ == "__main__":
    with Eye() as eye:
        eye.load_openness()





