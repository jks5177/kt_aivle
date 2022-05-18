################################################################
# 타이타닉 생존 예측 웹 서비스
################################################################
# -------------------------------------------------------------
# 1. 라이브러리 로딩
from flask_restful import reqparse 
from flask import Flask
import numpy as np 
import pandas as pd
import joblib
import json
import datetime
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# -------------------------------------------------------------
# 2. 앱 선언 및 필요 함수, 변수 선언
# 서버 앱 선언
app = Flask(__name__) 

# 필요한 변수 선언
use_cols = joblib.load('preprocess/use_cols.plk')
simpleimputer_list = joblib.load('preprocess/simpleimputer_list.plk')
dumm_list = joblib.load('preprocess/dumm_list.plk')

# 시간 데이터 - 파일 저장을 위해 사용
now = datetime.datetime.now()
date = now.strftime('%Y%m%d')
timestamp = now.strftime('%Y%m%d_%H%M%S')

# 필요한 함수 선언
def mobile_to_categorical(df, cat) :
    tmp = df.copy()
    for k, v in cat.items() :
        tmp[k] = pd.Categorical(tmp[k], categories=v, ordered=False)
    tmp = pd.get_dummies(tmp, columns=cat.keys(), drop_first=True)
    return tmp

def mobile_datapipeline(df, simpleimputer, simpleimputer_list, dumm_list, scaler, knnimputer) :
    tmp = df.copy()
    
    # simpleImputer
    tmp[simpleimputer_list] = simpleimputer.transform(tmp[simpleimputer_list])

    # 가변수화
    tmp = mobile_to_categorical(tmp, dumm_list)
    
    x_col = list(tmp)

    # 스케일링
    tmp = scaler.transform(tmp)

    # KNNImputer
    tmp = knnimputer.transform(tmp)

    # DataFrame 변환
    return pd.DataFrame(tmp, columns=x_col)

# -------------------------------------------------------------
# 3. 웹서비스

@app.route('/predict/', methods=['POST']) 
def predict(): 
    # 입력받은 json 파일에서 정보 뽑기(파싱)
    parser = reqparse.RequestParser()
    for v in use_cols :
        parser.add_argument(v, action='append')
    
    # 뽑은 값을 딕셔너리로 저장
    args = parser.parse_args()
    
    # 딕셔너리를 데이터프레임(2차원으로 만들기)
    x_input = pd.DataFrame(args)
    
    # 전처리
    x_input = mobile_datapipeline(x_input, simpleimputer, simpleimputer_list, dumm_list, scaler, knnimputer)
    
    # 예측/ 결과는 넘파이 어레이
    pred = model.predict(x_input)
    
    # 결과를 식별가능한 문자로 변환
    result = np.where(pred==0, 'stay', 'leave')
    
    # result : json 형태로 전송해야 한다.
    out = {'pred' : list(result)}
 
    # 데이터 저장
    file_data = x_input.copy()
    
    file_data['time'] = timestamp
    file_data['pred'] = out['pred']
    
    if os.path.exists('data/update_data_'+date+'.csv') :
        tmp = pd.read_csv('data/update_data_'+date+'.csv')
        file_data = pd.concat([tmp, file_data])
    
    file_data.to_csv('data/update_data_'+date+'.csv', index=False)
      
    return out

@app.route('/feature')
def feature_name() :
    return {'feature_name':use_cols}

@app.update('/update')
def model_update() :
    # 데이터 불러오기
    file_list = os.listdir('data/')
    update_data = []
    for file_name in file_list :
        if 'update' in file_name and file_name.endwith('.csv') :
            tmp = pd.read_csv(file_name)
            data = tmp[use_cols]
            data['CHURN'] = np.where(tmp['pred']=='stay', 0, 1)
            update_data.append(data)
    tmp = pd.read_csv('data/mobile.csv')
    origin_data = tmp[use_cols]
    origin_data['CHURN'] = tmp['CHURN']
    
    new_data = pd.concat(origin_data, update_data)
    
    # 데이터 분할하기
    target = 'CHURN'
    x = new_data.drop(target, axis=1)
    y = new_data[target]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    
    # 전처리
    x_train = mobile_datapipeline(x_input, simpleimputer, simpleimputer_list, dumm_list, scaler, knnimputer)
    
    # 새로운 모델 학습하기
    model_new = joblib.load('model.plk')
    model_new.fit(x_train, y_train)
    new_pred = model.predict(x_test)
    
    joblib.dump(model_new, 'model/model'+timestamp+'.plk')
    
    # 기존 모델 불러오기
    model_origin = joblib.load('model.plk')
    origin_pred = model.predict(x_test)
    
    # 모델 평가 비교 : accuracy
    new_acc_score = accuracy_score(y_test, new_pred)
    origin_acc_score = accuracy_score(y_test, origin_pred)
    
    if new_acc_score > origin_acc_score :
        joblib.dump(model_new, 'model/model.plk')
  
    return print('save model')

# -------------------------------------------------------------
# 4.웹서비스 직접 실행시 수행 
if __name__ == '__main__': 

    # 전처리 + 모델 불러오기
    simpleimputer = joblib.load('preprocess/simpleimputer.plk')
    scaler = joblib.load('preprocess/minmaxscaler.plk')
    knnimputer = joblib.load('preprocess/knnimputer.plk')
    
    model = joblib.load('model/model.plk')
    
    # 웹서비스 실행 및 접속 정보
    app.run(host='127.0.0.1', port=8080, debug=True)

# -------------------------------------------------------------
