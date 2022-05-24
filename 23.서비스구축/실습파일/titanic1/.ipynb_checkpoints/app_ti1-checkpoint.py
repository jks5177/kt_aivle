#############################################
# 타이타닉 생존 예측 웹 서비스
#############################################

# 1. 라이브러리 로딩
from flask_restful import reqparse 
from flask import Flask
import numpy as np 
import pandas as pd
import joblib
import json

# -------------------------------------------------------------
# 2. 앱 선언 및 필요 함수, 변수 선언
# 서버 앱 선언
app = Flask(__name__) 

# 필요한 변수 선언
features = ['Pclass', 'Sex', 'Age', 'SibSp' ,'Parch', 'Fare', 'Embarked']
target = 'Survived'
imputer1_list = ['Embarked']
cat = {'Sex':["female", "male"], 'Embarked':["C", "Q", "S"], 'Pclass':[1,2,3]}

# 필요한 함수 선언
def titanic_fe(df):
    temp = df.copy()
    # Family 변수 추가
    temp['Family'] = temp['SibSp'].astype('int32') + temp['Parch'].astype('int32') + 1
    temp.drop(['SibSp', 'Parch'], axis = 1, inplace = True)

    # OOO 추가...
    return temp

def titanic_dumm(df, cat):
    for k, v in cat.items():
        df[k] = pd.Categorical(df[k], categories=v, ordered=False)
    df = pd.get_dummies(df, columns =cat.keys(), drop_first = 1)
    return df

def titanic_datapipeline(df, simpleimputer, simple_impute_list, dumm_list, knnimputer, scaler):

    temp = df.copy()

    # Feature Engineering
    temp = titanic_fe(temp)

    # NaN 조치① : SimpleImputer
    temp[simple_impute_list] = simpleimputer.fit_transform(temp[simple_impute_list])

    # 가변수화
    temp = titanic_dumm(temp, dumm_list)

    x_cols = list(temp)
    # NaN 조치② : KNNImputer
    temp = knnimputer.transform(temp)

    # 스케일링
    temp = scaler.transform(temp)

    return pd.DataFrame(temp, columns = x_cols)


# -------------------------------------------------------------
# 3. 웹서비스

@app.route('/predict/', methods=['POST']) 
def predict(): 
    
    # 입력받은 json 파일에서 정보 뽑기(파싱)
    parser = reqparse.RequestParser() 
    for v in features :
        parser.add_argument(v, action='append') 

    # 뽑은 값을 딕셔너리로 저장
    args = parser.parse_args() 
        
    # 딕셔너리를 데이터프레임(2차원으로 만들기)
    x_input = pd.DataFrame(args)
    print(x_input)
    
    # 전처리
    x_input = titanic_datapipeline(x_input, imputer1, imputer1_list, cat, imputer2, scaler)
    print(x_input)
    # 예측. 결과는 넘파이 어레이
    pred = model.predict(x_input) 
    
    # 결과를 식별가능한 문자로 변환(0,1로 반환할 때, 타입오류가 날 수 있음.)
    result = np.where(pred == 0, 'Died','Survived')
    
    # result : json 형태로 전송해야 한다.
    out = {'pred': list(result)} 
   
    return out

# -------------------------------------------------------------

# 4.웹서비스 직접 실행시 수행 
if __name__ == '__main__': 

    # 전처리 + 모델 불러오기
    imputer1 = joblib.load('preprocess/imputer1_ti1.pkl')
    imputer2 = joblib.load('preprocess/imputer2_ti1.pkl')
    scaler = joblib.load('preprocess/scaler_ti1.pkl')
    model = joblib.load('model/model_ti1.pkl')
    
    # 웹서비스 실행 및 접속 정보
    app.run(host = '127.0.0.1', port=8080, debug=True)

# -------------------------------------------------------------
