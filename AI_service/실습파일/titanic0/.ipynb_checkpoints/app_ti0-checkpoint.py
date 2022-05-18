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

# -------------------------------------------------------------
# 2. 앱 선언 및 필요 함수, 변수 선언
# 서버 앱 선언
app = Flask(__name__) 

# 필요한 변수 선언

# 필요한 함수 선언


# -------------------------------------------------------------
# 3. 웹서비스
@app.route('/predict/', methods=['POST']) 
def predict(): 
    
    # 입력받은 json 파일에서 정보 뽑기(파싱)
    parser = reqparse.RequestParser() 
    parser.add_argument('Pclass', action='append') 
    parser.add_argument('Age', action='append') 
    parser.add_argument('Sex', action='append') 

    # 뽑은 값을 딕셔너리로 저장
    args = parser.parse_args() 
        
    # 딕셔너리를 데이터프레임(2차원으로 만들기)
    x_input = pd.DataFrame(args)
    
    # 전처리
    
    # 예측. 결과는 넘파이 어레이
    pred = model.predict(x_input) 
    
    # 식별가능한 문자로 변환(0,1로 반환할 때, 타입오류 날 수 있음.)
    result = np.where(pred == 0, 'Died','Survived')
    
    # result : json 형태로 전송해야 한다.
    out = {'pred': list(result)} 
   
    return out

# -------------------------------------------------------------
# 4.웹서비스 직접 실행시 수행 
if __name__ == '__main__': 

    # 전처리 + 모델 불러오기
    model = joblib.load('model/model_ti0.pkl')
    
    # 웹서비스 실행 및 접속 정보
    app.run(host = '127.0.0.1', port=8080, debug=True)

# -------------------------------------------------------------
