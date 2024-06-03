import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

# 데이터 불러오기
path = r'D:\AllProjectRepo\Inha_ML_2024_1_Proj\soccer\full_data.csv'
data = pd.read_csv(path)

# 필요한 열 선택
columns = ['League', 'Home', 'Away', 'H_Score', 'Away_Score', 'stadium', 'date', 'attendance']
data = data[columns]

# 결측치 처리
data.dropna(subset=['attendance'], inplace=True)

# 날짜 열을 datetime 형식으로 변환
data['date'] = pd.to_datetime(data['date'])

# 연도, 월, 일로 분리
data['year'] = data['date'].dt.year
data['month'] = data['date'].dt.month
data['day'] = data['date'].dt.day

# 카테고리형 변수를 숫자로 변환
data = pd.get_dummies(data, columns=['home_team', 'away_team', 'stadium'], drop_first=True)

# 특징과 타겟 분리
X = data.drop(columns=['date', 'attendance'])
y = data['attendance']

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 스케일링
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# SVR 모델 학습
svr = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
svr.fit(X_train_scaled, y_train)

# 예측
y_pred = svr.predict(X_test_scaled)

# 평가
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f'RMSE: {rmse}')
