import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# 데이터 불러오기
file_path = r'D:\AllProjectRepo\Inha_ML_2024_1_Proj\soccer\bulidata.csv'
data = pd.read_csv(file_path)

# NaN 데이터 제거
data.dropna(subset=['LEAGUE_NAME', 'LOCATION', 'HOME_TEAM', 'AWAY_TEAM', 'VIEWER'], inplace=True)

# 날짜를 datetime 형식으로 변환
data['MATCH_DATE'] = pd.to_datetime(data['MATCH_DATE'])

# 연도, 월, 일 특징 생성
data['YEAR'] = data['MATCH_DATE'].dt.year
data['MONTH'] = data['MATCH_DATE'].dt.month
data['DAY'] = data['MATCH_DATE'].dt.day

# 필요한 열 선택 및 특징/레이블 분리
data = data[['LEAGUE_NAME', 'LOCATION', 'HOME_TEAM', 'AWAY_TEAM', 'YEAR', 'MONTH', 'DAY', 'VIEWER']]
X = data.drop(columns=['VIEWER'])
y = data['VIEWER']

# 원-핫 인코딩
X = pd.get_dummies(X, columns=['LEAGUE_NAME', 'LOCATION', 'HOME_TEAM', 'AWAY_TEAM'], drop_first=True)

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 스케일링
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 랜덤 포레스트 회귀 모델 학습
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train)

# 예측
y_pred_rf = rf.predict(X_test_scaled)

# 평가 지표 계산
mse_rf = mean_squared_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mse_rf)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print(f'Random Forest RMSE: {rmse_rf}')
print(f'Random Forest MAE: {mae_rf}')
print(f'Random Forest R^2 Score: {r2_rf}')

# 예측 결과 시각화
plt.figure(figsize=(14, 7))

# 실제 값과 예측 값 비교
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred_rf, alpha=0.5)
plt.xlabel('Actual Viewer Count')
plt.ylabel('Predicted Viewer Count')
plt.title('Actual vs Predicted Viewer Count (Random Forest)')

# 잔차(residual) 플롯
residuals_rf = y_test - y_pred_rf
plt.subplot(1, 2, 2)
plt.scatter(y_pred_rf, residuals_rf, alpha=0.5)
plt.hlines(y=0, xmin=min(y_pred_rf), xmax=max(y_pred_rf), linestyles='dashed')
plt.xlabel('Predicted Viewer Count')
plt.ylabel('Residuals')
plt.title('Residuals vs Predicted Viewer Count (Random Forest)')

plt.tight_layout()
plt.show()
