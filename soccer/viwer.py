import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 데이터 불러오기
file_path = r'D:\AllProjectRepo\Inha_ML_2024_1_Proj\soccer\bulidata.csv'
data = pd.read_csv(file_path)

# NaN 데이터 제거
data.dropna(subset=['LEAGUE_NAME', 'LOCATION', 'HOME_TEAM', 'AWAY_TEAM', 'VIEWER'], inplace=True)

# 필요한 열만 선택
data = data[['LEAGUE_NAME', 'LOCATION', 'HOME_TEAM', 'AWAY_TEAM', 'VIEWER']]

# 특징과 레이블 분리
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

# SVR 모델 학습
svr = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
svr.fit(X_train_scaled, y_train)

# 예측
y_pred = svr.predict(X_test_scaled)

# 평가
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

# 평가 지표 계산
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'RMSE: {rmse}')
print(f'MAE: {mae}')
print(f'R^2 Score: {r2}')

# 예측 결과 시각화
plt.figure(figsize=(14, 7))

# 실제 값과 예측 값 비교
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel('Actual Viewer Count')
plt.ylabel('Predicted Viewer Count')
plt.title('Actual vs Predicted Viewer Count')

# 잔차(residual) 플롯
residuals = y_test - y_pred
plt.subplot(1, 2, 2)
plt.scatter(y_pred, residuals, alpha=0.5)
plt.hlines(y=0, xmin=min(y_pred), xmax=max(y_pred), linestyles='dashed')
plt.xlabel('Predicted Viewer Count')
plt.ylabel('Residuals')
plt.title('Residuals vs Predicted Viewer Count')

plt.tight_layout()
plt.show()
