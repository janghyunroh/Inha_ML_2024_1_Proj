import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV
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

# 하이퍼파라미터 그리드 설정
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# 랜덤 포레스트 모델 및 Grid Search 초기화
rf = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, 
                           cv=3, n_jobs=-1, verbose=2, scoring='neg_mean_squared_error')

# Grid Search 수행
grid_search.fit(X_train_scaled, y_train)

# 최적 하이퍼파라미터 출력
print(f'Best parameters found: {grid_search.best_params_}')

# 최적 하이퍼파라미터로 모델 학습
best_rf = grid_search.best_estimator_
best_rf.fit(X_train_scaled, y_train)

# 예측
y_pred_best_rf = best_rf.predict(X_test_scaled)

# 평가 지표 계산
mse_best_rf = mean_squared_error(y_test, y_pred_best_rf)
rmse_best_rf = np.sqrt(mse_best_rf)
mae_best_rf = mean_absolute_error(y_test, y_pred_best_rf)
r2_best_rf = r2_score(y_test, y_pred_best_rf)

print(f'Best Random Forest RMSE: {rmse_best_rf}')
print(f'Best Random Forest MAE: {mae_best_rf}')
print(f'Best Random Forest R^2 Score: {r2_best_rf}')

# 예측 결과 시각화
plt.figure(figsize=(14, 7))

# 실제 값과 예측 값 비교
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred_best_rf, alpha=0.5)
plt.xlabel('Actual Viewer Count')
plt.ylabel('Predicted Viewer Count')
plt.title('Actual vs Predicted Viewer Count (Best Random Forest)')

# 잔차(residual) 플롯
residuals_best_rf = y_test - y_pred_best_rf
plt.subplot(1, 2, 2)
plt.scatter(y_pred_best_rf, residuals_best_rf, alpha=0.5)
plt.hlines(y=0, xmin=min(y_pred_best_rf), xmax=max(y_pred_best_rf), linestyles='dashed')
plt.xlabel('Predicted Viewer Count')
plt.ylabel('Residuals')
plt.title('Residuals vs Predicted Viewer Count (Best Random Forest)')

plt.tight_layout()
plt.show()
