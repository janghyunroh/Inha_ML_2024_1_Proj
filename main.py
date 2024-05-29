import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score

# 데이터 로드
data = pd.read_csv('golf_players.csv')

# 데이터 확인
print(data.head())

# 특성과 목표 변수 설정
X = data[['height(cm)', 'weight(Ibs)', 'age']]
y = data['avg_score']

# 데이터 분할: 80%는 training, 20%는 validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 데이터 스케일링
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_val_scaled = scaler_X.transform(X_val)

y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()
y_val_scaled = scaler_y.transform(y_val.values.reshape(-1, 1)).flatten()

# SVR 모델 학습
svr = SVR(kernel='rbf')
svr.fit(X_train_scaled, y_train_scaled)

# 예측
y_train_pred = svr.predict(X_train_scaled)
y_val_pred = svr.predict(X_val_scaled)

# 스케일링 복원
y_train_pred_rescaled = scaler_y.inverse_transform(y_train_pred.reshape(-1, 1)).flatten()
y_val_pred_rescaled = scaler_y.inverse_transform(y_val_pred.reshape(-1, 1)).flatten()

# 평가
mse_train = mean_squared_error(y_train, y_train_pred_rescaled)
r2_train = r2_score(y_train, y_train_pred_rescaled)

mse_val = mean_squared_error(y_val, y_val_pred_rescaled)
r2_val = r2_score(y_val, y_val_pred_rescaled)

print(f'Training MSE: {mse_train:.3f}, R2: {r2_train:.3f}')
print(f'Validation MSE: {mse_val:.3f}, R2: {r2_val:.3f}')
