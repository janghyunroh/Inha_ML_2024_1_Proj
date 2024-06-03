import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import uniform, randint

# 데이터 로드
data = pd.read_csv(r'D:\AllProjectRepo\Inha_ML_2024_1_Proj\golf_player\golf_players.csv')

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

# SVR 모델
svr = SVR()

# 하이퍼파라미터 그리드 설정
param_distributions = {
    'C': uniform(0.1, 10),
    'epsilon': uniform(0.01, 1),
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'degree': randint(2, 5)  # 'poly' 커널 사용 시만 적용됩니다.
}

# RandomizedSearchCV 설정
random_search = RandomizedSearchCV(estimator=svr, param_distributions=param_distributions,
                                   n_iter=100, cv=3, verbose=2, random_state=42, n_jobs=-1)

# RandomizedSearchCV 학습
random_search.fit(X_train_scaled, y_train_scaled)

# 최적의 하이퍼파라미터 출력
print(f'Best hyperparameters: {random_search.best_params_}')

# 최적의 모델로 예측
best_svr = random_search.best_estimator_
y_train_pred = best_svr.predict(X_train_scaled)
y_val_pred = best_svr.predict(X_val_scaled)

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

# 모델 성능 시각화
import matplotlib.pyplot as plt

# 예측 결과 시각화
plt.figure(figsize=(14, 7))

# 실제 값과 예측 값 비교
plt.subplot(1, 2, 1)
plt.scatter(y_val, y_val_pred_rescaled, alpha=0.5)
plt.xlabel('Actual Average Score')
plt.ylabel('Predicted Average Score')
plt.title('Actual vs Predicted Average Score (Best SVR)')

plt.tight_layout()
plt.show()

