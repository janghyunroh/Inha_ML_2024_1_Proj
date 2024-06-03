import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import randint

# 데이터 로드
path = r'D:\AllProjectRepo\Inha_ML_2024_1_Proj\golf_player\golf_players.csv'
data = pd.read_csv(path)

# 특성과 목표 변수 설정
X = data[['height(cm)', 'weight(Ibs)', 'age']]
y = data['avg_score']

# 데이터 분할: 80%는 training, 20%는 validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 데이터 스케일링
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_val_scaled = scaler_X.transform(X_val)

# 랜덤 포레스트 모델
rf = RandomForestRegressor(random_state=42)

# 하이퍼파라미터 그리드 설정
param_distributions = {
    'n_estimators': randint(100, 500),
    'max_depth': randint(3, 20),
    'min_samples_split': randint(2, 10),
    'min_samples_leaf': randint(1, 10),
    'max_features': ['sqrt', 'log2', 0.2, 0.4, 0.6, 0.8],
    'bootstrap': [True, False]
}

# RandomizedSearchCV 설정
random_search = RandomizedSearchCV(estimator=rf, param_distributions=param_distributions,
                                   n_iter=100, cv=3, verbose=2, random_state=42, n_jobs=-1,
                                   error_score='raise')

try:
    # RandomizedSearchCV 학습
    random_search.fit(X_train_scaled, y_train)
except Exception as e:
    print(f"Error during fit: {e}")

# 최적의 하이퍼파라미터 출력
print(f'Best hyperparameters: {random_search.best_params_}')

# 최적의 모델로 예측
best_rf = random_search.best_estimator_
y_train_pred = best_rf.predict(X_train_scaled)
y_val_pred = best_rf.predict(X_val_scaled)

# 평가
mse_train = mean_squared_error(y_train, y_train_pred)
r2_train = r2_score(y_train, y_train_pred)

mse_val = mean_squared_error(y_val, y_val_pred)
r2_val = r2_score(y_val, y_val_pred)

print(f'Training MSE: {mse_train:.3f}, R2: {r2_train:.3f}')
print(f'Validation MSE: {mse_val:.3f}, R2: {r2_val:.3f}')
