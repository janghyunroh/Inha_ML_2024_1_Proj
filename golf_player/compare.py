import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# CSV 파일 읽기
file_path = r'D:\AllProjectRepo\Inha_ML_2024_1_Proj\golf_player\golf_players.csv'
data = pd.read_csv(file_path)

# 데이터 확인
print(data.head())

# 특성과 레이블 분리
X = data[['height(cm)', 'weight(Ibs)', 'age']]
y = data['avg_score']

# 데이터 스케일링
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 데이터 분할 (훈련 세트와 테스트 세트)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 모델 정의
svr_model = SVR()
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
lr_model = LinearRegression()

# 모델 학습 및 평가 (교차 검증)
models = {'SVR': svr_model, 'Random Forest': rf_model, 'Linear Regression': lr_model}
for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    rmse_scores = np.sqrt(-scores)
    print(f"{name} - Cross-Validation RMSE: {rmse_scores.mean():.2f} ± {rmse_scores.std():.2f}")

# 최종 모델 학습 및 테스트 세트 평가
final_model = svr_model
final_model.fit(X_train, y_train)
y_pred = final_model.predict(X_test)
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"SVR - Test RMSE: {test_rmse:.2f}")

# 예측 결과 시각화
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.xlabel('Actual Average Score')
plt.ylabel('Predicted Average Score')
plt.title('SVR Predictions vs Actual')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.show()
