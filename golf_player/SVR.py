import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import uniform, randint

# 데이터 로드
data = pd.read_csv(r'D:\AllProjectRepo\Inha_ML_2024_1_Proj\golf_player\golf_players.csv')

# 특성과 목표 변수 설정
X = data[['height(cm)', 'weight(Ibs)']]
y = data['avg_score']

# 데이터 분할: 80%는 training, 20%는 validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

model = SVR(kernel = 'rbf', C=10, gamma = 0.01, epsilon = 0.1)

model.fit(X_train, y_train)
y_pred = model.predict(X_val)

mse = mean_squared_error(y_val, y_pred)
mae = mean_absolute_error(y_val, y_pred)
r2 = r2_score(y_val, y_pred)
print(f"SVR Model Evaluation:")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"R^2 Score: {r2:.2f}")

# 예측 결과 시각화
plt.figure(figsize=(10, 6))
plt.scatter(y_val, y_pred, alpha=0.6)
plt.xlabel('Actual Average Score')
plt.ylabel('Predicted Average Score')
plt.title('SVR Predictions vs Actual')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.show()