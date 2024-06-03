import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# 데이터 로드
file_path = r'D:\AllProjectRepo\Inha_ML_2024_1_Proj\NBA\players_stats.csv'  # 데이터셋 경로
data = pd.read_csv(file_path)

# 데이터 살펴보기
print(data.head())

# Feature와 label 분리
features = data[['PTS', 'REB', 'AST', 'STL', 'BLK']]  # 예시 feature (전 시즌 성적)
labels = data['next_season_PTS']  # 예측하려는 label (다음 시즌 성적)

# 결측치가 있는 행 제거
features = features.dropna()
labels = labels[features.index]

# 데이터 분할 (훈련 데이터와 테스트 데이터)
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# 데이터 스케일링
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# SVR 모델 학습
svr_model = SVR(kernel='rbf', C=100, gamma=0.1)
svr_model.fit(X_train_scaled, y_train)

# 예측 및 평가
y_pred = svr_model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
print("Root Mean Squared Error: {:.2f}".format(rmse))

# 결과 시각화
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Points')
plt.ylabel('Predicted Points')
plt.title('Actual vs Predicted Points')
plt.show()
