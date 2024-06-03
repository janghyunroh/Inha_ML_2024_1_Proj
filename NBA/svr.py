import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
# 데이터 로드
data = pd.read_csv(r'D:\AllProjectRepo\Inha_ML_2024_1_Proj\NBA\NBA_players_clean.csv')

# 데이터 확인
print(data.head())
print(data.info())
print(data.describe())
print(data.columns)

# 특성과 레이블 분리 (예: 'height', 'weight', 'age'를 특성으로, 'average_points'를 레이블로 가정)
X = data[['Height', 'Wt', 'G', 'Pos']]
y = data['PTS']

# 범주형 변수를 원-핫 인코딩
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['Height', 'Wt', 'G']),
        ('cat', OneHotEncoder(), ['Pos'])
    ])

# 데이터 스케일링 및 원-핫 인코딩
X_preprocessed = preprocessor.fit_transform(X)

# 데이터 분할 (훈련 세트와 테스트 세트)
X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size=0.2, random_state=42)


# SVR 모델 정의 및 하이퍼파라미터 튜닝
svr_model = SVR()
svr_param_grid = {
    'C': [0.1, 1, 10, 100],
    'epsilon': [0.1, 0.2, 0.5, 1],
    'kernel': ['linear', 'rbf', 'poly']
}
svr_grid_search = GridSearchCV(svr_model, svr_param_grid, cv=100, scoring='neg_mean_squared_error')
svr_grid_search.fit(X_train, y_train)
print(f"Best SVR Parameters: {svr_grid_search.best_params_}")
svr_best_model = svr_grid_search.best_estimator_

# 최종 모델 학습 및 예측
svr_best_model.fit(X_train, y_train)
y_pred = svr_best_model.predict(X_test)

# 평가 지표 계산
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"SVR Model Evaluation:")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"R^2 Score: {r2:.2f}")

# 예측 결과 시각화
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.xlabel('Actual Average Points')
plt.ylabel('Predicted Average Points')
plt.title('SVR Predictions vs Actual')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.show()
