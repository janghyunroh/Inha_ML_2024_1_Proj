import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

# 데이터 로드
file_path = r'D:\AllProjectRepo\Inha_ML_2024_1_Proj\BaseBall\baseball_players.csv'
data = pd.read_csv(file_path)

# Feature와 label 분리
features = data[['Height(inches)', 'Weight(pounds)', 'Age']]
labels = data['Position']

# 데이터 분할 (훈련 데이터와 테스트 데이터)
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# 결측치 처리: 평균값으로 대체
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# 데이터 스케일링
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

# SMOTE를 사용한 오버샘플링
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

# SVM 모델 학습 및 평가
svm_model = SVC(kernel='rbf', C = 10, gamma = 0.01)
svm_model.fit(X_train_resampled, y_train_resampled)
y_pred_svm = svm_model.predict(X_test_scaled)
classification_report_svm = classification_report(y_test, y_pred_svm)
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print("SVM Classification Report:\n", classification_report_svm)
print("SVM Accuracy: {:.2f}%".format(accuracy_svm * 100))

# Random Forest 모델 학습 및 평가
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train_resampled, y_train_resampled)
y_pred_rf = rf_model.predict(X_test_scaled)
classification_report_rf = classification_report(y_test, y_pred_rf)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print("Random Forest Classification Report:\n", classification_report_rf)
print("Random Forest Accuracy: {:.2f}%".format(accuracy_rf * 100))

# XGBoost 모델 학습 및 평가 (xgboost 설치 필요)
# import xgboost as xgb
# xgb_model = xgb.XGBClassifier(random_state=42)
# xgb_model.fit(X_train_resampled, y_train_resampled)
# y_pred_xgb = xgb_model.predict(X_test_scaled)
# classification_report_xgb = classification_report(y_test, y_pred_xgb)
# print("XGBoost Classification Report:\n", classification_report_xgb)

# 정확성 시각화
models = ['SVM', 'Random Forest']
accuracies = [accuracy_svm * 100, accuracy_rf * 100]

plt.figure(figsize=(10, 6))
plt.bar(models, accuracies, color=['blue', 'green'])
plt.xlabel('Models')
plt.ylabel('Accuracy (%)')
plt.title('Model Accuracy Comparison')
plt.ylim(0, 100)
for i, v in enumerate(accuracies):
    plt.text(i, v + 1, f"{v:.2f}%", ha='center', va='bottom')
plt.show()
