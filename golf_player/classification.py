import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# CSV 파일 읽기
file_path = r'D:\AllProjectRepo\Inha_ML_2024_1_Proj\golf_player\golf_players.csv'
data = pd.read_csv(file_path)

# 데이터 확인
print(data.head())

# 레이블 변환: avg_score가 70.8 이상이면 1, 미만이면 0
data['label'] = (data['avg_score'] >= 70.8).astype(int)

# 특성과 새로운 레이블 분리
X = data[['height(cm)', 'weight(Ibs)', 'age']]
y = data['label']

# 데이터 스케일링
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 데이터 분할 (훈련 세트와 테스트 세트)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 모델 정의
log_reg_model = LogisticRegression()
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
svm_model = SVC()

# 모델 학습 및 평가 (교차 검증)
models = {'Logistic Regression': log_reg_model, 'Random Forest': rf_model, 'SVM': svm_model}
for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    print(f"{name} - Cross-Validation Accuracy: {scores.mean() * 100:.2f}% ± {scores.std() * 100:.2f}%")

# 최종 모델 학습 및 테스트 세트 평가
final_model = rf_model
final_model.fit(X_train, y_train)
y_pred = final_model.predict(X_test)

# 테스트 세트 평가
test_accuracy = accuracy_score(y_test, y_pred)
print(f"Random Forest - Test Accuracy: {test_accuracy:.2f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# 혼동 행렬 시각화
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted 0', 'Predicted 1'], yticklabels=['Actual 0', 'Actual 1'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()
