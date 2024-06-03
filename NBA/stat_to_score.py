import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

file_path = r'D:\AllProjectRepo\Inha_ML_2024_1_Proj\NBA\NBA_players_clean.csv'

data = pd.read_csv(file_path)

X = data[['Height', 'Wt', 'G', 'Pos']]  # feature
Y = data['PTS'] #label

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['Height', 'Wt', 'G']),
        ('cat', OneHotEncoder(), ['Pos'])
    ])

X_preprocessor = preprocessor.fit_transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(X_preprocessor, Y, test_size=0.2, random_state=42 )

model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])

# 조기 종료 콜백 정의
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# 모델 학습
history = model.fit(X_train, Y_train, epochs=100, validation_split=0.2, batch_size=32, callbacks=[early_stopping])

# 모델 평가
loss, mae = model.evaluate(X_test, Y_test)
print(f'Mean Absolute Error (MAE): {mae:.2f}')

# 예측 결과 시각화
y_pred = model.predict(X_test)

plt.figure(figsize=(10, 6))
plt.scatter(Y_test, y_pred, alpha=0.6)
plt.xlabel('Actual Points')
plt.ylabel('Predicted Points')
plt.title('Neural Network Predictions vs Actual')
plt.plot([Y.min(), Y.max()], [Y.min(), Y.max()], 'k--', lw=2)
plt.show()