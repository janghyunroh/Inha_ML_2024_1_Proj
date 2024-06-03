import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# CSV 파일 읽기
file_path = r'D:\AllProjectRepo\Inha_ML_2024_1_Proj\golf_player\golf_players.csv'
data = pd.read_csv(file_path)

# 데이터 확인 (첫 몇 줄 출력)
print(data.head())

# 3D 산점도 그리기
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(data['height(cm)'], data['weight(Ibs)'], c=data['avg_score'], cmap='viridis')

# 축 라벨 설정
ax.set_xlabel('Height (cm)')
ax.set_ylabel('Weight (Ibs)')
plt.title('3D Scatter Plot of Golf Players Data')

# 컬러바 추가
cbar = plt.colorbar(sc)
cbar.set_label('Average Score')

plt.show()
