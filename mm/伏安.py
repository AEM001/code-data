import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

# 中文字体支持（可根据系统选择字体）
plt.rcParams['font.family'] = 'SimHei'  # 黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
plt.rcParams['mathtext.fontset'] = 'stix'  # 解决公式显示问题

# 电压
voltage = np.array([-2, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50])

# 光电流数据（单位：10^-10 A）
current_phi1 = np.array([0.0, 0.3, 1.5, 2.6, 3.1, 3.6, 4.1, 4.4, 4.7, 5.0, 5.3, 5.4, 5.6, 5.7, 5.9, 6.0, 6.1, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8, 6.8, 6.9, 7.0, 7.1])
current_phi2 = np.array([0.0, 1.1, 5.5, 10.4, 13.0, 15.4, 17.7, 19.5, 20.7, 21.7, 22.7, 23.4, 24.2, 25.2, 25.7, 26.2, 26.9, 27.1, 27.6, 28.0, 28.3, 28.9, 29.3, 29.5, 29.8, 30.2, 30.8])
current_phi3 = np.array([0.1, 4.9, 24.0, 42.5, 51.5, 59.3, 66.8, 73.1, 78.4, 83.2, 85.8, 90.0, 92.0, 94.5, 98.3, 99.1, 102.1, 102.7, 104.3, 107.0, 107.8, 110.4, 110.6, 112.6, 114.1, 114.1, 114.6])

# 绘图
plt.figure(figsize=(10, 6))
plt.plot(voltage, current_phi1, 'o-', label='Φ1 = 2mm', markersize=5)
plt.plot(voltage, current_phi2, '^-', label='Φ2 = 4mm', markersize=5)
plt.plot(voltage, current_phi3, 's-', label='Φ3 = 8mm', markersize=5)

# 标注与美化
plt.title('不同光阑孔径下的伏安特性曲线')
plt.xlabel('阳极-阴极电压 $U_{ak}$ (V)')
plt.ylabel('光电流 $I$ ($\\times 10^{-10}$ A)')
plt.grid(True)
plt.legend()

# 设置更细致的横坐标刻度
plt.xticks(np.arange(-2, 51, 2))  # 从-2到50，每隔2个单位显示一个刻度

# 添加细致的纵坐标刻度
plt.yticks(np.arange(0, 120, 10))  # 从0到110，每隔10个单位显示一个刻度

plt.tight_layout()
plt.show()
