import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei']  # Windows系统常用字体
rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 常数
e = 1.602e-19  # 元素电荷，单位：C
h_true = 6.626e-34  # 普朗克常数真实值，单位：J·s

# 实验数据
frequency = np.array([8.214, 7.407, 6.879, 5.490, 5.196])  # 单位：10^14 Hz
frequency_SI = frequency * 1e14  # 转换为 Hz
stopping_voltage = np.array([1.756, 1.519, 1.128, 0.615, 0.504])  # 单位：V

# 线性拟合
coeffs = np.polyfit(frequency_SI, stopping_voltage, 1)
slope, intercept = coeffs
fit_func = np.poly1d(coeffs)

# 计算普朗克常数 h = slope * e
h_exp = slope * e
error = abs(h_exp - h_true) / h_true * 100  # 相对误差（百分比）

# 拟合线数据
x_fit = np.linspace(0, max(frequency_SI) * 1.1, 200)
y_fit = fit_func(x_fit)

# 绘图
plt.figure(figsize=(8, 6))
plt.scatter(frequency_SI, stopping_voltage, color='blue', label='实验数据')
plt.plot(x_fit, y_fit, color='red', label='线性拟合')

# 标注文本：拟合结果、h值、相对误差
textstr = '\n'.join((
    f'拟合斜率 = {slope:.2e} V·s',
    f'h (实验) = {h_exp:.3e} J·s',
    f'h (真实) = {h_true:.3e} J·s',
    f'相对误差 = {error:.2f}%'
))
plt.text(0.05 * max(frequency_SI), max(stopping_voltage)*0.8, textstr,
         fontsize=10, bbox=dict(facecolor='white', edgecolor='gray'))

# 坐标轴和图形设置
plt.title('光电效应截止电压与频率关系图')
plt.xlabel('频率 ν (Hz)')
plt.ylabel('截止电压 (V)')
plt.grid(True)
plt.legend()
plt.xlim(0, max(x_fit))
plt.ylim(0, max(stopping_voltage) + 0.5)
plt.tight_layout()
plt.show()
