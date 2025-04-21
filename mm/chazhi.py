import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 读取Excel文件
data = pd.read_excel(r"D:\Anaconda\Mat\人口数.xlsx")

# 数据验证
if data.empty:
    raise ValueError("读取的数据为空，请检查Excel文件")
if '年份' not in data.columns or '人口(万）' not in data.columns:
    raise ValueError("Excel文件中缺少必要的列：'年份' 或 '人口(万）'")

x = data['年份']
y = data['人口(万）']

# 牛顿插值函数（优化版）
def newton_interpolation(x_data, y_data, xi):
    n = len(x_data)
    # 计算差分表
    diff_table = [y_data.copy()]
    for i in range(1, n):
        diff_table.append([])
        for j in range(n - i):
            diff = (diff_table[i-1][j+1] - diff_table[i-1][j]) / (x_data[j+i] - x_data[j])
            diff_table[i].append(diff)
    
    # 插值计算
    if isinstance(xi, (list, np.ndarray)):
        return np.array([_newton_eval(x_data, diff_table, pt) for pt in xi])
    return _newton_eval(x_data, diff_table, xi)

def _newton_eval(x_data, diff_table, xi):
    result = diff_table[0][0]
    for i in range(1, len(diff_table)):
        term = diff_table[i][0]
        for j in range(i):
            term *= (xi - x_data[j])
        result += term
    return result

# 生成插值结果
xi = np.linspace(x.min(), x.max(), 100)
yi = newton_interpolation(x, y, xi)

# 绘制结果
plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='red', label='原始数据点', zorder=3)
plt.plot(xi, yi, label='牛顿插值', linestyle='--')  # 修正标签
plt.xlabel("年份")
plt.ylabel("人口(万）")
plt.title("人口数据插值结果")
plt.legend()
plt.grid(True)
plt.show()