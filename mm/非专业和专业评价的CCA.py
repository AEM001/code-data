import pandas as pd
import numpy as np
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import StandardScaler

# Step 1: 读取 Excel 数据
# 假设你的 Excel 文件名为 "data.xlsx"，并且数据在第一个工作表中
file_path = "D:\\Anaconda\\Mat\\tv.xlsx"
data = pd.read_excel(file_path)

# Step 2: 查看表头并分离两组变量
# 假设左边三列是非专业人士组 (X)，右边三列是另一组 (Y)
# 你需要根据实际的列名调整下面的选择
X_columns = data.columns[:3]  # 非专业人士组的列名
Y_columns = data.columns[3:]  # 另一组的列名

# 提取两组数据
data_X = data[X_columns].values
data_Y = data[Y_columns].values

# Step 3: 数据标准化
# 对两组数据分别进行标准化处理
scaler_X = StandardScaler()
scaler_Y = StandardScaler()

data_X = scaler_X.fit_transform(data_X)
data_Y = scaler_Y.fit_transform(data_Y)

# Step 4: 创建CCA模型
# n_components 设置为 min(X.shape[1], Y.shape[1])，即两组变量中指标数量的较小值
cca = CCA(n_components=3)

# Step 5: 拟合模型并转换数据
cca.fit(data_X, data_Y)
X_c, Y_c = cca.transform(data_X, data_Y)

# Step 6: 输出典型相关系数
print("典型相关系数:")
for i in range(X_c.shape[1]):
    corr = np.corrcoef(X_c[:, i], Y_c[:, i])[0, 1]
    print(f"第 {i+1} 对典型变量的相关系数: {corr:.4f}")

# Step 7: 查看典型变量的权重
print("\nX 组变量的典型权重:")
print(cca.x_weights_)

print("\nY 组变量的典型权重:")
print(cca.y_weights_)
# 计算典型载荷 (Canonical Loadings)
# 典型载荷是原始变量与典型变量之间的相关性

# Step 1: 计算 X 组的典型载荷
X_loadings = np.corrcoef(data_X.T, X_c.T)[:data_X.shape[1], data_X.shape[1]:]

# Step 2: 计算 Y 组的典型载荷
Y_loadings = np.corrcoef(data_Y.T, Y_c.T)[:data_Y.shape[1], data_Y.shape[1]:]

# 输出典型载荷
print("\nX 组变量的典型载荷:")
print(X_loadings)

print("\nY 组变量的典型载荷:")
print(Y_loadings)