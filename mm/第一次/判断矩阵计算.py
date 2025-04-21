import numpy as np
import pandas as pd

# 读取CSV文件中的数据
file_path = 'D:\\Code\\Python\\mm\\第一次\\judgment_matrix.csv'
data = pd.read_csv(file_path, header=None)
matrix = data.values

# 计算特征值和特征向量
eigenvalues, eigenvectors = np.linalg.eig(matrix)

# 找到最大特征值
max_eigenvalue = np.max(eigenvalues)

# 计算一致性指标 (CI)
n = matrix.shape[0]  # 矩阵的阶数
CI = (max_eigenvalue - n) / (n - 1)

# 计算一致性比率 (CR)，假设随机一致性指数 RI 为 0.9
RI = 0.9
CR = CI / RI

# 判断一致性
if CR < 0.1:
    print("矩阵具有良好的一致性")
else:
    print("矩阵不一致，CR值为:", CR)

# 归一化处理：使用特征值法计算权重向量
# 找到最大特征值对应的特征向量
max_eigenvalue_index = np.argmax(eigenvalues)
max_eigenvector = eigenvectors[:, max_eigenvalue_index]

# 归一化权重向量
weights = np.abs(max_eigenvector) / np.sum(np.abs(max_eigenvector))

# 输出结果
print("最大特征值:", max_eigenvalue)
print("一致性指标 (CI):", CI)
print("一致性比率 (CR):", CR)
print("权重向量:", weights)