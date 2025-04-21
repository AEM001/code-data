import numpy as np
import pandas as pd

# 读取相关系数矩阵
corr_matrix = pd.read_csv('d:/Documents/code&data/Data_generate/Data/Main/model_collapse_correlation_matrix.csv', header=None).values

# 计算特征值和特征向量
eigenvalues, eigenvectors = np.linalg.eig(corr_matrix)

# 按特征值降序排列
idx = eigenvalues.argsort()[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

# 统一特征向量方向（首分量正数）
for i in range(eigenvectors.shape[1]):
    if eigenvectors[0, i] < 0:
        eigenvectors[:, i] *= -1

# 计算贡献率和累计贡献率
p = corr_matrix.shape[0]
variance_ratio = eigenvalues / p
cumulative_variance_ratio = np.cumsum(variance_ratio)

# 创建符合图片要求的表格
metrics = ['$P_m$', '$G_m$', '$H_m$', '$E_m$', '$R_{1,M}$', '$R_{2,M}$', '$R_{L,M}$', '$M_m$']
components = ['$a_1$', '$a_2$', '$a_3$', '$a_4$', '$a_5$', '$a_6$', '$a_7$', '$a_8$']

# 构建特征向量部分
eigenvectors_part = pd.DataFrame(
    eigenvectors,
    index=metrics,
    columns=components
)

# 构建统计量部分
stats_part = pd.DataFrame({
    '特征值': eigenvalues,
    '贡献率': variance_ratio,
    '累计贡献率': cumulative_variance_ratio
}, index=components).T

# 合并两个部分
final_table = pd.concat([eigenvectors_part, stats_part])

# 保存结果（保留6位小数）
final_table.to_csv('d:/Documents/code&data/Data_generate/Data/Main/pca_final_table.csv', float_format='%.6f')

print("主成分分析结果已保存为完整表格:")
print("\n完整表格预览:")
print(final_table)