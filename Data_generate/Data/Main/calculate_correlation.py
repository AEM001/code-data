import pandas as pd
import numpy as np

# 读取标准化后的数据
input_path = 'd:/Documents/code&data/Data_generate/Data/Main/model_collapse_metrics_normalized.csv'
output_path = 'd:/Documents/code&data/Data_generate/Data/Main/model_collapse_correlation_matrix.csv'

# 读取数据（无表头）
data = pd.read_csv(input_path, header=None)

# 计算相关系数矩阵
correlation_matrix = data.corr()

# 保存相关系数矩阵
correlation_matrix.to_csv(output_path)

print("相关系数矩阵计算完成，已保存到:", output_path)
print("\n相关系数矩阵预览:")
print(correlation_matrix)


import seaborn as sns
import matplotlib.pyplot as plt

# 可视化相关系数矩阵
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Metrics Correlation Matrix')
plt.savefig('d:/Documents/code&data/Data_generate/Data/Main/correlation_matrix_heatmap.png')
plt.show()