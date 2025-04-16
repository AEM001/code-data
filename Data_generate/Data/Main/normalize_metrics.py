import pandas as pd
from sklearn.preprocessing import StandardScaler

# 读取原始数据矩阵
input_path = 'd:/Documents/code&data/Data_generate/Data/Main/model_collapse_metrics_matrix.csv'
output_path = 'd:/Documents/code&data/Data_generate/Data/Main/model_collapse_metrics_normalized.csv'

# 读取数据（无表头）
data = pd.read_csv(input_path, header=None)

# 初始化标准化器
scaler = StandardScaler()

# 对数据进行Z-score标准化
normalized_data = scaler.fit_transform(data)

# 将结果保存为新的CSV文件
pd.DataFrame(normalized_data).to_csv(output_path, index=False, header=False)

print(f"标准化处理完成，结果已保存到: {output_path}")
