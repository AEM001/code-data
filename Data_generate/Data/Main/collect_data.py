import pandas as pd

# 定义文件列表
file_paths = [
    'd:/Documents/code&data/Data_generate/Data/Main/model_collapse_full_metrics_corrected.csv',
    'd:/Documents/code&data/Data_generate/Data/Main/model_collapse_metrics_50_percent.csv',
    'd:/Documents/code&data/Data_generate/Data/Main/model_collapse_metrics_90_percent.csv'
]

# 定义指标列
metrics = ['Perplexity', '3gram_Diversity', 'HighFreq_Ratio', 'Entropy', 
           'Rouge-1', 'Rouge-2', 'Rouge-L', 'METEOR']

# 存储所有数据的列表
all_data = []

# 处理每个文件
for file_path in file_paths:
    df = pd.read_csv(file_path)
    # 计算每代三个run的平均值
    avg_data = df.groupby('Generation')[metrics].mean().values
    all_data.extend(avg_data)

# 转换为DataFrame
result_df = pd.DataFrame(all_data, columns=metrics)

# 保存为CSV（纯数据，无索引和列名）
output_path = 'd:/Documents/code&data/Data_generate/Data/Main/model_collapse_metrics_matrix.csv'
result_df.to_csv(output_path, index=False, header=False)

print(f"纯数据矩阵已保存到: {output_path}")
print(f"矩阵形状: {result_df.shape} (行数: {result_df.shape[0]}, 列数: {result_df.shape[1]})")
