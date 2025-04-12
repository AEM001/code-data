import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 设置工作目录和文件路径
data_dir = r"d:\Documents\100\action"
output_dir = r"d:\Documents\100\action\results"
os.makedirs(output_dir, exist_ok=True)

# 读取数据
df = pd.read_csv(os.path.join(data_dir, "model_collapse_full_metrics_corrected.csv"))

# 检查数据
print("数据概览:")
print(df.head())
print("\n数据信息:")
print(df.info())
print("\n数据统计描述:")
print(df.describe())

# 检查缺失值
print("\n缺失值检查:")
print(df.isnull().sum())

# 重命名列以便于SEM模型使用
df = df.rename(columns={
    '3gram_Diversity': 'Diversity',
    'HighFreq_Ratio': 'HighFreq',
    'Rouge-1': 'Rouge_1',
    'Rouge-2': 'Rouge_2',
    'Rouge-L': 'Rouge_L'
})

# 创建滞后变量(用于分析代际间影响)
df['Perplexity_lag1'] = df.groupby('Run')['Perplexity'].shift(1)
df['Diversity_lag1'] = df.groupby('Run')['Diversity'].shift(1)
df['HighFreq_lag1'] = df.groupby('Run')['HighFreq'].shift(1)
df['METEOR_lag1'] = df.groupby('Run')['METEOR'].shift(1)

# 计算指标变化率
df['Perplexity_change'] = df.groupby('Run')['Perplexity'].pct_change()
df['Diversity_change'] = df.groupby('Run')['Diversity'].pct_change()
df['HighFreq_change'] = df.groupby('Run')['HighFreq'].pct_change()
df['METEOR_change'] = df.groupby('Run')['METEOR'].pct_change()

# 添加非线性项
df['HighFreq_sq'] = df['HighFreq']**2
df['Diversity_sq'] = df['Diversity']**2

# 数据标准化
features_to_scale = ['Perplexity', 'Diversity', 'HighFreq', 'Entropy', 
                     'Rouge_1', 'Rouge_2', 'Rouge_L', 'METEOR']

scaler = StandardScaler()
scaled_data = scaler.fit_transform(df[features_to_scale])
scaled_df = pd.DataFrame(scaled_data, columns=[f"{col}_scaled" for col in features_to_scale])

# 将标准化后的数据合并回原始数据框
df = pd.concat([df.reset_index(drop=True), scaled_df.reset_index(drop=True)], axis=1)

# 创建崩溃指标
df['is_collapsed'] = df['METEOR'] < 0.4
df['next_gen_collapse'] = df.groupby('Run')['is_collapsed'].shift(-1).fillna(False)

# 保存预处理后的数据
df.to_csv(os.path.join(output_dir, "preprocessed_data.csv"), index=False)
print(f"\n预处理后的数据已保存至 {os.path.join(output_dir, 'preprocessed_data.csv')}")

# 数据可视化
plt.figure(figsize=(12, 10))

# 1. 困惑度与多样性的关系
plt.subplot(2, 2, 1)
sns.scatterplot(data=df, x='Perplexity', y='Diversity', hue='Generation', palette='viridis')
plt.title('困惑度与多样性的关系')
plt.xlabel('困惑度 (Perplexity)')
plt.ylabel('多样性 (3gram Diversity)')

# 2. 高频词比例与语义质量的关系
plt.subplot(2, 2, 2)
sns.scatterplot(data=df, x='HighFreq', y='METEOR', hue='Generation', palette='viridis')
plt.title('高频词比例与语义质量的关系')
plt.xlabel('高频词比例 (HighFreq Ratio)')
plt.ylabel('METEOR 分数')

# 3. 多样性与语义质量的关系
plt.subplot(2, 2, 3)
sns.scatterplot(data=df, x='Diversity', y='METEOR', hue='Generation', palette='viridis')
plt.title('多样性与语义质量的关系')
plt.xlabel('多样性 (3gram Diversity)')
plt.ylabel('METEOR 分数')

# 4. 困惑度随代数的变化
plt.subplot(2, 2, 4)
sns.lineplot(data=df, x='Generation', y='Perplexity', hue='Run')
plt.title('困惑度随代数的变化')
plt.xlabel('代数 (Generation)')
plt.ylabel('困惑度 (Perplexity)')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "data_exploration.png"), dpi=300)
plt.close()

# 相关性分析
correlation_matrix = df[features_to_scale].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('指标间相关性矩阵')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "correlation_matrix.png"), dpi=300)
plt.close()

print("数据可视化已完成，图表已保存至结果目录")

# 创建SEM分析所需的数据子集
sem_data = df.dropna(subset=['Perplexity_scaled', 'Diversity_scaled', 'HighFreq_scaled', 
                             'Entropy_scaled', 'Rouge_1_scaled', 'Rouge_2_scaled', 
                             'Rouge_L_scaled', 'METEOR_scaled'])

# 保存SEM分析数据
sem_data.to_csv(os.path.join(output_dir, "sem_data.csv"), index=False)
print(f"SEM分析数据已保存至 {os.path.join(output_dir, 'sem_data.csv')}")