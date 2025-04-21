import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# 读取异常样本与正常样本的统计比较数据
stats_df = pd.read_csv('d:/Code/contest/action/results/sem_analysis_step3/direct_anomaly_stats_comparison.csv')

# 创建图3：异常样本与正常样本的指标对比
plt.figure(figsize=(10, 6))
indicators = stats_df.iloc[0:4, 0].tolist()
anomaly_means = stats_df.iloc[0:4, 1].astype(float).tolist()
normal_means = stats_df.iloc[0:4, 2].astype(float).tolist()
diff_percent = stats_df.iloc[0:4, 3].astype(float).tolist()

x = np.arange(len(indicators))
width = 0.35

fig, ax1 = plt.subplots(figsize=(12, 7))
rects1 = ax1.bar(x - width/2, anomaly_means, width, label='异常样本均值')
rects2 = ax1.bar(x + width/2, normal_means, width, label='正常样本均值')

ax1.set_ylabel('指标均值')
ax1.set_title('异常样本与正常样本的关键指标对比')
ax1.set_xticks(x)
ax1.set_xticklabels(indicators)
ax1.legend(loc='upper left')

ax2 = ax1.twinx()
ax2.plot(x, diff_percent, 'r-o', label='差异百分比(%)')
ax2.set_ylabel('差异百分比(%)')
ax2.legend(loc='upper right')

fig.tight_layout()
plt.savefig('d:/Code/contest/action/results/sem_analysis_step3/indicator_comparison.png', dpi=300, bbox_inches='tight')
plt.close()