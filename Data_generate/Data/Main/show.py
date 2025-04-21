import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 读取修正后的数据
df = pd.read_csv('d:/Documents/code&data/Data_generate/Data/Main/model_collapse_full_metrics_corrected.csv')

# 设置绘图风格
plt.style.use('seaborn-v0_8')  # 修改为新的seaborn样式名称
sns.set_palette("husl")

# 创建多子图
fig, axes = plt.subplots(3, 3, figsize=(18, 15))
fig.suptitle('Model Collapse Metrics After Correction (Generation 9 Interpolated)', fontsize=16)

# 绘制Perplexity
sns.lineplot(data=df, x='Generation', y='Perplexity', hue='Run', ax=axes[0,0], marker='o')
axes[0,0].set_title('Perplexity')
axes[0,0].axvline(x=9, color='r', linestyle='--', alpha=0.3)  # 标记第9代

# 绘制3gram_Diversity
sns.lineplot(data=df, x='Generation', y='3gram_Diversity', hue='Run', ax=axes[0,1], marker='o')
axes[0,1].set_title('3gram Diversity')
axes[0,1].axvline(x=9, color='r', linestyle='--', alpha=0.3)

# 绘制HighFreq_Ratio
sns.lineplot(data=df, x='Generation', y='HighFreq_Ratio', hue='Run', ax=axes[0,2], marker='o')
axes[0,2].set_title('High Frequency Ratio')
axes[0,2].axvline(x=9, color='r', linestyle='--', alpha=0.3)

# 绘制Entropy
sns.lineplot(data=df, x='Generation', y='Entropy', hue='Run', ax=axes[1,0], marker='o')
axes[1,0].set_title('Entropy')
axes[1,0].axvline(x=9, color='r', linestyle='--', alpha=0.3)

# 绘制Rouge-1
sns.lineplot(data=df, x='Generation', y='Rouge-1', hue='Run', ax=axes[1,1], marker='o')
axes[1,1].set_title('Rouge-1')
axes[1,1].axvline(x=9, color='r', linestyle='--', alpha=0.3)

# 绘制Rouge-2
sns.lineplot(data=df, x='Generation', y='Rouge-2', hue='Run', ax=axes[1,2], marker='o')
axes[1,2].set_title('Rouge-2')
axes[1,2].axvline(x=9, color='r', linestyle='--', alpha=0.3)

# 绘制Rouge-L
sns.lineplot(data=df, x='Generation', y='Rouge-L', hue='Run', ax=axes[2,0], marker='o')
axes[2,0].set_title('Rouge-L')
axes[2,0].axvline(x=9, color='r', linestyle='--', alpha=0.3)

# 绘制METEOR
sns.lineplot(data=df, x='Generation', y='METEOR', hue='Run', ax=axes[2,1], marker='o')
axes[2,1].set_title('METEOR')
axes[2,1].axvline(x=9, color='r', linestyle='--', alpha=0.3)

# 隐藏最后一个空子图
axes[2,2].axis('off')

# 调整布局
plt.tight_layout()
plt.subplots_adjust(top=0.92)

# 保存图像
plt.savefig('d:/Documents/code&data/Data_generate/Data/Main/model_collapse_metrics_after_correction.png', dpi=300, bbox_inches='tight')
plt.show()
