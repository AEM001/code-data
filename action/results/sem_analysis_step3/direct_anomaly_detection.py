import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# 设置工作目录和文件路径
data_dir = r"d:\Documents\100\action"
# 设置输出目录
output_dir = r"d:\Documents\100\action\results\sem_analysis_step3"
os.makedirs(output_dir, exist_ok=True)

print("开始执行直接异常检测分析...")

# 设置matplotlib使用中文字体
try:
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    print("已设置matplotlib使用中文字体")
except Exception as e:
    print(f"设置中文字体时出错: {e}")

# 读取预处理后的数据
try:
    # 尝试从sem_analysis_step1目录读取
    df = pd.read_csv(os.path.join(data_dir, "results", "sem_analysis_step1", "sem_data.csv"))
    print(f"成功从sem_analysis_step1读取数据，共 {len(df)} 行")
except FileNotFoundError:
    try:
        # 如果找不到，尝试从results目录读取
        df = pd.read_csv(os.path.join(data_dir, "results", "sem_data.csv"))
        print(f"成功从results读取数据，共 {len(df)} 行")
    except FileNotFoundError:
        print("错误: 未找到sem_data.csv文件")
        exit(1)

# 方法1: 使用隔离森林进行异常检测
print("\n方法1: 使用隔离森林进行异常检测...")

# 选择关键特征
features = ['Perplexity', 'Diversity', 'HighFreq', 'Entropy', 
            'Rouge_1', 'Rouge_2', 'Rouge_L', 'METEOR']

# 确保所有特征都存在
for feature in features:
    if feature not in df.columns:
        print(f"警告: 数据中缺少特征 {feature}")
        features.remove(feature)

# 标准化特征（如果数据中没有标准化版本）
if 'Perplexity_scaled' not in df.columns:
    print("对原始特征进行标准化...")
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(
        scaler.fit_transform(df[features]),
        columns=[f"{col}_scaled" for col in features]
    )
    for col in df_scaled.columns:
        df[col] = df_scaled[col].values
    scaled_features = [f"{col}_scaled" for col in features]
else:
    print("使用已有的标准化特征...")
    scaled_features = ['Perplexity_scaled', 'Diversity_scaled', 'HighFreq_scaled', 'Entropy_scaled',
                      'Rouge_1_scaled', 'Rouge_2_scaled', 'Rouge_L_scaled', 'METEOR_scaled']

# 训练隔离森林模型
contamination = 0.1  # 假设10%的样本是异常的
iso_forest = IsolationForest(contamination=contamination, random_state=42)
df['anomaly_score'] = iso_forest.fit_predict(df[scaled_features])
df['is_anomaly'] = df['anomaly_score'] == -1

# 统计异常样本数量
anomaly_count = df['is_anomaly'].sum()
print(f"检测到 {anomaly_count} 个异常样本 (占比 {anomaly_count/len(df)*100:.2f}%)")

# 方法2: 基于指标变化率的异常检测
print("\n方法2: 基于指标变化率的异常检测...")

# 计算关键指标的变化率
df['Diversity_change'] = df.groupby('Run')['Diversity'].pct_change()
df['HighFreq_change'] = df.groupby('Run')['HighFreq'].pct_change()
df['METEOR_change'] = df.groupby('Run')['METEOR'].pct_change()

# 使用3个标准差作为阈值（比2个标准差更严格）
diversity_change_threshold = 3 * df['Diversity_change'].std()
highfreq_change_threshold = 3 * df['HighFreq_change'].std()
meteor_change_threshold = 3 * df['METEOR_change'].std()

# 标记异常变化
df['diversity_change_anomaly'] = abs(df['Diversity_change']) > diversity_change_threshold
df['highfreq_change_anomaly'] = abs(df['HighFreq_change']) > highfreq_change_threshold
df['meteor_change_anomaly'] = abs(df['METEOR_change']) > meteor_change_threshold

# 综合异常标记
df['change_anomaly'] = (df['diversity_change_anomaly'] | 
                        df['highfreq_change_anomaly'] | 
                        df['meteor_change_anomaly'])

# 统计异常样本数量
change_anomaly_count = df['change_anomaly'].sum()
print(f"检测到 {change_anomaly_count} 个变化率异常样本 (占比 {change_anomaly_count/len(df)*100:.2f}%)")

# 方法3: 基于指标间关系的异常检测
print("\n方法3: 基于指标间关系的异常检测...")

# 计算高频词比例与多样性的预期关系
from sklearn.linear_model import LinearRegression

# 拟合高频词比例与多样性的关系
reg_diversity = LinearRegression()
reg_diversity.fit(df[['HighFreq_scaled']], df['Diversity_scaled'])
expected_diversity = reg_diversity.predict(df[['HighFreq_scaled']])
df['diversity_relation_residual'] = df['Diversity_scaled'] - expected_diversity

# 拟合多样性与METEOR的关系
reg_meteor = LinearRegression()
reg_meteor.fit(df[['Diversity_scaled']], df['METEOR_scaled'])
expected_meteor = reg_meteor.predict(df[['Diversity_scaled']])
df['meteor_relation_residual'] = df['METEOR_scaled'] - expected_meteor

# 使用2.5个标准差作为阈值
diversity_relation_threshold = 2.5 * df['diversity_relation_residual'].std()
meteor_relation_threshold = 2.5 * df['meteor_relation_residual'].std()

# 标记关系异常
df['diversity_relation_anomaly'] = abs(df['diversity_relation_residual']) > diversity_relation_threshold
df['meteor_relation_anomaly'] = abs(df['meteor_relation_residual']) > meteor_relation_threshold
df['relation_anomaly'] = df['diversity_relation_anomaly'] | df['meteor_relation_anomaly']

# 统计异常样本数量
relation_anomaly_count = df['relation_anomaly'].sum()
print(f"检测到 {relation_anomaly_count} 个关系异常样本 (占比 {relation_anomaly_count/len(df)*100:.2f}%)")

# 综合所有异常检测方法
df['combined_anomaly'] = df['is_anomaly'] | df['change_anomaly'] | df['relation_anomaly']
combined_anomaly_count = df['combined_anomaly'].sum()
print(f"\n综合检测到 {combined_anomaly_count} 个异常样本 (占比 {combined_anomaly_count/len(df)*100:.2f}%)")

# 保存带有异常标记的数据
df.to_csv(os.path.join(output_dir, "direct_anomaly_detection_results.csv"), index=False)
print(f"异常检测结果已保存至 {os.path.join(output_dir, 'direct_anomaly_detection_results.csv')}")

# 可视化异常样本
print("\n生成异常检测可视化图...")

# 图1: 高频词比例与多样性的关系，标记异常点
plt.figure(figsize=(10, 6))
plt.scatter(df.loc[~df['combined_anomaly'], 'HighFreq_scaled'], 
            df.loc[~df['combined_anomaly'], 'Diversity_scaled'], 
            c='blue', alpha=0.6, label='正常样本')
plt.scatter(df.loc[df['combined_anomaly'], 'HighFreq_scaled'], 
            df.loc[df['combined_anomaly'], 'Diversity_scaled'], 
            c='red', s=50, label='异常样本')
plt.xlabel('高频词比例 (标准化)')
plt.ylabel('多样性 (标准化)')
plt.title('异常检测: 高频词比例与多样性')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig(os.path.join(output_dir, "direct_anomaly_diversity.png"), dpi=300)
plt.close()

# 图2: 多样性与语义质量的关系，标记异常点
plt.figure(figsize=(10, 6))
plt.scatter(df.loc[~df['combined_anomaly'], 'Diversity_scaled'], 
            df.loc[~df['combined_anomaly'], 'METEOR_scaled'], 
            c='blue', alpha=0.6, label='正常样本')
plt.scatter(df.loc[df['combined_anomaly'], 'Diversity_scaled'], 
            df.loc[df['combined_anomaly'], 'METEOR_scaled'], 
            c='red', s=50, label='异常样本')
plt.xlabel('多样性 (标准化)')
plt.ylabel('METEOR得分 (标准化)')
plt.title('异常检测: 多样性与语义质量')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig(os.path.join(output_dir, "direct_anomaly_quality.png"), dpi=300)
plt.close()

# 图3: 按Run分组的异常比例
run_anomaly = df.groupby('Run')['combined_anomaly'].mean().reset_index()
run_anomaly.columns = ['Run', 'Anomaly_Rate']
run_anomaly = run_anomaly.sort_values('Anomaly_Rate', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Run', y='Anomaly_Rate', data=run_anomaly)
plt.xlabel('Run')
plt.ylabel('异常比例')
plt.title('各Run的异常样本比例')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "direct_run_anomaly_rate.png"), dpi=300)
plt.close()

# 图4: 异常样本在生成轮次上的分布
plt.figure(figsize=(10, 6))
sns.countplot(x='Generation', hue='combined_anomaly', data=df)
plt.xlabel('生成轮次')
plt.ylabel('样本数量')
plt.title('异常样本在生成轮次上的分布')
plt.legend(['正常样本', '异常样本'])
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "anomaly_by_generation.png"), dpi=300)
plt.close()

# 分析异常样本特征
print("\n分析异常样本特征...")
anomaly_df = df[df['combined_anomaly']]
normal_df = df[~df['combined_anomaly']]

if not anomaly_df.empty:
    # 计算异常样本和正常样本的关键指标均值
    anomaly_stats = anomaly_df[['Perplexity', 'Diversity', 'HighFreq', 'METEOR']].mean()
    normal_stats = normal_df[['Perplexity', 'Diversity', 'HighFreq', 'METEOR']].mean()
    stats_comparison = pd.DataFrame({'异常样本均值': anomaly_stats, '正常样本均值': normal_stats})
    stats_comparison['差异百分比'] = (anomaly_stats - normal_stats) / normal_stats * 100

    print("\n异常样本与正常样本的指标对比:")
    print(stats_comparison)

    # 保存统计结果
    stats_comparison.to_csv(os.path.join(output_dir, "direct_anomaly_stats_comparison.csv"))
    print(f"异常样本统计结果已保存至 {os.path.join(output_dir, 'direct_anomaly_stats_comparison.csv')}")
else:
    print("警告: 没有检测到异常样本，无法进行特征分析")

# 提取预警规则
print("\n提取预警规则...")

# 基于异常样本特征提取预警规则
if not anomaly_df.empty:
    # 计算异常样本的特征分布
    anomaly_percentiles = {
        'HighFreq': {
            'min': anomaly_df['HighFreq'].min(),
            'p25': anomaly_df['HighFreq'].quantile(0.25),
            'median': anomaly_df['HighFreq'].median(),
            'p75': anomaly_df['HighFreq'].quantile(0.75),
            'max': anomaly_df['HighFreq'].max()
        },
        'Diversity': {
            'min': anomaly_df['Diversity'].min(),
            'p25': anomaly_df['Diversity'].quantile(0.25),
            'median': anomaly_df['Diversity'].median(),
            'p75': anomaly_df['Diversity'].quantile(0.75),
            'max': anomaly_df['Diversity'].max()
        },
        'METEOR': {
            'min': anomaly_df['METEOR'].min(),
            'p25': anomaly_df['METEOR'].quantile(0.25),
            'median': anomaly_df['METEOR'].median(),
            'p75': anomaly_df['METEOR'].quantile(0.75),
            'max': anomaly_df['METEOR'].max()
        }
    }
    
    # 计算变化率阈值
    change_thresholds = {
        'Diversity_change': diversity_change_threshold,
        'HighFreq_change': highfreq_change_threshold,
        'METEOR_change': meteor_change_threshold
    }
    
    # 生成预警规则
    warning_rules = {
        '规则1': f"高频词比例 > {anomaly_percentiles['HighFreq']['p75']:.4f}",
        '规则2': f"多样性 < {anomaly_percentiles['Diversity']['p25']:.2f}",
        '规则3': f"METEOR得分 < {anomaly_percentiles['METEOR']['p25']:.4f}",
        '规则4': f"多样性变化率 < -{diversity_change_threshold:.4f} 或 > {diversity_change_threshold:.4f}",
        '规则5': f"高频词比例变化率 < -{highfreq_change_threshold:.4f} 或 > {highfreq_change_threshold:.4f}",
        '规则6': f"高频词比例与多样性关系异常 (残差 > {diversity_relation_threshold:.4f})"
    }
    
    # 保存预警规则
    with open(os.path.join(output_dir, "warning_rules.txt"), 'w', encoding='utf-8') as f:
        f.write("# 模型崩溃预警规则\n\n")
        f.write("以下规则基于异常检测结果提取，可用于早期预警模型崩溃:\n\n")
        for rule_name, rule_desc in warning_rules.items():
            f.write(f"## {rule_name}\n")
            f.write(f"{rule_desc}\n\n")
        
        f.write("## 综合预警\n")
        f.write("当满足以上任意两条规则时，系统应发出预警。\n\n")
        
        f.write("## 异常样本特征统计\n")
        f.write(stats_comparison.to_string())
    
    print(f"预警规则已保存至 {os.path.join(output_dir, 'warning_rules.txt')}")
else:
    print("警告: 没有检测到异常样本，无法提取预警规则")

print("\n直接异常检测分析完成!")