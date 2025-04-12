import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from semopy import Model, semplot
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error

# 设置工作目录和文件路径
data_dir = r"d:\Documents\100\action"
output_dir = r"d:\Documents\100\action\results"
os.makedirs(output_dir, exist_ok=True)

# 读取预处理后的数据
df = pd.read_csv(os.path.join(output_dir, "sem_data.csv"))

# 定义基础SEM模型规范
base_model_spec = """
# 测量模型
GQ =~ Perplexity_scaled + Entropy_scaled
SQ =~ Rouge_1_scaled + Rouge_2_scaled + Rouge_L_scaled + METEOR_scaled

# 结构模型
Diversity_scaled ~ HighFreq_scaled + GQ
SQ ~ Diversity_scaled + HighFreq_scaled
"""

# 定义替代模型规范（用于模型比较）
alt_model_1_spec = """
# 测量模型
GQ =~ Perplexity_scaled + Entropy_scaled
SQ =~ Rouge_1_scaled + Rouge_2_scaled + Rouge_L_scaled + METEOR_scaled

# 结构模型
Diversity_scaled ~ HighFreq_scaled + GQ
SQ ~ Diversity_scaled
"""

alt_model_2_spec = """
# 测量模型
GQ =~ Perplexity_scaled + Entropy_scaled
SQ =~ Rouge_1_scaled + Rouge_2_scaled + Rouge_L_scaled + METEOR_scaled

# 结构模型
Diversity_scaled ~ HighFreq_scaled
SQ ~ Diversity_scaled + HighFreq_scaled + GQ
"""

# 非线性模型规范
nonlinear_model_spec = """
# 测量模型
GQ =~ Perplexity_scaled + Entropy_scaled
SQ =~ Rouge_1_scaled + Rouge_2_scaled + Rouge_L_scaled + METEOR_scaled

# 结构模型
Diversity_scaled ~ HighFreq_scaled + HighFreq_sq + GQ
SQ ~ Diversity_scaled + HighFreq_scaled
"""

# 函数：拟合模型并返回结果
def fit_sem_model(model_spec, data, model_name):
    model = Model(model_spec)
    result = model.fit(data)
    
    # 保存模型参数 - 修正方法名
    params = model.inspect()
    print(f"模型 {model_name} 参数：")
    print(params)  # 打印参数以便调试
    params.to_csv(os.path.join(output_dir, f"{model_name}_parameters.csv"))
    
    # 打印参数索引，帮助调试
    print(f"参数索引类型: {type(params.index)}")
    print(f"参数索引值: {params.index.tolist()}")
    
    # 保存模型拟合指标 - 使用semopy 2.3.11版本的正确方法
    try:
        # 尝试使用不同的方法获取拟合指标
        stats = model.fit_info
        print(f"模型 {model_name} 拟合指标 (fit_info)：")
        print(stats)
    except Exception as e:
        print(f"获取模型拟合指标时出错 (fit_info): {e}")
        stats = {}
    
    # 如果stats中的某些键不存在，则设置为默认值而不是None
    fit_indices = {
        'chi_square': stats.get('Chi2', 0),
        'df': stats.get('df', 0),
        'p_value': stats.get('p_value', 1),
        'cfi': stats.get('CFI', 0),
        'tli': stats.get('TLI', 0),
        'rmsea': stats.get('RMSEA', 0),
        'aic': stats.get('AIC', 0),
        'bic': stats.get('BIC', 0)
    }
    
    pd.DataFrame([fit_indices]).to_csv(os.path.join(output_dir, f"{model_name}_fit_indices.csv"), index=False)
    
    # 尝试绘制模型路径图，如果失败则跳过
    try:
        fig = semplot(model, "std")
        plt.savefig(os.path.join(output_dir, f"{model_name}_path_diagram.png"), dpi=300)
        plt.close()
    except ModuleNotFoundError:
        print(f"警告: 无法绘制模型路径图，缺少graphviz模块。跳过绘图步骤。")
    
    return model, params, fit_indices

# 拟合基础模型
print("拟合基础SEM模型...")
base_model, base_params, base_fit = fit_sem_model(base_model_spec, df, "base_model")

# 拟合替代模型
print("拟合替代模型1...")
alt_model_1, alt_params_1, alt_fit_1 = fit_sem_model(alt_model_1_spec, df, "alt_model_1")

print("拟合替代模型2...")
alt_model_2, alt_params_2, alt_fit_2 = fit_sem_model(alt_model_2_spec, df, "alt_model_2")

print("拟合非线性模型...")
nonlinear_model, nonlinear_params, nonlinear_fit = fit_sem_model(nonlinear_model_spec, df, "nonlinear_model")

# 模型比较
models_comparison = pd.DataFrame([
    base_fit,
    alt_fit_1,
    alt_fit_2,
    nonlinear_fit
], index=['基础模型', '替代模型1', '替代模型2', '非线性模型'])

models_comparison.to_csv(os.path.join(output_dir, "models_comparison.csv"))

# 打印模型比较结果
print("\n模型比较结果:")
print(models_comparison[['chi_square', 'p_value', 'cfi', 'rmsea', 'aic', 'bic']])

# 效应分解（以基础模型为例）
# 使用inspect()方法获取参数
params = base_model.inspect()
print("\n基础模型参数：")
print(params)  # 打印参数以便调试

# 检查参数是否存在，使用正确的格式匹配参数名
# 注意：semopy的参数索引格式是 "lval op rval"，而不是 "lval~rval"
try:
    # 查找正确的参数名
    sq_highfreq_param = params.loc[(params['lval'] == 'SQ') & (params['op'] == '~') & (params['rval'] == 'HighFreq_scaled')]
    diversity_highfreq_param = params.loc[(params['lval'] == 'Diversity_scaled') & (params['op'] == '~') & (params['rval'] == 'HighFreq_scaled')]
    sq_diversity_param = params.loc[(params['lval'] == 'SQ') & (params['op'] == '~') & (params['rval'] == 'Diversity_scaled')]
    
    if not sq_highfreq_param.empty and not diversity_highfreq_param.empty and not sq_diversity_param.empty:
        direct_effect = sq_highfreq_param['Estimate'].values[0]
        indirect_effect = diversity_highfreq_param['Estimate'].values[0] * sq_diversity_param['Estimate'].values[0]
        total_effect = direct_effect + indirect_effect
    else:
        print("警告：无法找到必要的参数，使用默认值")
        direct_effect = 0
        indirect_effect = 0
        total_effect = 0
except Exception as e:
    print(f"计算效应分解时出错: {e}")
    direct_effect = 0
    indirect_effect = 0
    total_effect = 0

effects = pd.DataFrame({
    'Effect_Type': ['直接效应', '间接效应', '总效应'],
    'Path': ['HighFreq -> SQ', 'HighFreq -> Diversity -> SQ', 'HighFreq -> SQ (总)'],
    'Value': [direct_effect, indirect_effect, total_effect]
})

effects.to_csv(os.path.join(output_dir, "effects_decomposition.csv"), index=False)
print("\n效应分解:")
print(effects)

# 多组分析（按Run分组）
print("\n按Run分组进行SEM分析...")
run_results = []

for run in df['Run'].unique():
    run_data = df[df['Run'] == run]
    if len(run_data) > 10:  # 确保有足够的样本
        try:
            run_model = Model(base_model_spec)
            run_model.fit(run_data)
            
            # 获取关键路径系数 - 修正方法
            params = run_model.inspect()
            
            # 使用正确的方式查找参数
            diversity_highfreq_param = params.loc[(params['lval'] == 'Diversity_scaled') & (params['op'] == '~') & (params['rval'] == 'HighFreq_scaled')]
            sq_diversity_param = params.loc[(params['lval'] == 'SQ') & (params['op'] == '~') & (params['rval'] == 'Diversity_scaled')]
            sq_highfreq_param = params.loc[(params['lval'] == 'SQ') & (params['op'] == '~') & (params['rval'] == 'HighFreq_scaled')]
            
            if not diversity_highfreq_param.empty and not sq_diversity_param.empty and not sq_highfreq_param.empty:
                diversity_highfreq = diversity_highfreq_param['Estimate'].values[0]
                sq_diversity = sq_diversity_param['Estimate'].values[0]
                sq_highfreq = sq_highfreq_param['Estimate'].values[0]
                
                run_results.append({
                    'Run': run,
                    'Diversity~HighFreq': diversity_highfreq,
                    'SQ~Diversity': sq_diversity,
                    'SQ~HighFreq': sq_highfreq,
                    'Sample_Size': len(run_data)
                })
        except Exception as e:
            print(f"Run {run} 分析失败: {e}")

run_comparison = pd.DataFrame(run_results)
run_comparison.to_csv(os.path.join(output_dir, "run_comparison.csv"), index=False)
print("\nRun分组分析结果:")
print(run_comparison)

# 交叉验证
print("\n进行交叉验证...")
kf = KFold(n_splits=3, shuffle=True, random_state=42)
cv_results = []

for fold, (train_idx, test_idx) in enumerate(kf.split(df)):
    train_df = df.iloc[train_idx]
    test_df = df.iloc[test_idx]
    
    try:
        # 训练SEM模型
        cv_model = Model(base_model_spec)
        cv_model.fit(train_df)
        
        # 记录路径系数 - 使用正确的方式查找参数
        params = cv_model.inspect()
        
        # 使用正确的方式查找参数
        diversity_highfreq_param = params.loc[(params['lval'] == 'Diversity_scaled') & 
                                             (params['op'] == '~') & 
                                             (params['rval'] == 'HighFreq_scaled')]
        sq_diversity_param = params.loc[(params['lval'] == 'SQ') & 
                                       (params['op'] == '~') & 
                                       (params['rval'] == 'Diversity_scaled')]
        
        if not diversity_highfreq_param.empty and not sq_diversity_param.empty:
            diversity_highfreq = diversity_highfreq_param['Estimate'].values[0]
            sq_diversity = sq_diversity_param['Estimate'].values[0]
            
            # 评估预测性能
            predictions = cv_model.predict(test_df)
            mae = mean_absolute_error(test_df['METEOR_scaled'], predictions['METEOR_scaled'])
            
            cv_results.append({
                'Fold': fold + 1,
                'Diversity~HighFreq': diversity_highfreq,
                'SQ~Diversity': sq_diversity,
                'MAE': mae,
                'Train_Size': len(train_df),
                'Test_Size': len(test_df)
            })
    except Exception as e:
        print(f"交叉验证折 {fold+1} 失败: {e}")

cv_df = pd.DataFrame(cv_results)
cv_df.to_csv(os.path.join(output_dir, "cross_validation_results.csv"), index=False)
print("\n交叉验证结果:")
print(cv_df)

# Bootstrap分析
print("\n进行Bootstrap分析...")
n_bootstrap = 100
bootstrap_results = []

for i in range(n_bootstrap):
    # 有放回抽样
    bootstrap_sample = df.sample(frac=1.0, replace=True, random_state=i)
    
    # 拟合模型
    try:
        bootstrap_model = Model(base_model_spec)
        bootstrap_model.fit(bootstrap_sample)
        
        # 记录关键路径系数 - 修正方法
        params = bootstrap_model.inspect()
        
        # 使用正确的方式查找参数
        diversity_highfreq_param = params.loc[(params['lval'] == 'Diversity_scaled') & 
                                             (params['op'] == '~') & 
                                             (params['rval'] == 'HighFreq_scaled')]
        sq_diversity_param = params.loc[(params['lval'] == 'SQ') & 
                                       (params['op'] == '~') & 
                                       (params['rval'] == 'Diversity_scaled')]
        sq_highfreq_param = params.loc[(params['lval'] == 'SQ') & 
                                      (params['op'] == '~') & 
                                      (params['rval'] == 'HighFreq_scaled')]
        
        if not diversity_highfreq_param.empty and not sq_diversity_param.empty and not sq_highfreq_param.empty:
            diversity_highfreq = diversity_highfreq_param['Estimate'].values[0]
            sq_diversity = sq_diversity_param['Estimate'].values[0]
            sq_highfreq = sq_highfreq_param['Estimate'].values[0]
            
            bootstrap_results.append({
                'Bootstrap': i + 1,
                'Diversity~HighFreq': diversity_highfreq,
                'SQ~Diversity': sq_diversity,
                'SQ~HighFreq': sq_highfreq
            })
    except Exception as e:
        print(f"Bootstrap {i+1} 失败: {e}")

bootstrap_df = pd.DataFrame(bootstrap_results)
bootstrap_df.to_csv(os.path.join(output_dir, "bootstrap_results.csv"), index=False)

# 计算Bootstrap置信区间
bootstrap_ci = pd.DataFrame({
    'Path': ['Diversity~HighFreq', 'SQ~Diversity', 'SQ~HighFreq'],
    'Mean': [
        bootstrap_df['Diversity~HighFreq'].mean(),
        bootstrap_df['SQ~Diversity'].mean(),
        bootstrap_df['SQ~HighFreq'].mean()
    ],
    'Std': [
        bootstrap_df['Diversity~HighFreq'].std(),
        bootstrap_df['SQ~Diversity'].std(),
        bootstrap_df['SQ~HighFreq'].std()
    ],
    'CI_Lower': [
        bootstrap_df['Diversity~HighFreq'].quantile(0.025),
        bootstrap_df['SQ~Diversity'].quantile(0.025),
        bootstrap_df['SQ~HighFreq'].quantile(0.025)
    ],
    'CI_Upper': [
        bootstrap_df['Diversity~HighFreq'].quantile(0.975),
        bootstrap_df['SQ~Diversity'].quantile(0.975),
        bootstrap_df['SQ~HighFreq'].quantile(0.975)
    ]
})

bootstrap_ci.to_csv(os.path.join(output_dir, "bootstrap_confidence_intervals.csv"), index=False)
print("\nBootstrap置信区间:")
print(bootstrap_ci)

# 可视化Bootstrap结果
plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
sns.histplot(bootstrap_df['Diversity~HighFreq'], kde=True)
plt.axvline(bootstrap_ci.loc[0, 'Mean'], color='red', linestyle='--')
plt.axvline(bootstrap_ci.loc[0, 'CI_Lower'], color='green', linestyle=':')
plt.axvline(bootstrap_ci.loc[0, 'CI_Upper'], color='green', linestyle=':')
plt.title('Diversity~HighFreq 路径系数分布')

plt.subplot(1, 3, 2)
sns.histplot(bootstrap_df['SQ~Diversity'], kde=True)
plt.axvline(bootstrap_ci.loc[1, 'Mean'], color='red', linestyle='--')
plt.axvline(bootstrap_ci.loc[1, 'CI_Lower'], color='green', linestyle=':')
plt.axvline(bootstrap_ci.loc[1, 'CI_Upper'], color='green', linestyle=':')
plt.title('SQ~Diversity 路径系数分布')

plt.subplot(1, 3, 3)
sns.histplot(bootstrap_df['SQ~HighFreq'], kde=True)
plt.axvline(bootstrap_ci.loc[2, 'Mean'], color='red', linestyle='--')
plt.axvline(bootstrap_ci.loc[2, 'CI_Lower'], color='green', linestyle=':')
plt.axvline(bootstrap_ci.loc[2, 'CI_Upper'], color='green', linestyle=':')
plt.title('SQ~HighFreq 路径系数分布')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "bootstrap_distributions.png"), dpi=300)
plt.close()

print("\nSEM模型构建与评估完成！所有结果已保存至结果目录。")

# 生成结果摘要报告
with open(os.path.join(output_dir, "sem_analysis_summary.md"), "w", encoding="utf-8") as f:
    f.write("# 结构方程模型分析摘要报告\n\n")
    
    f.write("## 1. 模型比较\n\n")
    f.write("| 模型 | Chi-Square | p值 | CFI | RMSEA | AIC | BIC |\n")
    f.write("|------|------------|-----|-----|-------|-----|-----|\n")
    for idx, row in models_comparison[['chi_square', 'p_value', 'cfi', 'rmsea', 'aic', 'bic']].iterrows():
        f.write(f"| {idx} | {row['chi_square']:.3f} | {row['p_value']:.3f} | {row['cfi']:.3f} | {row['rmsea']:.3f} | {row['aic']:.1f} | {row['bic']:.1f} |\n")
    
    f.write("\n## 2. 效应分解\n\n")
    f.write("| 效应类型 | 路径 | 值 |\n")
    f.write("|---------|------|----|\n")
    for _, row in effects.iterrows():
        f.write(f"| {row['Effect_Type']} | {row['Path']} | {row['Value']:.3f} |\n")
    
    f.write("\n## 3. 路径系数Bootstrap置信区间\n\n")
    f.write("| 路径 | 平均值 | 标准差 | 95%置信区间下限 | 95%置信区间上限 |\n")
    f.write("|------|--------|--------|----------------|----------------|\n")
    for _, row in bootstrap_ci.iterrows():
        f.write(f"| {row['Path']} | {row['Mean']:.3f} | {row['Std']:.3f} | {row['CI_Lower']:.3f} | {row['CI_Upper']:.3f} |\n")
    
    f.write("\n## 4. 主要发现\n\n")
    
    # 确定最佳模型
    best_model_idx = models_comparison['aic'].idxmin()
    f.write(f"- 基于AIC准则，{best_model_idx}表现最佳\n")
    
    # 关键路径显著性
    if abs(bootstrap_ci.loc[0, 'Mean']) > 2 * bootstrap_ci.loc[0, 'Std']:
        f.write("- 高频词比例对多样性的影响显著\n")
    
    if abs(bootstrap_ci.loc[1, 'Mean']) > 2 * bootstrap_ci.loc[1, 'Std']:
        f.write("- 多样性对语义质量的影响显著\n")
    
    if abs(bootstrap_ci.loc[2, 'Mean']) > 2 * bootstrap_ci.loc[2, 'Std']:
        f.write("- 高频词比例对语义质量的直接影响显著\n")
    
    # 间接效应与直接效应比较
    if abs(indirect_effect) > abs(direct_effect):
        f.write("- 高频词比例通过多样性对语义质量的间接影响大于直接影响\n")
    else:
        f.write("- 高频词比例对语义质量的直接影响大于通过多样性的间接影响\n")

print(f"分析摘要报告已生成: {os.path.join(output_dir, 'sem_analysis_summary.md')}")