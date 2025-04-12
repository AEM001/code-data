import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from semopy import Model
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error

# 设置工作目录和文件路径
data_dir = r"d:\Documents\100\action"
# 从环境变量获取输出目录，如果未设置则使用默认值
output_dir = os.environ.get("SEM_OUTPUT_DIR", r"d:\Documents\100\action\results\sem_analysis_results")
os.makedirs(output_dir, exist_ok=True)

# 读取预处理后的数据
df = pd.read_csv(os.path.join(data_dir, "results", "sem_data.csv"))

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
# 在函数中添加以下代码来获取拟合指标
def fit_sem_model(model_spec, data, model_name):
    model = Model(model_spec)
    result = model.fit(data)
    
    # 保存模型参数
    params = model.inspect()
    params.to_csv(os.path.join(output_dir, f"{model_name}_parameters.csv"))
    
    # 尝试获取模型拟合指标 - 使用semopy 2.3.11版本的正确方法
    try:
        # 直接从模型对象获取拟合指标
        chi2 = getattr(model, 'chi_square', None)
        df_model = getattr(model, 'df', None)
        p_value = getattr(model, 'p_value', None)
        
        # 手动计算拟合指标
        n = len(data)
        aic = chi2 + 2 * len(params) if chi2 is not None else None
        bic = chi2 + len(params) * np.log(n) if chi2 is not None else None
        
        # 尝试计算CFI, TLI, RMSEA
        cfi = getattr(model, 'cfi', None)
        tli = getattr(model, 'tli', None)
        rmsea = getattr(model, 'rmsea', None)
        
        stats = {
            'chisq': chi2,
            'df': df_model,
            'pvalue': p_value,
            'cfi': cfi,
            'tli': tli,
            'rmsea': rmsea,
            'aic': aic,
            'bic': bic
        }
        
        print(f"模型 {model_name} 拟合指标:")
        for k, v in stats.items():
            print(f"  {k}: {v}")
    except Exception as e:
        print(f"获取模型拟合指标时出错: {e}")
        stats = {}
    
    # 如果stats中的某些键不存在，则设置为0而不是None
    fit_indices = {
        'chi_square': stats.get('chisq', 0),
        'df': stats.get('df', 0),
        'p_value': stats.get('pvalue', 1),
        'cfi': stats.get('cfi', 0),
        'tli': stats.get('tli', 0),
        'rmsea': stats.get('rmsea', 0),
        'aic': stats.get('aic', 0),
        'bic': stats.get('bic', 0)
    }
    
    pd.DataFrame([fit_indices]).to_csv(os.path.join(output_dir, f"{model_name}_fit_indices.csv"), index=False)
    
    # 尝试绘制模型路径图
    try:
        from semopy import semplot
        fig = semplot(model, "std")
        plt.savefig(os.path.join(output_dir, f"{model_name}_path_diagram.png"), dpi=300)
        plt.close()
    except:
        print(f"警告: 无法绘制{model_name}的路径图")
    
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
print("\n进行效应分解分析...")
try:
    # 使用inspect()方法获取参数
    params = base_model.inspect()
    
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
        
        # 记录关键路径系数
        params = bootstrap_model.inspect()
        
        # 查找正确的参数名
        sq_highfreq_param = params.loc[(params['lval'] == 'SQ') & (params['op'] == '~') & (params['rval'] == 'HighFreq_scaled')]
        diversity_highfreq_param = params.loc[(params['lval'] == 'Diversity_scaled') & (params['op'] == '~') & (params['rval'] == 'HighFreq_scaled')]
        sq_diversity_param = params.loc[(params['lval'] == 'SQ') & (params['op'] == '~') & (params['rval'] == 'Diversity_scaled')]
        
        if not sq_highfreq_param.empty and not diversity_highfreq_param.empty and not sq_diversity_param.empty:
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

# 计算Bootstrap置信区间
if bootstrap_results:
    bootstrap_df = pd.DataFrame(bootstrap_results)
    bootstrap_summary = pd.DataFrame({
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
            np.percentile(bootstrap_df['Diversity~HighFreq'], 2.5),
            np.percentile(bootstrap_df['SQ~Diversity'], 2.5),
            np.percentile(bootstrap_df['SQ~HighFreq'], 2.5)
        ],
        'CI_Upper': [
            np.percentile(bootstrap_df['Diversity~HighFreq'], 97.5),
            np.percentile(bootstrap_df['SQ~Diversity'], 97.5),
            np.percentile(bootstrap_df['SQ~HighFreq'], 97.5)
        ]
    })
    
    bootstrap_summary.to_csv(os.path.join(output_dir, "bootstrap_confidence_intervals.csv"), index=False)
    print("\nBootstrap置信区间:")
    print(bootstrap_summary)
else:
    print("警告: Bootstrap分析未产生有效结果")

print("\n模型评估与效应分解分析完成")