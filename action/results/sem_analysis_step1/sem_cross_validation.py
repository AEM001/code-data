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

# 设置matplotlib使用中文字体
if "MATPLOTLIBRC" in os.environ:
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False

# 读取预处理后的数据
try:
    df = pd.read_csv(os.path.join(data_dir, "results", "sem_data.csv"))
    print(f"成功读取数据，共 {len(df)} 行")
except Exception as e:
    print(f"读取数据时出错: {e}")
    df = pd.DataFrame()

if df.empty:
    print("错误: 数据为空，无法继续分析")
    exit(1)

# 定义基础SEM模型规范
base_model_spec = """
# 测量模型
GQ =~ Perplexity_scaled + Entropy_scaled
SQ =~ Rouge_1_scaled + Rouge_2_scaled + Rouge_L_scaled + METEOR_scaled

# 结构模型
Diversity_scaled ~ HighFreq_scaled + GQ
SQ ~ Diversity_scaled + HighFreq_scaled
"""

# 多组分析（按Run分组）
print("\n按Run分组进行SEM分析...")
run_results = []

for run in df['Run'].unique():
    run_data = df[df['Run'] == run]
    if len(run_data) > 10:  # 确保有足够的样本
        try:
            print(f"分析Run {run}，样本量: {len(run_data)}")
            run_model = Model(base_model_spec)
            run_model.fit(run_data)
            
            # 获取关键路径系数
            params = run_model.inspect()
            
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
                
                run_results.append({
                    'Run': run,
                    'Diversity~HighFreq': diversity_highfreq,
                    'SQ~Diversity': sq_diversity,
                    'SQ~HighFreq': sq_highfreq,
                    'Sample_Size': len(run_data)
                })
        except Exception as e:
            print(f"Run {run} 分析失败: {e}")

if run_results:
    run_comparison = pd.DataFrame(run_results)
    run_comparison.to_csv(os.path.join(output_dir, "run_comparison.csv"), index=False)
    print("\nRun分组分析结果:")
    print(run_comparison)
else:
    print("警告: 未能生成有效的Run分组分析结果")

# 交叉验证
print("\n进行交叉验证...")
kf = KFold(n_splits=3, shuffle=True, random_state=42)
cv_results = []

for fold, (train_idx, test_idx) in enumerate(kf.split(df)):
    try:
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]
        
        print(f"折 {fold+1}: 训练集 {len(train_df)} 样本, 测试集 {len(test_df)} 样本")
        
        # 训练SEM模型
        cv_model = Model(base_model_spec)
        cv_model.fit(train_df)
        
        # 记录路径系数
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

if cv_results:
    cv_df = pd.DataFrame(cv_results)
    cv_df.to_csv(os.path.join(output_dir, "cross_validation_results.csv"), index=False)
    print("\n交叉验证结果:")
    print(cv_df)
else:
    print("警告: 未能生成有效的交叉验证结果")

print("\n交叉验证和多组分析完成")