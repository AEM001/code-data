import pandas as pd
from scipy.interpolate import lagrange

# 读取CSV文件
df = pd.read_csv('d:/Documents/code&data/Data_generate/Data/Main/model_collapse_full_metrics.csv')

# 为每个Run单独处理
for run in [1, 2, 3]:
    # 获取当前Run的第8、10、11代数据作为插值点
    interpolation_points = df[(df['Run'] == run) & 
                             (df['Generation'].isin([8, 10, 11]))].sort_values('Generation')
    
    # 对每个指标进行三次拉格朗日插值
    for column in ['Perplexity', '3gram_Diversity', 'HighFreq_Ratio', 'Entropy', 
                  'Rouge-1', 'Rouge-2', 'Rouge-L', 'METEOR']:
        # 准备插值数据点 (x=Generation, y=metric)
        x = interpolation_points['Generation'].values
        y = interpolation_points[column].values
        
        # 创建三次拉格朗日插值多项式
        poly = lagrange(x, y)
        
        # 计算第9代的插值结果
        gen9_value = poly(9)
        
        # 更新DataFrame中的值
        df.loc[(df['Generation'] == 9) & (df['Run'] == run), column] = gen9_value

# 保存修正后的数据
df.to_csv('d:/Documents/code&data/Data_generate/Data/Main/model_collapse_full_metrics_corrected.csv', index=False)
