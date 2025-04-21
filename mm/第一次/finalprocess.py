import pandas as pd
import numpy as np

# 数据
data = {
    '动物': ['猪', '鸡', '鸭', '兔'],
    '经济性 (C1) (%)': [12, 18, 15, 8],
    '技术可行性 (C2)': [7, 9, 8, 6],
    '环境影响 (C3)': [4, 6, 5, 8],
    '社会效益 (C4)': [8, 6, 5, 4]
}

# 创建DataFrame
df = pd.DataFrame(data)

# 先将数值列转换为浮点数类型（修改这里）
float_columns = df.columns[1:]
df[float_columns] = df[float_columns].apply(pd.to_numeric, errors='coerce').astype(float)

# 归一化处理（使用比例标准化）
# 对于正向指标（经济性、技术可行性、社会效益）：直接除以列总和
# 对于反向指标（环境影响）：先取倒数再除以列总和
for col in df.columns[1:]:
    if col == '环境影响 (C3)':  # 反向指标处理
        df[col] = (1 / df[col]) / (1 / df[col]).sum()  # 取倒数后标准化
    else:  # 正向指标处理
        df[col] = df[col] / df[col].sum()  # 直接标准化

# 打印归一化后的完整表格
print("\n归一化后的完整表格：")
print(df.to_string(index=False))  # 使用to_string避免重复表头

# 权重向量
weights = np.array([0.57381126, 0.23882897, 0.13101431, 0.05634546])

# 计算最终值
df['最终值'] = np.dot(df.iloc[:, 1:], weights)

# 输出每种动物的得分情况
print("\n各动物最终得分：")
for index, row in df.iterrows():
    print(f"{row['动物']} 的最终得分为: {row['最终值']:.4f}")

# 输出到CSV文件
output_path = r"D:\Code\Python\mm\第一次\output.csv"
df.to_csv(output_path, index=False)