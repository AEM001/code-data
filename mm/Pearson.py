import pandas as pd

# 读取Excel文件
df = pd.read_excel("D:\\Anaconda\\Mat\\代码和例题数据\\八年级男生体测数据.xlsx", engine='openpyxl')

# 显示数据的前几行
print(df.head())