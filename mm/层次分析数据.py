import csv

# 定义判断矩阵数据
matrix = [
    [1, 3, 5, 7],
    [1/3, 1, 2, 5],
    [1/5, 1/2, 1, 3],
    [1/7, 1/5, 1/3, 1]
]

# 将数据写入CSV文件
with open('judgment_matrix.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(matrix)  # 写入多行数据

print("CSV文件已成功创建！")