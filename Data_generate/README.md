# 项目概述
Data_generate 是一个用于模型崩溃分析的项目，主要包含两个核心部分：
1. 数据生成与训练模块 (Train&Generate)
2. 数据分析模块 (Data/Main)

## 项目结构

```
Data_generate/
├── Train&Generate/              # 训练与数据生成模块
│   ├── main.py                  # 主程序，负责模型训练和数据生成
│   ├── dataset.py               # 数据集处理相关功能
│   ├── run_iterations.py        # 执行多代迭代训练的控制脚本
│   └── run_mixed_iterations.py  # 执行混合数据策略的迭代训练脚本
│
└── Data/                        # 数据分析模块
    └── Main/                    # 主要分析脚本
        ├── collect_data.py      # 收集和合并数据
        ├── normalize_metrics.py # 标准化指标数据
        ├── calculate_correlation.py # 计算指标间相关性
        ├── calculate_pca.py     # 执行主成分分析
        ├── 插值.py              # 对缺失数据进行拉格朗日插值
        └── show.py              # 可视化展示结果
```

## 功能模块详解

### 1. 训练与数据生成模块 (Train&Generate)

#### dataset.py
- 提供数据集加载和预处理功能
- 包含 `prepare_data()` 函数用于从本地或HuggingFace下载数据集
- 包含 `preprocess_datasets()` 函数用于对原始数据进行分词和处理
- 定义了数据加载器类 `MyDataLoader` 用于批量加载数据

#### main.py
- 项目的核心执行文件
- 负责模型的加载、训练、评估和生成
- 支持多种参数配置，如模型标签、学习率、批次大小等
- 包含数据生成功能，可以生成新的文本数据并计算困惑度

#### run_iterations.py
- 控制多代迭代训练的脚本
- 实现了从第一代到第N代的自动化训练和生成流程
- 每一代使用上一代生成的数据进行训练，然后生成新数据
- 保存每一代的模型、生成数据和困惑度指标

#### run_mixed_iterations.py
- 实现混合数据策略的迭代训练
- 可以指定原始数据和生成数据的混合比例
- 支持从指定代数恢复实验
- 提供更灵活的训练配置选项

### 2. 数据分析模块 (Data/Main)

#### collect_data.py
- 从多个CSV文件中收集模型崩溃指标数据
- 计算每代三个run的平均值
- 将结果保存为矩阵格式，用于后续分析

#### normalize_metrics.py
- 使用Z-score标准化方法对指标数据进行标准化处理
- 确保不同量级的指标可以进行比较

#### calculate_correlation.py
- 计算各指标间的相关系数矩阵
- 生成热力图可视化相关性

#### calculate_pca.py
- 对标准化后的数据进行主成分分析
- 计算特征值、特征向量、贡献率等
- 输出完整的PCA分析结果表格

#### 插值.py
- 使用拉格朗日插值法对缺失的第9代数据进行插值补全
- 确保数据的连续性和完整性

#### show.py
- 绘制各项指标随代数变化的趋势图
- 直观展示模型崩溃过程

## 执行流程

### 训练与生成流程
1. 使用 `run_iterations.py` 或 `run_mixed_iterations.py` 启动实验
2. 第一代使用原始数据训练模型
3. 使用训练好的模型生成新数据
4. 后续代使用上一代生成的数据（或混合数据）训练模型
5. 重复步骤3-4直到达到指定的迭代次数
6. 每一代保存模型、生成数据和困惑度指标

### 数据分析流程
1. 使用 `collect_data.py` 收集多个实验的指标数据
2. 使用 `normalize_metrics.py` 对数据进行标准化
3. 使用 `calculate_correlation.py` 分析指标间的相关性
4. 使用 `calculate_pca.py` 进行主成分分析
5. 如有缺失数据，使用 `插值.py` 进行补全
6. 最后使用 `show.py` 可视化展示分析结果

## 分析指标

项目分析的主要指标包括：
- Perplexity (困惑度)
- 3gram_Diversity (三元语法多样性)
- HighFreq_Ratio (高频词比例)
- Entropy (熵)
- Rouge-1 (Rouge-1评分)
- Rouge-2 (Rouge-2评分)
- Rouge-L (Rouge-L评分)
- METEOR (METEOR评分)

这些指标共同用于评估模型生成文本的质量和多样性，以及模型崩溃的程度。