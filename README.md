# 东南大学"友宏先锋队"统计建模竞赛代码与数据资料库

本资料库包含东南大学"友宏先锋队"参加统计建模竞赛的代码和数据资料，主要聚焦于模型崩溃分析与预警系统的研究与实现。

## 项目概述

本项目通过对模型崩溃现象的系统研究，构建了一套完整的分析框架和预警系统，包括数据生成、指标分析、因果关系建模和实时监控工具。项目分为两个主要模块：

1. **Data_generate**: 模型崩溃数据生成与基础分析模块
2. **action**: 基于结构方程模型的深度分析与预警系统模块

## 资料库结构

```
code-data/
├── Data_generate/                # 数据生成与基础分析模块
│   ├── Train&Generate/           # 训练与数据生成模块
│   │   ├── main.py               # 主程序，负责模型训练和数据生成
│   │   ├── dataset.py            # 数据集处理相关功能
│   │   ├── run_iterations.py     # 执行多代迭代训练的控制脚本
│   │   └── run_mixed_iterations.py # 混合数据策略的迭代训练脚本
│   │
│   └── Data/                     # 数据分析模块
│       └── Main/                 # 主要分析脚本
│           ├── collect_data.py   # 收集和合并数据
│           ├── normalize_metrics.py # 标准化指标数据
│           ├── calculate_correlation.py # 计算指标间相关性
│           ├── calculate_pca.py  # 执行主成分分析
│           ├── 插值.py           # 对缺失数据进行拉格朗日插值
│           └── show.py           # 可视化展示结果
│
└── action/                       # 深度分析与预警系统模块
    ├── sem_analysis_step1.py     # 数据预处理和初步分析
    ├── sem_analysis_step2.py     # 结构方程模型构建和效应分解
    ├── direct_anomaly_detection.py # 直接异常检测分析
    ├── generate_final_report.py  # 结果整合与报告生成
    ├── Project_process.md        # 项目执行流程文档
    │
    ├── Monitor/                  # 监控工具模块
    │   ├── model_crash_monitor.py # 模型崩溃监控工具
    │   ├── monitor_config.json   # 监控工具配置文件
    │   └── README.md             # 监控工具使用说明
    │
    └── results/                  # 分析结果目录
        ├── sem_analysis_step1/   # 预处理结果
        ├── sem_analysis_step2/   # SEM分析结果
        ├── sem_analysis_step3/   # 异常检测结果
        └── final_report/         # 综合分析报告
```

## 研究内容与方法

### 1. 模型崩溃数据生成 (Data_generate)

通过迭代训练和生成过程，模拟模型崩溃现象，收集关键指标数据：

- **迭代训练方法**：每一代使用上一代生成的数据进行训练，然后生成新数据
- **混合数据策略**：可以指定原始数据和生成数据的混合比例
- **指标收集**：记录每代模型的困惑度、多样性、高频词比例等指标
- **数据分析**：通过相关性分析、主成分分析等方法探索指标间关系

### 2. 结构方程模型分析 (action)

基于收集的指标数据，构建结构方程模型，分析指标间的因果关系：

- **因果路径验证**：验证"高频词比例→多样性→语义质量"的因果传导路径
- **效应分解**：量化高频词比例对语义质量的直接效应、间接效应和总效应
- **Bootstrap分析**：验证关键路径系数的稳定性和显著性
- **交叉验证**：评估模型的预测能力和泛化性

### 3. 异常检测与预警规则 (action)

基于SEM分析结果，开发异常检测方法和预警规则：

- **多维异常检测**：使用隔离森林识别多维特征空间中的异常样本
- **指标对比分析**：比较异常样本与正常样本在关键指标上的差异
- **预警规则提取**：基于异常样本特征提取预警规则和阈值
- **预警系统开发**：实现多级预警机制和干预建议

## 关键指标

本项目分析的主要指标包括：

- **Perplexity (困惑度)**：评估模型预测下一个词的能力
- **3gram_Diversity (三元语法多样性)**：评估生成文本的多样性
- **HighFreq_Ratio (高频词比例)**：评估生成文本中高频词的比例
- **Entropy (熵)**：评估生成文本的信息量
- **Rouge-1/2/L**：评估生成文本与参考文本的相似度
- **METEOR**：评估生成文本的语义质量

## 研究发现

### 1. 模型崩溃的因果机制

- 高频词比例增加是模型崩溃的早期信号
- 高频词比例通过降低多样性间接影响语义质量
- 多样性与语义质量之间存在显著的正向关系

### 2. 预警规则与阈值

基于异常检测分析，提取了以下预警规则：

- **高频词比例阈值**：0.8236（超过此值可能导致多样性下降）
- **多样性阈值**：25333.93（低于此值可能影响语义质量）
- **METEOR评分阈值**：0.3727（低于此值表示语义质量显著下降）
- **多样性变化率阈值**：0.5513（超过此变化率表示多样性急剧下降）
- **高频词比例变化率阈值**：0.2373（超过此变化率表示高频词比例急剧上升）
- **关系残差阈值**：1.5585（超过此值表示指标间关系异常）

## 使用指南

### 1. 数据生成与基础分析 (Data_generate)

#### 训练与生成流程
1. 使用 `run_iterations.py` 或 `run_mixed_iterations.py` 启动实验
2. 第一代使用原始数据训练模型
3. 使用训练好的模型生成新数据
4. 后续代使用上一代生成的数据（或混合数据）训练模型
5. 重复步骤3-4直到达到指定的迭代次数

#### 数据分析流程
1. 使用 `collect_data.py` 收集多个实验的指标数据
2. 使用 `normalize_metrics.py` 对数据进行标准化
3. 使用 `calculate_correlation.py` 分析指标间的相关性
4. 使用 `calculate_pca.py` 进行主成分分析
5. 如有缺失数据，使用 `插值.py` 进行补全
6. 最后使用 `show.py` 可视化展示分析结果

### 2. 深度分析与预警系统 (action)

#### 数据预处理与SEM分析流程
1. 执行数据预处理：`python sem_analysis_step1.py`
2. 进行SEM模型分析：`python sem_analysis_step2.py`
3. 分析结果保存在`results/sem_analysis_step1/`和`results/sem_analysis_step2/`目录下

#### 异常检测与预警规则流程
1. 执行异常检测分析：`python direct_anomaly_detection.py`
2. 生成综合分析报告：`python generate_final_report.py`
3. 分析结果保存在`results/sem_analysis_step3/`和`results/final_report/`目录下

#### 模型崩溃监控工具使用流程
1. 配置`monitor_config.json`文件，设置预警规则、监控参数和输出选项
2. 模拟模式：`python model_crash_monitor.py --mode simulate --data "results/sem_analysis_step3/direct_anomaly_detection_results.csv"`
3. 实时监控模式：`python model_crash_monitor.py --mode monitor --interval 30`

## 集成应用示例

```python
from model_crash_monitor import ModelCrashMonitor

# 初始化监控工具
monitor = ModelCrashMonitor()

# 在训练循环中使用
for epoch in range(num_epochs):
    # 训练代码...
    
    # 收集当前指标
    metrics = {
        'Perplexity': current_perplexity,
        'Diversity': current_diversity,
        'HighFreq': current_highfreq_ratio,
        'METEOR': current_meteor
    }
    
    # 检查是否有崩溃风险
    warning_level, warnings = monitor.check_metrics(metrics)
    
    # 根据预警级别采取措施
    if warning_level > 1:
        print(f"警告：检测到崩溃风险，级别{warning_level}")
        print(f"触发规则：{warnings}")
        # 采取干预措施...
```

## 团队成员

东南大学"友宏先锋队"成员（按姓氏拼音排序）：
- 鲁佳琦（队长）
- 许严中
- 张晨

## 致谢

感谢东南大学对本项目的支持，以及所有为本项目提供帮助和建议的老师和同学。

## 许可证

MIT License