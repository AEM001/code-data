# 模型崩溃监控与分析系统

本项目基于结构方程模型(SEM)和异常检测技术，构建了一套完整的模型崩溃分析与预警系统，用于监测、分析和预防模型崩溃现象。

## 项目概述

本项目通过分析模型崩溃相关指标之间的因果关系，建立预警机制，实现对模型崩溃过程的早期监测和干预。主要包含三个核心部分：
1. 数据预处理与SEM分析模块
2. 异常检测与预警规则模块
3. 实时监控工具模块

## 项目结构

```
action/
├── sem_analysis_step1.py        # 数据预处理和初步分析
├── sem_analysis_step2.py        # 结构方程模型构建和效应分解
├── direct_anomaly_detection.py  # 直接异常检测分析
├── generate_final_report.py     # 结果整合与报告生成
├── Project_process.md           # 项目执行流程文档
│
├── Monitor/                     # 监控工具模块
│   ├── model_crash_monitor.py   # 模型崩溃监控工具
│   ├── monitor_config.json      # 监控工具配置文件
│   └── README.md                # 监控工具使用说明
│
└── results/                     # 分析结果目录
    ├── sem_analysis_step1/      # 预处理结果
    │   ├── sem_preprocessing.py # 数据预处理脚本
    │   ├── sem_data.csv         # 预处理后的数据
    │   └── sem_model_building.py# 初步模型构建
    │
    ├── sem_analysis_step2/      # SEM分析结果
    │   ├── sem_model_evaluation.py  # 模型评估脚本
    │   ├── sem_cross_validation.py  # 交叉验证脚本
    │   ├── sem_analysis_report.py   # 分析报告生成
    │   ├── models_comparison.csv    # 模型比较结果
    │   └── effects_decomposition.csv# 效应分解结果
    │
    ├── sem_analysis_step3/      # 异常检测结果
    │   ├── direct_anomaly_detection.py  # 异常检测实现
    │   ├── distribution.py      # 分布分析脚本
    │   └── direct_anomaly_detection_results.csv # 异常检测结果
    │
    └── final_report/            # 综合分析报告
        ├── generate_final_report.py  # 报告生成脚本
        └── sem_anomaly_analysis_report.md # 综合分析报告
```

## 功能模块详解

### 1. 数据预处理与SEM分析模块

#### sem_analysis_step1.py
- 读取原始数据集并进行预处理
- 重命名列以便于SEM模型使用（如将`3gram_Diversity`重命名为`Diversity`）
- 创建滞后变量（如`Perplexity_lag1`）用于分析代际间影响
- 计算变化率指标（如`Diversity_change`）
- 标准化数据以便于比较
- 创建平方项用于非线性关系分析
- 添加崩溃标记（`is_collapsed`和`next_gen_collapse`）

#### sem_analysis_step2.py
- 构建多个SEM模型（基础模型、替代模型和非线性模型）
- 使用semopy库拟合模型并评估拟合度
- 比较不同模型的AIC/BIC、CFI、TLI、RMSEA等指标
- 进行Bootstrap抽样验证路径系数的稳定性
- 计算直接效应、间接效应和总效应
- 进行交叉验证评估模型预测能力
- 按Run分组分析不同批次的路径差异

### 2. 异常检测与预警规则模块

#### direct_anomaly_detection.py
- 使用隔离森林(Isolation Forest)进行多维特征异常检测
- 基于变化率的异常检测
- 基于指标关系的异常检测
- 比较异常样本与正常样本在关键指标上的差异
- 可视化异常样本在高频词比例与多样性关系中的分布
- 提取预警规则和阈值

#### generate_final_report.py
- 整合SEM分析结果和异常检测结果
- 生成综合分析报告
- 创建可视化图表展示关键发现
- 提取最终预警规则和阈值

### 3. 实时监控工具模块

#### model_crash_monitor.py
- 基于SEM分析和异常检测结果开发实时监控工具
- 实现多维度指标监控功能
- 实现多级预警机制
- 提供针对性干预建议
- 支持模拟模式和实时监控模式

#### monitor_config.json
- 配置预警规则阈值
- 设置监控参数（检查间隔、历史窗口大小等）
- 配置输出选项（日志目录、可视化开关等）

## 执行流程

### 1. 数据预处理与SEM分析流程
1. 执行数据预处理：`python sem_analysis_step1.py`
2. 进行SEM模型分析：`python sem_analysis_step2.py`
3. 分析结果保存在`results/sem_analysis_step1/`和`results/sem_analysis_step2/`目录下

### 2. 异常检测与预警规则流程
1. 执行异常检测分析：`python direct_anomaly_detection.py`
2. 生成综合分析报告：`python generate_final_report.py`
3. 分析结果保存在`results/sem_analysis_step3/`和`results/final_report/`目录下

### 3. 模型崩溃监控工具使用流程
1. 配置`monitor_config.json`文件，设置预警规则、监控参数和输出选项
2. 模拟模式：`python model_crash_monitor.py --mode simulate --data "results/sem_analysis_step3/direct_anomaly_detection_results.csv"`
3. 实时监控模式：`python model_crash_monitor.py --mode monitor --interval 30`

## 研究结果

### 1. 结构方程模型分析结果
- **因果路径验证**：成功验证了"高频词比例→多样性→语义质量"的因果传导路径
- **效应分解**：量化了高频词比例对语义质量的直接效应、间接效应和总效应
- **Bootstrap置信区间**：验证了关键路径系数的稳定性和显著性

### 2. 异常检测分析结果
- **多维异常检测**：识别出模型崩溃前的异常样本
- **指标对比分析**：发现异常样本在高频词比例、多样性和语义质量上的显著差异
- **预警规则提取**：基于异常样本特征提取了预警规则和阈值

### 3. 预警规则与阈值
- **高频词比例阈值**：0.8236（超过此值可能导致多样性下降）
- **多样性阈值**：25333.93（低于此值可能影响语义质量）
- **METEOR评分阈值**：0.3727（低于此值表示语义质量显著下降）
- **多样性变化率阈值**：0.5513（超过此变化率表示多样性急剧下降）
- **高频词比例变化率阈值**：0.2373（超过此变化率表示高频词比例急剧上升）
- **关系残差阈值**：1.5585（超过此值表示指标间关系异常）

## 技术实现与结果对应关系

1. **数据预处理**
   - 实现：`sem_preprocessing.py`
   - 结果：`sem_data.csv`

2. **SEM模型分析**
   - 实现：`sem_model_evaluation.py`, `sem_cross_validation.py`
   - 结果：`models_comparison.csv`, `effects_decomposition.csv`, `bootstrap_confidence_intervals.csv`

3. **异常检测分析**
   - 实现：`direct_anomaly_detection.py`
   - 结果：`direct_anomaly_detection_results.csv`, `warning_rules.txt`

4. **监控工具开发**
   - 实现：`model_crash_monitor.py`
   - 配置：`monitor_config.json`
   - 结果：`monitor_logs/monitor_log.txt`, `monitor_logs/monitor_status_*.png`

## 使用方法

### 1. 基础配置
配置`monitor_config.json`文件，设置预警规则、监控参数和输出选项。

### 2. 运行模式

- **模拟模式**：使用历史数据进行模拟监控
  ```bash
  python model_crash_monitor.py --mode simulate --data "results/sem_analysis_step3/direct_anomaly_detection_results.csv"
  ```

- **实时监控模式**：连接实时数据源进行监控
  ```bash
  python model_crash_monitor.py --mode monitor --interval 30
  ```

### 3. 集成到训练流程

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

## 注意事项

- 确保所有Python脚本中的文件路径正确指向您的数据文件位置
- 分析流程应按照上述顺序执行，以确保每个步骤的输入数据正确
- 监控工具的预警阈值可能需要根据具体模型和数据集进行调整
```
