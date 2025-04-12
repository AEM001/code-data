# 结构方程模型与异常检测分析项目执行流程

下面我将按照执行顺序，详细介绍项目的执行步骤、技术细节以及相关代码文件的功能。

## 第一步：数据预处理与准备

### 执行操作
```bash
python d:\Documents\100\action\sem_analysis_step1.py
```

### 技术细节
- 读取原始数据集 `model_collapse_full_metrics_corrected.csv`
- 重命名列以便于SEM模型使用（如将`3gram_Diversity`重命名为`Diversity`）
- 创建滞后变量（如`Perplexity_lag1`）用于分析代际间影响
- 计算变化率指标（如`Diversity_change`）
- 标准化数据以便于比较
- 创建平方项用于非线性关系分析
- 添加崩溃标记（`is_collapsed`和`next_gen_collapse`）

### 关键代码文件
- `d:\Documents\100\action\results\sem_analysis_step1\sem_preprocessing.py`：负责数据清洗和预处理
  ```python
  # 重命名列以便于SEM模型使用
  df = df.rename(columns={
      '3gram_Diversity': 'Diversity',
      'HighFreq_Ratio': 'HighFreq',
      'Rouge-1': 'Rouge_1',
      'Rouge-2': 'Rouge_2',
      'Rouge-L': 'Rouge_L'
  })
  
  # 创建滞后变量(用于分析代际间影响)
  df['Perplexity_lag1'] = df.groupby('Run')['Perplexity'].shift(1)
  df['Diversity_lag1'] = df.groupby('Run')['Diversity'].shift(1)
  ```

### 输出结果
- `d:\Documents\100\action\results\sem_analysis_step1\sem_data.csv`：预处理后的数据，包含标准化变量和计算的新特征
- 数据探索性分析图表和统计结果

## 第二步：结构方程模型分析

### 执行操作
```bash
python d:\Documents\100\action\sem_analysis_step2.py
```

### 技术细节
- 构建多个SEM模型（基础模型、替代模型和非线性模型）
- 使用semopy库拟合模型并评估拟合度
- 比较不同模型的AIC/BIC、CFI、TLI、RMSEA等指标
- 进行Bootstrap抽样验证路径系数的稳定性
- 计算直接效应、间接效应和总效应
- 进行交叉验证评估模型预测能力
- 按Run分组分析不同批次的路径差异

### 关键代码文件
- `d:\Documents\100\action\results\sem_analysis_step2\sem_model_evaluation.py`：模型构建和评估
  ```python
  # 定义基础SEM模型规范
  base_model_spec = """
  # 测量模型
  GQ =~ Perplexity_scaled + Entropy_scaled
  SQ =~ Rouge_1_scaled + Rouge_2_scaled + Rouge_L_scaled + METEOR_scaled
  
  # 结构模型
  Diversity_scaled ~ HighFreq_scaled + GQ
  SQ ~ Diversity_scaled + HighFreq_scaled
  """
  ```

- `d:\Documents\100\action\results\sem_analysis_step2\sem_cross_validation.py`：交叉验证分析
  ```python
  # 交叉验证
  kf = KFold(n_splits=3, shuffle=True, random_state=42)
  for fold, (train_idx, test_idx) in enumerate(kf.split(df)):
      train_df = df.iloc[train_idx]
      test_df = df.iloc[test_idx]
      
      # 训练SEM模型
      model.fit(train_df)
      
      # 评估预测性能
      predictions = model.predict(test_df)
  ```

- `d:\Documents\100\action\results\sem_analysis_step2\run_complete_analysis.py`：完整分析流程控制

### 输出结果
- `d:\Documents\100\action\results\sem_analysis_step2\models_comparison.csv`：不同模型的拟合指标比较
- `d:\Documents\100\action\results\effects_decomposition.csv`：效应分解结果
- `d:\Documents\100\action\results\sem_analysis_step2\bootstrap_confidence_intervals.csv`：Bootstrap置信区间
- `d:\Documents\100\action\results\sem_analysis_step2\sem_analysis_summary.md`：SEM分析摘要报告

## 第三步：异常检测分析

### 执行操作
```bash
python d:\Documents\100\action\direct_anomaly_detection.py
```

### 技术细节
- 使用隔离森林(Isolation Forest)进行多维特征异常检测
- 基于变化率的异常检测
- 基于指标关系的异常检测
- 比较异常样本与正常样本在关键指标上的差异
- 可视化异常样本在高频词比例与多样性关系中的分布
- 提取预警规则和阈值

### 关键代码文件
- `d:\Documents\100\action\results\sem_analysis_step3\direct_anomaly_detection.py`：异常检测实现
  ```python
  # 使用隔离森林进行异常检测
  features = ['Perplexity_scaled', 'Diversity_scaled', 'HighFreq_scaled', 'METEOR_scaled']
  X = df[features].values
  
  # 初始化并训练隔离森林模型
  iso_forest = IsolationForest(contamination=0.1, random_state=42)
  df['anomaly_score'] = iso_forest.fit_predict(X)
  df['is_anomaly'] = df['anomaly_score'] == -1
  ```

### 输出结果
- `d:\Documents\100\action\results\sem_analysis_step3\direct_anomaly_detection_results.csv`：异常检测结果
- `d:\Documents\100\action\results\sem_analysis_step3\warning_rules.txt`：提取的预警规则
- 异常检测可视化图表

## 第四步：结果整合与报告生成

### 执行操作
```bash
python d:\Documents\100\action\generate_final_report.py
```

### 技术细节
- 整合SEM分析结果和异常检测结果
- 生成综合分析报告
- 创建可视化图表展示关键发现
- 提取最终预警规则和阈值

### 关键代码文件
- `d:\Documents\100\action\results\final_report\generate_final_report.py`：报告生成
  ```python
  # 读取SEM分析结果
  effects_df = pd.read_csv(os.path.join(sem_step2_dir, "effects_decomposition.csv"))
  bootstrap_df = pd.read_csv(os.path.join(sem_step2_dir, "bootstrap_confidence_intervals.csv"))
  
  # 读取异常检测结果
  anomaly_df = pd.read_csv(os.path.join(sem_step3_dir, "direct_anomaly_detection_results.csv"))
  ```

### 输出结果
- `d:\Documents\100\action\results\final_report\sem_anomaly_analysis_report.md`：综合分析报告
- 综合可视化图表

## 第五步：模型崩溃监控工具开发

### 执行操作
```bash
python d:\Documents\100\action\model_crash_monitor.py --mode simulate
```

### 技术细节
- 基于SEM分析和异常检测结果开发实时监控工具
- 实现多维度指标监控功能
- 实现多级预警机制
- 提供针对性干预建议
- 支持模拟模式和实时监控模式

### 关键代码文件
- `d:\Documents\100\action\model_crash_monitor.py`：监控工具主程序
  ```python
  class ModelCrashMonitor:
      """模型崩溃监控工具"""
      
      def __init__(self, config_path=None):
          """初始化监控工具"""
          # 默认配置
          self.config = {
              'warning_rules': {
                  'highfreq_threshold': 0.8236,
                  'diversity_threshold': 25333.93,
                  'meteor_threshold': 0.3727,
                  'diversity_change_threshold': 0.5513,
                  'highfreq_change_threshold': 0.2373,
                  'relation_residual_threshold': 1.5585
              },
              # ...其他配置
          }
  ```

- `d:\Documents\100\action\monitor_config.json`：监控工具配置文件
  ```json
  {
      "warning_rules": {
          "highfreq_threshold": 0.8236,
          "diversity_threshold": 25333.93,
          "meteor_threshold": 0.3727,
          "diversity_change_threshold": 0.5513,
          "highfreq_change_threshold": 0.2373,
          "relation_residual_threshold": 1.5585
      },
      "monitoring": {
          "check_interval": 10,
          "history_window": 20,
          "warning_level_thresholds": [1, 2, 3]
      },
      "output": {
          "log_dir": "d:\\Documents\\100\\action\\results\\monitor_logs",
          "visualization": true
      }
  }
  ```

### 输出结果
- `d:\Documents\100\action\results\monitor_logs\monitor_log.txt`：监控日志
- `d:\Documents\100\action\results\monitor_logs\monitor_status_*.png`：状态可视化图表

## 技术实现与结果对应关系

1. **数据预处理**
   - 实现：`sem_preprocessing.py`
   - 结果：`sem_data.csv`

2. **SEM模型分析**
   - 实现：`sem_model_evaluation.py`, `sem_cross_validation.py`
   - 结果：`models_comparison.csv`, `effects_decomposition.csv`, `bootstrap_confidence_intervals.csv`

3. **异常检测**
   - 实现：`direct_anomaly_detection.py`
   - 结果：`direct_anomaly_detection_results.csv`, `warning_rules.txt`

4. **报告生成**
   - 实现：`generate_final_report.py`
   - 结果：`sem_anomaly_analysis_report.md`

5. **监控工具**
   - 实现：`model_crash_monitor.py`, `monitor_config.json`
   - 结果：`monitor_logs/`目录下的日志和图表

通过这一系列步骤，项目成功实现了从数据预处理、模型构建、效应分解到异常检测和预警规则提取的完整分析流程，最终开发出了一个实用的模型崩溃监控工具。