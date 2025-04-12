# 模型崩溃监控工具 (Model Crash Monitor)

## 项目概述

本工具是基于结构方程模型(SEM)分析和异常检测研究成果开发的实时监控系统，用于在模型训练过程中检测潜在的崩溃风险。通过多维度指标监控、异常检测和预警机制，帮助研究人员及时发现并解决训练过程中的问题。

## 主要功能

### 1. 实时指标监控
- 高频词比例监控
- 多样性指标监控
- 语义质量监控（METEOR分数）
- 指标变化率分析

### 2. 多维度异常检测
- 基于隔离森林的多维特征异常检测
- 基于变化率的异常检测
- 基于指标关系的异常检测

### 3. 预警系统
- 多级预警机制（提示、警告、严重警告）
- 可视化预警信息
- 实时预警日志

### 4. 干预建议
- 基于异常类型的针对性建议
- 自动化干预参数推荐
- 预警等级相关的处理建议

## 安装说明

### 环境要求
- Python 3.7+
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

### 安装依赖
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## 使用方法

### 1. 基础配置
在使用工具前，请先配置 `monitor_config.json`：

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

### 2. 运行模式

#### 2.1 模拟模式
使用历史数据进行模拟监控：
```bash
python model_crash_monitor.py --mode simulate --data "d:\Documents\100\action\results\sem_analysis_step3\direct_anomaly_detection_results.csv"
```

#### 2.2 实时监控模式
连接实时数据源进行监控：
```bash
python model_crash_monitor.py --mode monitor --interval 30
```

#### 2.3 自定义配置运行
使用自定义配置文件：
```bash
python model_crash_monitor.py --mode simulate --config "d:\Documents\100\action\monitor_config.json"
```

### 3. 集成到训练流程

```python
from model_crash_monitor import ModelCrashMonitor

# 初始化监控工具
monitor = ModelCrashMonitor()

# 训练循环中使用
for epoch in range(num_epochs):
    for batch in dataloader:
        # 训练步骤
        ...
        
        # 每N步进行监控
        if step % monitor_interval == 0:
            metrics = {
                'Run': run_id,
                'Generation': step,
                'Perplexity': perplexity,
                'Diversity': diversity,
                'HighFreq': high_freq_ratio,
                'METEOR': meteor_score
            }
            monitor.add_observation(metrics)
            warnings = monitor.check_warnings()
            
            # 处理预警
            if warnings:
                suggestions = monitor.get_intervention_suggestions(warnings)
                # 实现自动干预逻辑
                ...
```

## 输出说明

### 1. 监控日志
- 位置：配置文件中指定的 `log_dir`
- 内容：包含时间戳、状态信息、预警详情和干预建议
- 格式：文本文件，易于阅读和分析

### 2. 可视化图表
- 关键指标趋势图
- 高频词比例与多样性关系图
- 变化率监控图
- 预警状态统计图

### 3. 预警信息
- 预警级别：提示、警告、严重警告
- 预警内容：具体异常指标和超限情况
- 干预建议：针对性的改进建议

## 配置详解

### 1. 预警规则 (warning_rules)
- `highfreq_threshold`: 高频词比例上限
- `diversity_threshold`: 多样性下限
- `meteor_threshold`: METEOR分数下限
- `diversity_change_threshold`: 多样性变化率阈值
- `highfreq_change_threshold`: 高频词比例变化率阈值
- `relation_residual_threshold`: 指标关系异常阈值

### 2. 监控设置 (monitoring)
- `check_interval`: 检查间隔（秒）
- `history_window`: 历史窗口大小
- `warning_level_thresholds`: 预警级别阈值

### 3. 输出设置 (output)
- `log_dir`: 日志输出目录
- `visualization`: 是否生成可视化图表

## 最佳实践

### 1. 监控频率设置
- 建议根据训练步骤的耗时调整监控间隔
- 对于快速训练，可以设置较短的间隔（如10秒）
- 对于长时训练，可以适当增加间隔（如60秒）

### 2. 预警阈值调整
- 初次使用时建议使用默认阈值
- 根据实际训练情况逐步调整阈值
- 可以通过模拟模式测试不同阈值的效果

### 3. 干预策略
- 对于轻微预警，可以继续观察
- 对于中度预警，考虑调整相关参数
- 对于严重预警，建议及时干预或暂停训练

## 常见问题

### 1. 监控工具不响应
- 检查数据源连接是否正常
- 确认配置文件格式是否正确
- 验证输出目录权限是否充足

### 2. 预警过于频繁
- 检查预警阈值是否过于严格
- 考虑增加检查间隔
- 调整预警级别阈值

### 3. 可视化图表不生成
- 确认matplotlib环境配置
- 检查输出目录权限
- 验证可视化开关是否打开

## 维护与更新

### 1. 日志管理
- 定期清理旧的日志文件
- 备份重要的监控记录
- 分析历史数据优化预警规则

### 2. 配置更新
- 记录有效的配置调整
- 建立配置版本控制
- 保存不同场景的配置模板

## 贡献指南

欢迎提供改进建议和代码贡献：
1. Fork 项目仓库
2. 创建特性分支
3. 提交改进代码
4. 发起合并请求

## 许可证

MIT License

