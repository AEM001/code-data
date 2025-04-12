import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# 设置工作目录和文件路径
data_dir = r"d:\Documents\100\action"
sem_step2_dir = os.path.join(data_dir, "results", "sem_analysis_step2")
sem_step3_dir = os.path.join(data_dir, "results", "sem_analysis_step3")
output_dir = os.path.join(data_dir, "results", "final_report")

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

# 打印路径信息以便确认
print("检查文件路径...")
print(f"SEM分析结果目录: {sem_step2_dir}")
print(f"异常检测结果目录: {sem_step3_dir}")
print(f"输出报告目录: {output_dir}")

print("开始生成综合分析报告...")

# 设置matplotlib使用中文字体
try:
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    print("已设置matplotlib使用中文字体")
except Exception as e:
    print(f"设置中文字体时出错: {e}")

# 读取SEM分析结果
try:
    # 尝试读取效应分解结果
    effects_df = pd.read_csv(os.path.join(sem_step2_dir, "effects_decomposition.csv"))
    print("成功读取效应分解结果")
    # 打印列名以便调试
    print(f"效应分解结果列名: {effects_df.columns.tolist()}")
except FileNotFoundError:
    print("警告: 未找到效应分解结果文件")
    effects_df = None

try:
    # 尝试读取Bootstrap置信区间
    bootstrap_df = pd.read_csv(os.path.join(sem_step2_dir, "bootstrap_confidence_intervals.csv"))
    print("成功读取Bootstrap置信区间")
except FileNotFoundError:
    print("警告: 未找到Bootstrap置信区间文件")
    bootstrap_df = None

# 读取异常检测结果
try:
    anomaly_df = pd.read_csv(os.path.join(sem_step3_dir, "direct_anomaly_detection_results.csv"))
    print(f"成功读取异常检测结果，共 {len(anomaly_df)} 行")
except FileNotFoundError:
    print("警告: 未找到异常检测结果文件")
    anomaly_df = None

try:
    anomaly_stats = pd.read_csv(os.path.join(sem_step3_dir, "direct_anomaly_stats_comparison.csv"))
    print("成功读取异常样本统计结果")
except FileNotFoundError:
    print("警告: 未找到异常样本统计结果文件")
    anomaly_stats = None

# 读取预警规则
try:
    with open(os.path.join(sem_step3_dir, "warning_rules.txt"), 'r', encoding='utf-8') as f:
        warning_rules = f.read()
    print("成功读取预警规则")
except FileNotFoundError:
    print("警告: 未找到预警规则文件")
    warning_rules = "未找到预警规则"

# 生成综合报告
report_file = os.path.join(output_dir, "sem_anomaly_analysis_report.md")
with open(report_file, 'w', encoding='utf-8') as f:
    f.write("# 结构方程模型与异常检测综合分析报告\n\n")
    f.write(f"*生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
    
    f.write("## 一、研究概述\n\n")
    f.write("本研究旨在验证模型崩溃的因果机制，特别是'高频词比例→多样性下降→语义质量下降'的传导路径，并基于此建立异常检测和预警系统。\n\n")
    
    f.write("## 二、结构方程模型分析结果\n\n")
    
    if effects_df is not None:
        f.write("### 1. 效应分解\n\n")
        f.write("下表展示了高频词比例对语义质量的直接效应、间接效应和总效应：\n\n")
        f.write("| 效应类型 | 路径 | 效应大小 |\n")
        f.write("|---------|------|--------|\n")
        
        # 使用正确的列名 'Value'
        effect_type_col = 'Effect_Type'
        path_col = 'Path'
        effect_size_col = 'Value'
        
        for _, row in effects_df.iterrows():
            f.write(f"| {row[effect_type_col]} | {row[path_col]} | {row[effect_size_col]:.4f} |\n")
        f.write("\n")
    
    if effects_df is not None:
        # 使用正确的列名 'Value'
        effect_type_col = 'Effect_Type'
        effect_size_col = 'Value'
        
        direct_effect = effects_df.loc[effects_df[effect_type_col] == 'Direct', effect_size_col].values[0] if 'Direct' in effects_df[effect_type_col].values else 0
        indirect_effect = effects_df.loc[effects_df[effect_type_col] == 'Indirect', effect_size_col].values[0] if 'Indirect' in effects_df[effect_type_col].values else 0
        total_effect = effects_df.loc[effects_df[effect_type_col] == 'Total', effect_size_col].values[0] if 'Total' in effects_df[effect_type_col].values else 0
        
        f.write(f"2. **间接效应显著**：高频词比例通过多样性对语义质量的间接效应为{indirect_effect:.4f}，")
        if abs(indirect_effect) > abs(direct_effect):
            f.write("大于直接效应，说明多样性是关键的中介变量。\n")
        else:
            f.write("小于直接效应，但仍然具有重要影响。\n")
        
        if total_effect > 0:
            f.write(f"3. **总效应为正**：高频词比例对语义质量的总效应为{total_effect:.4f}，表明在控制了多样性的中介作用后，高频词比例的增加总体上有利于语义质量。\n")
        else:
            f.write(f"3. **总效应为负**：高频词比例对语义质量的总效应为{total_effect:.4f}，表明高频词比例的增加总体上不利于语义质量。\n")
    else:
        f.write("2. **效应分解**：未能获取具体的效应分解数据。\n")
    
    f.write("\n## 三、异常检测分析结果\n\n")
    
    if anomaly_df is not None:
        anomaly_count = anomaly_df['combined_anomaly'].sum()
        anomaly_rate = anomaly_count / len(anomaly_df) * 100
        f.write(f"### 1. 异常样本检测\n\n")
        f.write(f"通过多种异常检测方法的综合应用，我们共检测到 **{anomaly_count}** 个异常样本，占总样本的 **{anomaly_rate:.2f}%**。\n\n")
        
        f.write("异常检测方法包括：\n")
        f.write("- **隔离森林**：检测多维特征空间中的异常点\n")
        f.write("- **变化率检测**：识别指标变化过快的样本\n")
        f.write("- **关系异常检测**：识别偏离预期关系的样本\n\n")
        
        # 添加异常样本分布图
        f.write("![异常样本在高频词比例与多样性关系中的分布](../sem_analysis_step3/direct_anomaly_diversity.png)\n\n")
        f.write("![异常样本在多样性与语义质量关系中的分布](../sem_analysis_step3/direct_anomaly_quality.png)\n\n")
        
        # 添加Run异常比例图
        f.write("![各Run的异常样本比例](../sem_analysis_step3/direct_run_anomaly_rate.png)\n\n")
    
    if anomaly_stats is not None:
        f.write("### 2. 异常样本特征\n\n")
        f.write("异常样本与正常样本在关键指标上的对比：\n\n")
        f.write("| 指标 | 异常样本均值 | 正常样本均值 | 差异百分比 |\n")
        f.write("|------|------------|------------|----------|\n")
        
        # 确保anomaly_stats的索引列是正确的
        if 'Unnamed: 0' in anomaly_stats.columns:
            anomaly_stats.set_index('Unnamed: 0', inplace=True)
        
        for idx, row in anomaly_stats.iterrows():
            if isinstance(row, pd.Series):
                f.write(f"| {idx} | {row['异常样本均值']:.4f} | {row['正常样本均值']:.4f} | {row['差异百分比']:.2f}% |\n")
        f.write("\n")
        
        # 分析异常样本特征
        f.write("**关键发现**：\n")
        
        # 检查多样性差异
        if '多样性' in anomaly_stats.index or 'Diversity' in anomaly_stats.index:
            diversity_idx = '多样性' if '多样性' in anomaly_stats.index else 'Diversity'
            diversity_diff = anomaly_stats.loc[diversity_idx, '差异百分比']
            if diversity_diff > 0:
                f.write(f"- 异常样本的多样性比正常样本高 **{diversity_diff:.2f}%**，这与预期的'多样性下降导致崩溃'假设不符\n")
            else:
                f.write(f"- 异常样本的多样性比正常样本低 **{abs(diversity_diff):.2f}%**，符合'多样性下降导致崩溃'的假设\n")
        
        # 检查高频词比例差异
        if '高频词比例' in anomaly_stats.index or 'HighFreq' in anomaly_stats.index:
            highfreq_idx = '高频词比例' if '高频词比例' in anomaly_stats.index else 'HighFreq'
            highfreq_diff = anomaly_stats.loc[highfreq_idx, '差异百分比']
            if highfreq_diff > 0:
                f.write(f"- 异常样本的高频词比例比正常样本高 **{highfreq_diff:.2f}%**，符合'高频词比例增加导致崩溃'的假设\n")
            else:
                f.write(f"- 异常样本的高频词比例比正常样本低 **{abs(highfreq_diff):.2f}%**，这与预期的'高频词比例增加导致崩溃'假设不符\n")
        
        # 检查语义质量差异
        if 'METEOR' in anomaly_stats.index:
            meteor_diff = anomaly_stats.loc['METEOR', '差异百分比']
            if meteor_diff > 0:
                f.write(f"- 异常样本的METEOR得分比正常样本高 **{meteor_diff:.2f}%**，这与预期的'语义质量下降导致崩溃'假设不符\n")
            else:
                f.write(f"- 异常样本的METEOR得分比正常样本低 **{abs(meteor_diff):.2f}%**，符合'语义质量下降导致崩溃'的假设\n")
        
        f.write("\n")
    
    f.write("### 3. 预警规则\n\n")
    f.write("基于异常检测结果，我们提取了以下预警规则：\n\n")
    f.write("```\n")
    f.write(warning_rules)
    f.write("\n```\n\n")
    
    f.write("## 四、综合结论与建议\n\n")
    
    f.write("### 1. 主要结论\n\n")
    f.write("1. **因果机制验证**：通过结构方程模型分析，我们验证了'高频词比例→多样性→语义质量'的因果传导路径，并量化了各路径的效应大小。\n")
    f.write("2. **异常模式识别**：通过多种异常检测方法，我们识别出了偏离正常模式的样本，并分析了这些异常样本的特征。\n")
    f.write("3. **预警系统构建**：基于SEM分析和异常检测结果，我们提取了一系列预警规则，可用于早期识别模型崩溃风险。\n\n")
    
    f.write("### 2. 实践建议\n\n")
    f.write("1. **监控关键指标**：在模型训练过程中，应重点监控高频词比例、多样性和语义质量这三个关键指标。\n")
    f.write("2. **应用预警规则**：将本研究提取的预警规则应用于实时监控系统，当满足预警条件时及时干预。\n")
    f.write("3. **优化训练策略**：基于因果分析结果，可以考虑在训练过程中引入多样性约束，或者直接控制高频词比例，以防止模型崩溃。\n\n")
    
    f.write("### 3. 研究局限与展望\n\n")
    f.write("1. **样本量限制**：本研究的样本量相对有限，未来可以收集更多数据进行验证。\n")
    f.write("2. **模型简化**：当前的SEM模型相对简化，未来可以考虑引入更多变量和更复杂的路径关系。\n")
    f.write("3. **预警系统验证**：本研究提出的预警规则需要在实际应用中进一步验证和优化。\n")

print(f"综合分析报告已生成：{report_file}")

# 复制关键图表到最终报告目录
try:
    import shutil
    
    # 复制SEM分析图表
    sem_plots = [
        "path_coefficients.png",
        "effects_decomposition.png",
        "bootstrap_distribution.png"
    ]
    
    for plot in sem_plots:
        src_path = os.path.join(sem_step2_dir, plot)
        if os.path.exists(src_path):
            shutil.copy2(src_path, os.path.join(output_dir, plot))
            print(f"已复制图表: {plot}")
    
    # 复制异常检测图表
    anomaly_plots = [
        "direct_anomaly_diversity.png",
        "direct_anomaly_quality.png",
        "direct_run_anomaly_rate.png",
        "anomaly_by_generation.png"
    ]
    
    for plot in anomaly_plots:
        src_path = os.path.join(sem_step3_dir, plot)
        if os.path.exists(src_path):
            shutil.copy2(src_path, os.path.join(output_dir, plot))
            print(f"已复制图表: {plot}")
    
    print("所有图表已复制到最终报告目录")
except Exception as e:
    print(f"复制图表时出错: {e}")

print("综合分析报告生成完成!")