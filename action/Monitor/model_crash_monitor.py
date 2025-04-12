import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import os
import json
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

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
            'monitoring': {
                'check_interval': 10,  # 检查间隔（秒）
                'history_window': 20,  # 历史窗口大小
                'warning_level_thresholds': [1, 2, 3]  # 不同预警级别的规则触发数量
            },
            'output': {
                'log_dir': r"d:\Documents\100\action\results\monitor_logs",
                'visualization': True
            }
        }
        
        # 加载自定义配置
        if config_path:
            try:
                with open(config_path, 'r') as f:
                    custom_config = json.load(f)
                    self.config.update(custom_config)
            except Exception as e:
                print(f"加载配置文件失败: {e}，使用默认配置")
        
        # 创建输出目录
        os.makedirs(self.config['output']['log_dir'], exist_ok=True)
        
        # 初始化数据存储
        self.history = pd.DataFrame()
        self.warning_history = []
        self.current_status = "正常"
        
        # 初始化模型
        self.scaler = StandardScaler()
        self.iso_forest = IsolationForest(contamination=0.1, random_state=42)
        
        print("模型崩溃监控工具初始化完成")
        
    def add_observation(self, data):
        """添加新的观测数据"""
        if isinstance(data, dict):
            data = pd.DataFrame([data])
        
        # 确保必要的字段存在
        required_fields = ['Run', 'Generation', 'Perplexity', 'Diversity', 'HighFreq', 'METEOR']
        for field in required_fields:
            if field not in data.columns:
                raise ValueError(f"缺少必要字段: {field}")
        
        # 添加时间戳
        data['Timestamp'] = datetime.now()
        
        # 合并到历史数据
        self.history = pd.concat([self.history, data], ignore_index=True)
        
        # 保持历史窗口大小
        window_size = self.config['monitoring']['history_window']
        if len(self.history) > window_size:
            self.history = self.history.iloc[-window_size:]
        
        # 计算变化率
        if len(self.history) > 1:
            for col in ['Diversity', 'HighFreq', 'METEOR']:
                self.history[f'{col}_change'] = self.history.groupby('Run')[col].pct_change()
        
        print(f"添加新观测: Run={data['Run'].values[0]}, Generation={data['Generation'].values[0]}")
        
    def check_warnings(self):
        """检查是否触发预警规则"""
        if len(self.history) < 2:
            return []
        
        # 获取最新观测
        latest = self.history.iloc[-1]
        
        # 应用预警规则
        warnings = []
        rules = self.config['warning_rules']
        
        # 规则1: 高频词比例超过阈值
        if latest['HighFreq'] > rules['highfreq_threshold']:
            warnings.append({
                'rule': '规则1',
                'description': f"高频词比例 ({latest['HighFreq']:.4f}) 超过阈值 ({rules['highfreq_threshold']:.4f})",
                'severity': 'medium'
            })
        
        # 规则2: 多样性低于阈值
        if latest['Diversity'] < rules['diversity_threshold']:
            warnings.append({
                'rule': '规则2',
                'description': f"多样性 ({latest['Diversity']:.2f}) 低于阈值 ({rules['diversity_threshold']:.2f})",
                'severity': 'high'
            })
        
        # 规则3: METEOR得分低于阈值
        if latest['METEOR'] < rules['meteor_threshold']:
            warnings.append({
                'rule': '规则3',
                'description': f"METEOR得分 ({latest['METEOR']:.4f}) 低于阈值 ({rules['meteor_threshold']:.4f})",
                'severity': 'high'
            })
        
        # 规则4: 多样性变化率异常
        if 'Diversity_change' in latest and not pd.isna(latest['Diversity_change']):
            if abs(latest['Diversity_change']) > rules['diversity_change_threshold']:
                warnings.append({
                    'rule': '规则4',
                    'description': f"多样性变化率 ({latest['Diversity_change']:.4f}) 超过阈值 (±{rules['diversity_change_threshold']:.4f})",
                    'severity': 'medium'
                })
        
        # 规则5: 高频词比例变化率异常
        if 'HighFreq_change' in latest and not pd.isna(latest['HighFreq_change']):
            if abs(latest['HighFreq_change']) > rules['highfreq_change_threshold']:
                warnings.append({
                    'rule': '规则5',
                    'description': f"高频词比例变化率 ({latest['HighFreq_change']:.4f}) 超过阈值 (±{rules['highfreq_change_threshold']:.4f})",
                    'severity': 'medium'
                })
        
        # 规则6: 如果有足够的数据，检查关系异常
        if len(self.history) >= 5:
            # 标准化数据
            if len(self.history) >= 10:
                scaled_data = self.scaler.fit_transform(self.history[['HighFreq', 'Diversity']])
                self.history['HighFreq_scaled'] = scaled_data[:, 0]
                self.history['Diversity_scaled'] = scaled_data[:, 1]
                
                # 拟合高频词比例与多样性的关系
                from sklearn.linear_model import LinearRegression
                reg = LinearRegression()
                reg.fit(self.history[['HighFreq_scaled']], self.history['Diversity_scaled'])
                expected = reg.predict(self.history[['HighFreq_scaled']])
                self.history['relation_residual'] = self.history['Diversity_scaled'] - expected
                
                # 检查最新观测的残差
                latest_residual = self.history.iloc[-1]['relation_residual']
                if abs(latest_residual) > rules['relation_residual_threshold']:
                    warnings.append({
                        'rule': '规则6',
                        'description': f"高频词比例与多样性关系异常 (残差={latest_residual:.4f}, 阈值=±{rules['relation_residual_threshold']:.4f})",
                        'severity': 'high'
                    })
        
        # 记录预警
        if warnings:
            warning_record = {
                'timestamp': datetime.now(),
                'run': latest['Run'],
                'generation': latest['Generation'],
                'warnings': warnings,
                'warning_count': len(warnings)
            }
            self.warning_history.append(warning_record)
            
            # 更新状态
            warning_count = len(warnings)
            thresholds = self.config['monitoring']['warning_level_thresholds']
            if warning_count >= thresholds[2]:
                self.current_status = "严重警告"
            elif warning_count >= thresholds[1]:
                self.current_status = "警告"
            elif warning_count >= thresholds[0]:
                self.current_status = "提示"
            else:
                self.current_status = "正常"
        
        return warnings
    
    def get_intervention_suggestions(self, warnings):
        """基于预警生成干预建议"""
        if not warnings:
            return []
        
        suggestions = []
        
        # 高频词比例相关建议
        if any(w['rule'] in ['规则1', '规则5'] for w in warnings):
            suggestions.append({
                'target': '高频词比例',
                'suggestion': '考虑增加词汇惩罚因子，或调整采样温度以减少高频词的使用',
                'priority': 'high'
            })
        
        # 多样性相关建议
        if any(w['rule'] in ['规则2', '规则4'] for w in warnings):
            suggestions.append({
                'target': '多样性',
                'suggestion': '增加采样温度或使用top-k/top-p采样策略以提高输出多样性',
                'priority': 'high'
            })
        
        # 语义质量相关建议
        if any(w['rule'] == '规则3' for w in warnings):
            suggestions.append({
                'target': '语义质量',
                'suggestion': '考虑降低采样温度以提高输出的连贯性，或增加语义相关的奖励信号',
                'priority': 'medium'
            })
        
        # 关系异常相关建议
        if any(w['rule'] == '规则6' for w in warnings):
            suggestions.append({
                'target': '指标关系',
                'suggestion': '检查模型训练过程中的梯度更新是否异常，考虑降低学习率或增加正则化',
                'priority': 'high'
            })
        
        # 综合建议
        if len(warnings) >= 3:
            suggestions.append({
                'target': '综合',
                'suggestion': '建议暂停当前训练，回滚到上一个稳定检查点，并调整多个超参数后重新开始',
                'priority': 'critical'
            })
        
        return suggestions
    
    def visualize_status(self):
        """可视化当前状态"""
        if not self.config['output']['visualization'] or len(self.history) < 5:
            return
        
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. 关键指标趋势
        for i, col in enumerate(['HighFreq', 'Diversity', 'METEOR']):
            ax = axes[0, 0]
            ax.plot(self.history['Generation'], self.history[col], marker='o', label=col)
        
        axes[0, 0].set_title('关键指标趋势')
        axes[0, 0].set_xlabel('生成轮次')
        axes[0, 0].legend()
        axes[0, 0].grid(True, linestyle='--', alpha=0.7)
        
        # 2. 高频词比例与多样性关系
        if 'HighFreq_scaled' in self.history.columns and 'Diversity_scaled' in self.history.columns:
            ax = axes[0, 1]
            scatter = ax.scatter(
                self.history['HighFreq_scaled'], 
                self.history['Diversity_scaled'],
                c=self.history['Generation'], 
                cmap='viridis', 
                s=50, 
                alpha=0.7
            )
            ax.set_title('高频词比例与多样性关系')
            ax.set_xlabel('高频词比例 (标准化)')
            ax.set_ylabel('多样性 (标准化)')
            fig.colorbar(scatter, ax=ax, label='生成轮次')
            ax.grid(True, linestyle='--', alpha=0.7)
        
        # 3. 变化率监控
        if 'Diversity_change' in self.history.columns and 'HighFreq_change' in self.history.columns:
            ax = axes[1, 0]
            ax.plot(self.history['Generation'], self.history['Diversity_change'], marker='o', label='多样性变化率')
            ax.plot(self.history['Generation'], self.history['HighFreq_change'], marker='s', label='高频词比例变化率')
            
            # 添加阈值线
            rules = self.config['warning_rules']
            ax.axhline(y=rules['diversity_change_threshold'], color='r', linestyle='--', alpha=0.5)
            ax.axhline(y=-rules['diversity_change_threshold'], color='r', linestyle='--', alpha=0.5)
            ax.axhline(y=rules['highfreq_change_threshold'], color='orange', linestyle='--', alpha=0.5)
            ax.axhline(y=-rules['highfreq_change_threshold'], color='orange', linestyle='--', alpha=0.5)
            
            ax.set_title('指标变化率监控')
            ax.set_xlabel('生成轮次')
            ax.set_ylabel('变化率')
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.7)
        
        # 4. 预警状态
        ax = axes[1, 1]
        warning_counts = [0] * len(self.history)
        for i, record in enumerate(self.warning_history):
            gen_idx = self.history[self.history['Generation'] == record['generation']].index
            if len(gen_idx) > 0:
                warning_counts[gen_idx[0]] = record['warning_count']
        
        bars = ax.bar(self.history['Generation'], warning_counts, color='skyblue')
        
        # 为不同预警级别设置不同颜色
        thresholds = self.config['monitoring']['warning_level_thresholds']
        for i, count in enumerate(warning_counts):
            if count >= thresholds[2]:
                bars[i].set_color('red')
            elif count >= thresholds[1]:
                bars[i].set_color('orange')
            elif count >= thresholds[0]:
                bars[i].set_color('yellow')
        
        ax.set_title('预警状态')
        ax.set_xlabel('生成轮次')
        ax.set_ylabel('触发的预警规则数')
        ax.set_ylim(0, max(6, max(warning_counts) + 1))
        
        # 添加当前状态文本
        status_colors = {
            "正常": "green",
            "提示": "blue",
            "警告": "orange",
            "严重警告": "red"
        }
        fig.text(0.5, 0.01, f"当前状态: {self.current_status}", 
                 ha='center', fontsize=14, 
                 color=status_colors.get(self.current_status, 'black'),
                 bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
        
        plt.tight_layout()
        
        # 保存图表
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fig_path = os.path.join(self.config['output']['log_dir'], f"monitor_status_{timestamp}.png")
        plt.savefig(fig_path, dpi=300)
        plt.close()
        
        print(f"状态可视化已保存: {fig_path}")
    
    def log_status(self):
        """记录当前状态"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_file = os.path.join(self.config['output']['log_dir'], "monitor_log.txt")
        
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"\n===== {timestamp} =====\n")
            f.write(f"当前状态: {self.current_status}\n")
            
            if self.history.empty:
                f.write("无监控数据\n")
            else:
                latest = self.history.iloc[-1]
                f.write(f"最新观测: Run={latest['Run']}, Generation={latest['Generation']}\n")
                f.write(f"关键指标: HighFreq={latest['HighFreq']:.4f}, Diversity={latest['Diversity']:.2f}, METEOR={latest['METEOR']:.4f}\n")
            
            if self.warning_history and self.warning_history[-1]['timestamp'].strftime("%Y-%m-%d %H:%M:%S") == timestamp:
                warnings = self.warning_history[-1]['warnings']
                f.write(f"触发预警: {len(warnings)}个\n")
                for w in warnings:
                    f.write(f"- {w['rule']}: {w['description']} (严重程度: {w['severity']})\n")
                
                suggestions = self.get_intervention_suggestions(warnings)
                if suggestions:
                    f.write("干预建议:\n")
                    for s in suggestions:
                        f.write(f"- [{s['target']}] {s['suggestion']} (优先级: {s['priority']})\n")
            else:
                f.write("无预警\n")
        
        print(f"状态已记录到日志: {log_file}")
    
    def run_monitoring_cycle(self):
        """运行一个完整的监控周期"""
        warnings = self.check_warnings()
        if warnings:
            print(f"检测到 {len(warnings)} 个预警:")
            for w in warnings:
                print(f"- {w['rule']}: {w['description']}")
            
            suggestions = self.get_intervention_suggestions(warnings)
            print("干预建议:")
            for s in suggestions:
                print(f"- [{s['target']}] {s['suggestion']}")
        else:
            print("未检测到预警")
        
        self.visualize_status()
        self.log_status()
    
    def start_monitoring(self, data_source=None, interval=None):
        """启动持续监控
        
        Args:
            data_source: 数据源函数，每次调用返回新的观测数据
            interval: 监控间隔（秒），如果为None则使用配置中的值
        """
        if interval is None:
            interval = self.config['monitoring']['check_interval']
        
        print(f"开始持续监控，间隔 {interval} 秒...")
        
        try:
            while True:
                if data_source:
                    try:
                        new_data = data_source()
                        if new_data is not None:
                            self.add_observation(new_data)
                    except Exception as e:
                        print(f"获取数据时出错: {e}")
                
                if not self.history.empty:
                    self.run_monitoring_cycle()
                
                time.sleep(interval)
        except KeyboardInterrupt:
            print("监控已停止")
    
    def simulate_with_data(self, data_path):
        """使用CSV数据文件进行模拟监控"""
        try:
            data = pd.read_csv(data_path)
            print(f"加载模拟数据: {len(data)} 条记录")
            
            for i, row in data.iterrows():
                print(f"\n--- 模拟步骤 {i+1}/{len(data)} ---")
                self.add_observation(row.to_dict())
                self.run_monitoring_cycle()
                
                # 模拟间隔
                time.sleep(1)
            
            print("\n模拟监控完成")
        except Exception as e:
            print(f"模拟监控时出错: {e}")


# 示例用法
if __name__ == "__main__":
    # 初始化监控工具
    monitor = ModelCrashMonitor()
    
    # 选择操作模式
    import argparse
    parser = argparse.ArgumentParser(description='模型崩溃监控工具')
    parser.add_argument('--mode', type=str, choices=['simulate', 'monitor'], default='simulate',
                        help='运行模式: simulate=使用历史数据模拟, monitor=实时监控')
    parser.add_argument('--data', type=str, default=r"d:\Documents\100\action\results\sem_analysis_step3\direct_anomaly_detection_results.csv",
                        help='模拟模式下使用的数据文件路径')
    parser.add_argument('--interval', type=int, default=10,
                        help='监控模式下的检查间隔（秒）')
    args = parser.parse_args()
    
    if args.mode == 'simulate':
        print(f"使用数据文件进行模拟: {args.data}")
        monitor.simulate_with_data(args.data)
    else:
        print("启动实时监控模式")
        # 这里需要实现一个实时数据源
        def dummy_data_source():
            # 示例：随机生成数据
            import random
            return {
                'Run': 'realtime',
                'Generation': int(time.time()) % 100,
                'Perplexity': random.uniform(50, 100),
                'Diversity': random.uniform(30000, 70000),
                'HighFreq': random.uniform(0.5, 0.9),
                'METEOR': random.uniform(0.3, 0.6)
            }
        
        monitor.start_monitoring(data_source=dummy_data_source, interval=args.interval)