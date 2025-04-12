# README - 统计建模竞赛项目
## 项目概述
本项目是东南大学"友宏先锋队"参加统计建模竞赛的代码和数据资料库。项目主要研究语言模型在连续生成过程中的性能变化（模型崩溃现象），通过多代迭代训练和生成实验，分析模型在不同代数的表现指标。
## 文件结构
```plainText
code&data/
├── Data/                          # 实验数据目录
│   ├── model_collapse_full_metrics.csv       # 完整实验指标数据
│   ├── model_collapse_metrics_50_percent.csv # 50%混合数据实验指标
│   ├── model_collapse_metrics_90_percent.csv # 90%混合数据实验指标
│   ├── dispersion_trend_data.csv   # 离散度趋势数据
│   └── dispersion_metrics.csv     # 离散度指标数据
├── calculate_metrics.py           # 指标计算脚本
├── dataset.py                     # 数据集处理模块
├── main.py                        # 主程序入口
├── run_iterations.py              # 迭代训练控制脚本
└── run_mixed_iterations.py        # 混合数据策略迭代脚本
```
## 主要功能模块
### 1. 模型训练与生成
main.py: 主程序入口，包含模型训练、评估和生成功能
支持多种运行模式：
- 纯训练模式
- 纯生成模式
- 训练+生成混合模式
### 2. 迭代实验控制
- run_iterations.py: 标准迭代实验控制脚本
- run_mixed_iterations.py: 支持混合原始数据和生成数据的迭代实验
### 3. 指标计算
calculate_metrics.py: 计算模型生成文本的多种评估指标
- 困惑度(Perplexity)
- 3-gram多样性
- 高频词比例
- 熵(Entropy)
- ROUGE分数
- METEOR分数
## 数据说明
### 实验指标数据
model_collapse_*.csv: 记录不同代数下的模型性能指标
列说明：
- Generation: 代数
- Run: 运行次数
- Perplexity: 困惑度
- 3gram_Diversity: 3-gram多样性
- HighFreq_Ratio: 高频词比例
- Entropy: 信息熵
- Rouge-1/2/L: ROUGE分数
- METEOR: METEOR分数
### 离散度数据
dispersion_*.csv: 记录文本离散度相关指标
包含各代数的均值、标准差、变异系数等统计量
## 使用说明
### 运行迭代实验
```
python run_iterations.py \
    --model_tag facebook/opt-125m \
    --num_iterations 12 \
    --max_epochs 6 \
    --batch_size 256
```
### 计算生成文本指标
```
python calculate_metrics.py \
    --generation_file generations/gen1_data.pkl \
    --model_tag facebook/opt-125m \
    --generation 1 \
    --run 1
```
## 环境要求
- Python 3.8+
- PyTorch 1.12+
- Transformers 库
- NLTK
- Pandas
## 团队信息
- 团队名称：友宏先锋队
- 所属单位：东南大学
- 竞赛名称：统计建模竞赛
## 注意事项
- 实验数据较大，建议在GPU服务器上运行
- 不同实验配置请修改对应参数
- 生成数据前请确保有足够的存储空间 
