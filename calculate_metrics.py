import os
import argparse
import pickle
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from datetime import datetime
from collections import Counter
import nltk
from nltk.util import ngrams
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
import platform
import math
from transformers import AutoTokenizer, AutoModelForCausalLM

# 确保NLTK资源已下载
try:
    nltk.data.find('wordnet')
except LookupError:
    nltk.download('wordnet')
    nltk.download('punkt')

# 根据操作系统设置基础目录
    base_dir = '/root/autodl-tmp/model_collapse_analysis'


def calculate_perplexity(model, tokenizer, texts, device='cuda'):
    """计算文本的困惑度"""
    model.eval()
    total_loss = 0
    total_length = 0
    
    with torch.no_grad():
        for text in tqdm(texts, desc="计算困惑度"):
            inputs = tokenizer(text, return_tensors="pt").to(device)
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss.item()
            
            # 计算非填充标记的数量
            length = torch.sum(inputs["attention_mask"]).item()
            
            total_loss += loss * length
            total_length += length
    
    # 计算平均损失并转换为困惑度
    avg_loss = total_loss / total_length
    perplexity = math.exp(avg_loss)
    
    return perplexity

def calculate_ngram_diversity(texts, n=3):
    """计算n-gram多样性（不同n-gram的数量）"""
    all_ngrams = []
    
    for text in tqdm(texts, desc=f"计算{n}-gram多样性"):
        # 分词
        tokens = nltk.word_tokenize(text.lower())
        # 生成n-grams
        text_ngrams = list(ngrams(tokens, n))
        all_ngrams.extend(text_ngrams)
    
    # 计算不同n-gram的数量
    unique_ngrams = set(all_ngrams)
    return len(unique_ngrams)

def calculate_high_freq_ratio(texts, threshold=0.01):
    """计算高频词比例（出现频率超过阈值的词占比）"""
    # 分词并统计词频
    all_tokens = []
    for text in tqdm(texts, desc="计算高频词比例"):
        tokens = nltk.word_tokenize(text.lower())
        all_tokens.extend(tokens)
    
    # 计算词频
    token_counter = Counter(all_tokens)
    total_tokens = len(all_tokens)
    
    # 计算高频词（出现频率超过阈值的词）
    high_freq_tokens = [token for token, count in token_counter.items() 
                        if count / total_tokens > threshold]
    
    # 计算高频词在总词汇中的比例
    high_freq_count = sum(token_counter[token] for token in high_freq_tokens)
    high_freq_ratio = high_freq_count / total_tokens
    
    return high_freq_ratio

def calculate_entropy(texts):
    """计算文本的熵（信息熵）"""
    # 分词并统计词频
    all_tokens = []
    for text in tqdm(texts, desc="计算熵"):
        tokens = nltk.word_tokenize(text.lower())
        all_tokens.extend(tokens)
    
    # 计算词频
    token_counter = Counter(all_tokens)
    total_tokens = len(all_tokens)
    
    # 计算每个词的概率
    probabilities = [count / total_tokens for count in token_counter.values()]
    
    # 计算熵
    entropy = -sum(p * math.log2(p) for p in probabilities)
    
    return entropy

def calculate_rouge_scores(generated_texts, reference_texts):
    """计算ROUGE分数"""
    # 初始化ROUGE计算器
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []
    
    for gen_text, ref_text in tqdm(zip(generated_texts, reference_texts), desc="计算ROUGE分数", total=len(generated_texts)):
        # 计算ROUGE分数
        scores = scorer.score(ref_text, gen_text)
        
        # 提取F1分数
        rouge1_scores.append(scores['rouge1'].fmeasure)
        rouge2_scores.append(scores['rouge2'].fmeasure)
        rougeL_scores.append(scores['rougeL'].fmeasure)
    
    # 计算平均分数
    avg_rouge1 = sum(rouge1_scores) / len(rouge1_scores)
    avg_rouge2 = sum(rouge2_scores) / len(rouge2_scores)
    avg_rougeL = sum(rougeL_scores) / len(rougeL_scores)
    
    return {
        'Rouge-1': avg_rouge1,
        'Rouge-2': avg_rouge2,
        'Rouge-L': avg_rougeL
    }

def calculate_meteor_score(generated_texts, reference_texts, sample_size=1000):
    """计算METEOR分数（使用采样以提高效率）"""
    # 如果文本数量过多，进行采样
    if len(generated_texts) > sample_size:
        indices = np.random.choice(len(generated_texts), sample_size, replace=False)
        gen_sample = [generated_texts[i] for i in indices]
        ref_sample = [reference_texts[i] for i in indices]
    else:
        gen_sample = generated_texts
        ref_sample = reference_texts
    
    meteor_scores = []
    
    for gen_text, ref_text in tqdm(zip(gen_sample, ref_sample), desc="计算METEOR分数", total=len(gen_sample)):
        # 分词
        gen_tokens = nltk.word_tokenize(gen_text.lower())
        ref_tokens = nltk.word_tokenize(ref_text.lower())
        
        # 计算METEOR分数
        score = meteor_score([ref_tokens], gen_tokens)
        meteor_scores.append(score)
    
    # 计算平均分数
    avg_meteor = sum(meteor_scores) / len(meteor_scores)
    
    return avg_meteor

def main():
    # 参数解析
    parser = argparse.ArgumentParser(description='计算模型崩溃指标')
    parser.add_argument('--generation_file', type=str, required=True,
                        help='生成文本的文件路径')
    parser.add_argument('--reference_file', type=str, default=None,
                        help='参考文本的文件路径（用于计算ROUGE和METEOR）')
    parser.add_argument('--model_tag', type=str, default='facebook/opt-125m',
                        help='用于计算困惑度的模型标签')
    parser.add_argument('--output_csv', type=str, default=os.path.join(base_dir, 'metrics', 'model_collapse_metrics.csv'),
                        help='输出CSV文件的路径')
    parser.add_argument('--generation', type=int, default=0,
                        help='当前生成的代数')
    parser.add_argument('--run', type=int, default=1,
                        help='当前运行的次数')
    parser.add_argument('--skip_perplexity', action='store_true',
                        help='跳过困惑度计算（计算较慢）')
    parser.add_argument('--sample_size', type=int, default=1000,
                        help='用于计算指标的样本数量（设为-1使用全部样本）')
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    
    # 加载生成的文本
    print(f"加载生成文本: {args.generation_file}")
    with open(args.generation_file, 'rb') as f:
        generated_data = pickle.load(f)
    
    # 提取文本
    if isinstance(generated_data, list) and isinstance(generated_data[0], dict) and 'generated_text' in generated_data[0]:
        generated_texts = [item['generated_text'] for item in generated_data]
        input_texts = [item.get('input_text', '') for item in generated_data]
    else:
        raise ValueError("不支持的生成数据格式，请确保数据包含'generated_text'字段")
    
    # 采样（如果需要）
    if args.sample_size > 0 and args.sample_size < len(generated_texts):
        print(f"从{len(generated_texts)}个样本中采样{args.sample_size}个进行指标计算")
        indices = np.random.choice(len(generated_texts), args.sample_size, replace=False)
        generated_texts_sample = [generated_texts[i] for i in indices]
        input_texts_sample = [input_texts[i] for i in indices]
    else:
        generated_texts_sample = generated_texts
        input_texts_sample = input_texts
    
    print(f"计算指标使用{len(generated_texts_sample)}个样本")
    
    # 初始化指标字典
    metrics = {
        'Generation': args.generation,
        'Run': args.run
    }
    
    # 计算困惑度（如果不跳过）
    if not args.skip_perplexity:
        print("加载模型用于计算困惑度...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 加载模型和tokenizer
        local_model_path = os.path.join(base_dir, f"model_cache_dir/{args.model_tag.split('/')[-1]}")
        if os.path.exists(local_model_path):
            tokenizer = AutoTokenizer.from_pretrained(local_model_path)
            model = AutoModelForCausalLM.from_pretrained(local_model_path).to(device)
        else:
            tokenizer = AutoTokenizer.from_pretrained(args.model_tag)
            model = AutoModelForCausalLM.from_pretrained(args.model_tag).to(device)
        
        # 计算困惑度
        perplexity = calculate_perplexity(model, tokenizer, generated_texts_sample[:min(100, len(generated_texts_sample))], device)
        metrics['Perplexity'] = perplexity
        print(f"困惑度: {perplexity:.4f}")
        
        # 释放GPU内存
        if device == 'cuda':
            del model
            torch.cuda.empty_cache()
    
    # 计算3-gram多样性
    diversity = calculate_ngram_diversity(generated_texts_sample, n=3)
    metrics['3gram_Diversity'] = diversity
    print(f"3-gram多样性: {diversity}")
    
    # 计算高频词比例
    high_freq_ratio = calculate_high_freq_ratio(generated_texts_sample)
    metrics['HighFreq_Ratio'] = high_freq_ratio
    print(f"高频词比例: {high_freq_ratio:.4f}")
    
    # 计算熵
    entropy = calculate_entropy(generated_texts_sample)
    metrics['Entropy'] = entropy
    print(f"熵: {entropy:.4f}")
    
    # 计算ROUGE和METEOR分数
    # 如果提供了参考文件，使用参考文件；否则使用输入文本作为参考
    if args.reference_file:
        print(f"加载参考文本: {args.reference_file}")
        with open(args.reference_file, 'rb') as f:
            reference_data = pickle.load(f)
        
        if isinstance(reference_data, list) and isinstance(reference_data[0], dict) and 'generated_text' in reference_data[0]:
            reference_texts = [item['generated_text'] for item in reference_data]
        else:
            raise ValueError("不支持的参考数据格式，请确保数据包含'generated_text'字段")
        
        # 采样参考文本（与生成文本样本大小相同）
        if len(reference_texts) > len(generated_texts_sample):
            indices = np.random.choice(len(reference_texts), len(generated_texts_sample), replace=False)
            reference_texts_sample = [reference_texts[i] for i in indices]
        else:
            reference_texts_sample = reference_texts
    else:
        # 使用输入文本作为参考
        reference_texts_sample = input_texts_sample
    
    # 确保参考文本和生成文本数量一致
    min_len = min(len(generated_texts_sample), len(reference_texts_sample))
    generated_texts_sample = generated_texts_sample[:min_len]
    reference_texts_sample = reference_texts_sample[:min_len]
    
    # 计算ROUGE分数
    rouge_scores = calculate_rouge_scores(generated_texts_sample, reference_texts_sample)
    metrics.update(rouge_scores)
    print(f"ROUGE-1: {rouge_scores['Rouge-1']:.4f}")
    print(f"ROUGE-2: {rouge_scores['Rouge-2']:.4f}")
    print(f"ROUGE-L: {rouge_scores['Rouge-L']:.4f}")
    
    # 计算METEOR分数
    meteor = calculate_meteor_score(generated_texts_sample, reference_texts_sample)
    metrics['METEOR'] = meteor
    print(f"METEOR: {meteor:.4f}")
    
    # 保存指标到CSV
    df = pd.DataFrame([metrics])
    
    # 检查文件是否存在，决定是否需要写入表头
    file_exists = os.path.isfile(args.output_csv)
    
    # 写入CSV
    df.to_csv(args.output_csv, mode='a', header=not file_exists, index=False)
    print(f"指标已保存到: {args.output_csv}")
    
    # 同时保存为单独的CSV（包含时间戳）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.dirname(args.output_csv)
    single_output = os.path.join(output_dir, f"metrics_gen{args.generation}_run{args.run}_{timestamp}.csv")
    df.to_csv(single_output, index=False)
    print(f"单独指标文件已保存到: {single_output}")

if __name__ == "__main__":
    main()