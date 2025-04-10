import argparse
import torch
import numpy as np
import random
import pytorch_lightning as pl
import pickle
import time
import os
#from torchmetrics import CatMetric, MetricCollection, MetricTracker

from transformers import AutoTokenizer
from transformers import AutoModel, AutoModelForSeq2SeqLM, AutoConfig, AutoModelForMaskedLM, AutoModelForCausalLM, default_data_collator
from pytorch_lightning.callbacks import ModelCheckpoint
from datetime import datetime
from datasets import load_from_disk  

from tqdm import tqdm
from plt_model import Wrapper
from dataset import WikiText2Dataset, MyDataLoader, prepare_data, preprocess_datasets

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import Callback

# 移除不必要的导入
# from rouge_score import rouge_scorer
# from nltk.translate.meteor_score import meteor_score
# import nltk


def plt_model_load(model, checkpoint):
    state_dict = torch.load(checkpoint)['state_dict']
    model.load_state_dict(state_dict)
    return model


def load_model(load_path, plt_model):
    if load_path is not None:
        if load_path.endswith(".ckpt"):
            checkpoint = load_path
        else:
            if load_path.endswith("/"):
                checkpoint = load_path + "best.ckpt"
            else:
                raise ValueError(
                    "if it is a directory, if must end with /; if it is a file, it must end with .ckpt"
                )
        plt_model = plt_model_load(plt_model, checkpoint)
        print(f"Loaded model from {checkpoint}")
    return plt_model


def main():
    # 简化参数定义，只保留长格式参数
    # 在参数定义部分添加困惑度计算相关参数
    arguments = {
        ('--model_tag',): {
            'type': str,
            'default': 'facebook/opt-125m',
            'help': 'Model tag for the model to use.',
        },
        ('--load-name',): {
            'type': str,
            'default': None,
            'help': 'Name of the saved model to restore.'
        },
        ('--save-name',): {
            'type': str,
            'default': None,
            'help': 'Name of the saved model to save.'
        },
        ('--save_perplexity',): {
            'type': str,
            'default': '/root/autodl-tmp/model_collapse_analysis/generated_data/default_perplexity.pkl',  # 修改为默认保存路径
            'help': 'Path to save perplexity scores separately',
        },
        ('--save_gen_perplexity',): {  # 添加缺失的参数
            'type': str,
            'default': None,
            'help': '保存生成数据困惑度的路径',
        },
        ('--eval_only',): {
            'action': 'store_true',
            'help': 'Only run eval on the test set',
        },
        ('--evalgen_only',): {
            'action': 'store_true',
            'help': 'Only run eval on the test set',
        },
        ('--optimizer',): {
            'type': str,
            'default': 'adamw',
            'help': 'Pick an optimizer.',
        },
        ('--learning-rate',): {
            'type': float,
            'default': 2e-5,
            'help': 'Initial learning rate.',
        },
        ('--max-epochs',): {
            'type': int,
            'default': 6,
            'help': 'Maximum number of epochs for training.',
        },
        ('--saveto',): {
            'type': str,
            'default': None,
            'help': 'Save the evaluation result to what location',
        },
        ('--batch-size',): {
            'type': int,
            'default': 256,
            'help': 'Batch size for training and evaluation.',
        },
        ('--memory_limit',): {
            'type': float,
            'default': 0.92,
            'help': 'GPU memory limit ratio',
        },
        ('--debug',): {
            'action': 'store_true',
            'help': 'Verbose debug',
        },
        ('--seed',): {
            'type': int,
            'default': 0,
            'help': 'Random seed for reproducibility',
        },
        ('--num_workers',): {
            'type': int,
            'default': 12,
            'help': 'Number of CPU workers.',
        },
        ('--num_devices',): {
            'type': int,
            'default': 1,
            'help': 'Number of GPU devices.',
        },
        ('--accelerator',): {
            'type': str,
            'default': 'gpu',
            'help': 'Accelerator style.',
        },
        ('--strategy',): {
            'type': str,
            'default': 'ddp',
            'help': 'Strategy style.',
        },
        ('--pretrained',): {
            'action': 'store_true',
            'help': 'Load a pretrained network from Huggingface',
        },
        ('--version_name',): {
            'type': str,
            'default': None,
            'help': 'Version name.',
        },
        ('--generate',): {
            'type': str,
            'default': '/root/autodl-tmp/model_collapse_analysis/generated_data/default_generation.pkl',  # 修改为默认保存路径
            'help': 'The file name to store the generated dataset.',
        },
        ('--save_perplexity',): {
            'type': str,
            'default': '/root/autodl-tmp/model_collapse_analysis/generated_data/default_perplexity.pkl',  # 修改为默认保存路径
            'help': 'Path to save perplexity scores separately',
        },
        ('--log_interval',): {
            'type': int,
            'default': 10,  # 更频繁的日志间隔
            'help': 'Interval for logging metrics during generation',
        },
        ('--evalgen_only',): {
            'action': 'store_true',
            'help': 'Only run eval on the test set',
        },
        ('--optimizer',): {
            'type': str,
            'default': 'adamw',
            'help': 'Pick an optimizer.',
        },
        ('--learning-rate',): {
            'type': float,
            'default': 2e-5,
            'help': 'Initial learning rate.',
        },
        ('--max-epochs',): {
            'type': int,
            'default': 6,
            'help': 'Maximum number of epochs for training.',
        },
        ('--saveto',): {
            'type': str,
            'default': None,
            'help': 'Save the evaluation result to what location',
        },
        ('--batch-size',): {
            'type': int,
            'default': 256,
            'help': 'Batch size for training and evaluation.',
        },
        ('--memory_limit',): {
            'type': float,
            'default': 0.92,
            'help': 'GPU memory limit ratio',
        },
        ('--debug',): {
            'action': 'store_true',
            'help': 'Verbose debug',
        },
        ('--seed',): {
            'type': int,
            'default': 0,
            'help': 'Random seed for reproducibility',
        },
        ('--num_workers',): {
            'type': int,
            'default': 12,
            'help': 'Number of CPU workers.',
        },
        ('--num_devices',): {
            'type': int,
            'default': 1,
            'help': 'Number of GPU devices.',
        },
        ('--accelerator',): {
            'type': str,
            'default': 'gpu',
            'help': 'Accelerator style.',
        },
        ('--strategy',): {
            'type': str,
            'default': 'ddp',
            'help': 'Strategy style.',
        },
        ('--pretrained',): {
            'action': 'store_true',
            'help': 'Load a pretrained network from Huggingface',
        },
        ('--version_name',): {
            'type': str,
            'default': None,
            'help': 'Version name.',
        },
        ('--generate',): {
            'type': str,
            'default': None,
            'help': 'The file name to store the generated dataset.',
        },
        ('--load-generate',): {
            'type': str,
            'default': None,
            'help': 'The file name to load the generated dataset.',
        },
        ('--generate_percentage',): {
            'type': float,
            'default': 1.0,
            'help': '使用生成数据的比例 (0.0-1.0)',
        },
        ('--original_percentage',): {
            'type': float,
            'default': 0.0,
            'help': '保留原始数据的比例 (0.0-1.0)',
        },
        ('--save_metrics',): {
            'type': str,
            'default': None,
            'help': '保存评估指标的路径',
        },
        ('--save_generations',): {
            'type': str,
            'default': None,
            'help': 'Path to save generated texts with generation info',
        },
        ('--log_interval',): {
            'type': int,
            'default': 100,
            'help': 'Interval for logging metrics during generation',
        },
        ('--saveperplexities',): {
            'type': str,
            'default': None,
            'help': '保存每个样本的困惑度列表',
        },
        ('--skip_preprocess',): {
            'action': 'store_true',
            'help': '跳过数据预处理，直接使用已处理好的数据',
        },
        ('--num_generations',): {
            'type': int,
            'default': 32000,
            'help': 'Number of samples to generate',
        },
        ('--gen_batch_size',): {
            'type': int,
            'default': 1024,
            'help': '生成时使用的批次大小',
        },
        ('--fp16',): {
            'action': 'store_true',
            'help': '使用半精度(FP16)进行训练和生成',
        },
        ('--generate_mode',): {
            'action': 'store_true',
            'help': '专用生成模式，跳过训练直接生成数据',
        },
        ('--text_data_format',): {
            'action': 'store_true',
            'help': '指示加载的生成数据是文本格式，需要重新tokenize',
        },
        ('--current_generation',): {
            'type': int,
            'default': 0,
            'help': '当前迭代的代数',
        },
        ('--current_run',): {
            'type': int,
            'default': 1,
            'help': '当前迭代的运行次数',
        },
        ('--save_metrics_csv',): {
            'type': str,
            'default': None,
            'help': '保存指标到CSV文件的路径',
        },
        ('--calculate_perplexity',): {
            'action': 'store_true',
            'help': '计算生成文本的困惑度',
        },
    }


    p = argparse.ArgumentParser(description='GPT-N')
    for k, v in arguments.items():
        p.add_argument(*k, **v)
    a = p.parse_args()

    if (a.saveto is not None) and os.path.exists(a.saveto):
        exit()

    # 设置随机种子
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    
    # 在模型加载后添加内存优化设置
    # GPU优化设置
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision('medium')
        
        # 添加内存优化设置
        torch.cuda.empty_cache()  # 清空缓存
        
        # 设置PyTorch内存分配器
        if hasattr(a, 'memory_limit') and a.memory_limit < 1.0:
            # 设置环境变量以优化内存分配
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = f'max_split_size_mb:128,expandable_segments:True'
            
            # 打印当前GPU内存使用情况
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / (1024 ** 2)
                cached = torch.cuda.memory_reserved() / (1024 ** 2)
                max_allocated = torch.cuda.max_memory_allocated() / (1024 ** 2)
                print(f"当前GPU内存使用情况:")
                print(f"分配: {allocated:.2f} MB")
                print(f"缓存: {cached:.2f} MB")
                print(f"最大分配: {max_allocated:.2f} MB")
        
        print(f"已启用GPU优化: cudnn benchmark={torch.backends.cudnn.benchmark}, float32_precision=medium")
    
    # 模型加载
    model_tag = a.model_tag
    local_model_path = f'/root/autodl-tmp/model_collapse_analysis/model_cache_dir/{model_tag.split("/")[-1]}'
    
    # 加载模型和tokenizer
    try:
        if a.pretrained:
            if not os.path.exists(local_model_path):
                print(f"模型缓存不存在，正在下载 {model_tag}...")
                os.makedirs(os.path.dirname(local_model_path), exist_ok=True)
                os.system(f'huggingface-cli download --resume-download {model_tag} --local-dir {local_model_path}')
            
            print(f"正在加载模型 {model_tag}...")
            model = AutoModelForCausalLM.from_pretrained(local_model_path)
            tokenizer = AutoTokenizer.from_pretrained(local_model_path, return_dict=True)
        else:
            config = AutoConfig.from_pretrained(model_tag)
            model = AutoModelForCausalLM.from_config(config=config)
            tokenizer = AutoTokenizer.from_pretrained(model_tag, return_dict=True)
        
        print(f"模型参数数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    except Exception as e:
        print(f"模型加载失败: {str(e)}")
        exit(1)

    # 数据加载
    print("\n" + "="*50)
    print("数据加载阶段")
    print("="*50)
    
    processed_data_path = '/root/autodl-tmp/model_collapse_analysis/data/wikitext2_processed/dataset.pkl'
    dataset = None
    
    # 尝试加载预处理数据
    if a.skip_preprocess and os.path.exists(processed_data_path):
        try:
            with open(processed_data_path, 'rb') as f:
                dataset = pickle.load(f)
            print(f"成功加载预处理数据")
        except Exception as e:
            print(f"加载预处理数据失败: {str(e)}")
            dataset = None
    
    # 如果需要，处理新数据
    if dataset is None:
        raw_dataset = prepare_data()
        tokenized_cache_path = '/root/autodl-tmp/model_collapse_analysis/data/wikitext2_tokenized'
        
        if os.path.exists(tokenized_cache_path):
            try:
                dataset = load_from_disk(tokenized_cache_path)
            except Exception:
                dataset = preprocess_datasets(raw_dataset, tokenizer)
                dataset.save_to_disk(tokenized_cache_path)
        else:
            dataset = preprocess_datasets(raw_dataset, tokenizer)
            os.makedirs(tokenized_cache_path, exist_ok=True)
            dataset.save_to_disk(tokenized_cache_path)
        
        os.makedirs(os.path.dirname(processed_data_path), exist_ok=True)
        with open(processed_data_path, 'wb') as f:
            pickle.dump(dataset, f)

    # 准备数据集
    if a.load_generate is not None:
        try:
            print(f"正在加载生成数据: {a.load_generate}")
            with open(a.load_generate, 'rb') as f:
                generated_data = pickle.load(f)
            
            # 检查是否为文本格式数据
            if (isinstance(generated_data, list) and len(generated_data) > 0 and 
                isinstance(generated_data[0], dict) and 'generated_text' in generated_data[0]):
                print("检测到文本格式数据，正在进行tokenize处理...")
                
                # 创建自定义数据集类来处理文本数据
                class TextGeneratedDataset(torch.utils.data.Dataset):
                    def __init__(self, data, tokenizer, max_length=128):
                        self.data = data
                        self.tokenizer = tokenizer
                        self.max_length = max_length
                    
                    def __len__(self):
                        return len(self.data)
                    
                    def __getitem__(self, idx):
                        item = self.data[idx]
                        # 优先使用生成的文本，如果没有则使用输入文本
                        text = item.get('generated_text', item.get('input_text', ''))
                        
                        # Tokenize文本
                        encodings = self.tokenizer(
                            text, 
                            max_length=self.max_length,
                            padding='max_length',
                            truncation=True,
                            return_tensors='pt'
                        )
                        
                        # 准备模型输入
                        input_ids = encodings['input_ids'].squeeze()
                        attention_mask = encodings['attention_mask'].squeeze()
                        
                        # 创建标签（与输入相同，用于自回归训练）
                        labels = input_ids.clone()
                        
                        return {
                            'input_ids': input_ids,
                            'attention_mask': attention_mask,
                            'labels': labels
                        }
                
                # 使用自定义数据集加载文本数据
                train_dataset = TextGeneratedDataset(generated_data, tokenizer)
                print(f"成功加载并处理了 {len(train_dataset)} 个生成的文本样本")
            else:
                # 如果是旧格式数据（已经tokenize好的），直接使用
                train_dataset = generated_data
                print(f"成功加载了 {len(train_dataset)} 个预处理好的生成样本")
        except Exception as e:
            print(f"加载生成数据集失败: {str(e)}")
            import traceback
            print(traceback.format_exc())
            print("回退到使用原始训练数据...")
            train_dataset = WikiText2Dataset(dataset=dataset, partition='train', tokenizer=tokenizer)
    else:
        train_dataset = WikiText2Dataset(dataset=dataset, partition='train', tokenizer=tokenizer)

    val_dataset = WikiText2Dataset(dataset=dataset, partition='validation', tokenizer=tokenizer)
    test_dataset = WikiText2Dataset(dataset=dataset, partition='test', tokenizer=tokenizer)

    if a.evalgen_only:
        test_dataset = train_dataset

    # 在数据加载部分优化线程数使用
    # 根据CPU核心数动态设置最佳工作线程数
    optimal_workers = min(a.num_workers, os.cpu_count() or 4)
    print(f"使用数据加载线程数: {optimal_workers}")
    
    # 修改数据加载器创建部分
    data_loader = MyDataLoader(
        'wikitext2',
        optimal_workers,  # 使用优化后的线程数
        train_dataset, 
        val_dataset, 
        test_dataset, 
        batch_size=a.batch_size)
    
    # 设置模型保存
    if a.save_name is not None:
        os.makedirs(a.save_name, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="val_loss",
        mode="min",
        filename="best",
        dirpath=a.save_name,
        save_last=True,
    )

    # 初始化模型和训练器
    plt_model = Wrapper(
        model,
        learning_rate=a.learning_rate,
        epochs=a.max_epochs)
    
    logger = TensorBoardLogger(
        '/root/autodl-tmp/model_collapse_analysis/logs', 
        version=a.version_name
    )

    plt_model = load_model(plt_model=plt_model, load_path=a.load_name)

    # 修改trainer配置，添加梯度累积
    trainer = pl.Trainer(
        max_epochs=a.max_epochs,
        devices=1,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        strategy=None,
        callbacks=[checkpoint_callback],
        logger=logger,
        enable_progress_bar=True,
        enable_model_summary=True,
        detect_anomaly=False,
        num_sanity_val_steps=0,
        accumulate_grad_batches=a.gradient_accumulation_steps if hasattr(a, 'gradient_accumulation_steps') else 1,  # 添加梯度累积
        precision=16 if hasattr(a, 'fp16') and a.fp16 else 32  # 根据fp16参数设置精度
    )

    # 训练和评估
    # 在训练前添加内存使用情况打印
    if torch.cuda.is_available():
        print(f"当前GPU内存使用情况:")
        print(f"分配: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"缓存: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
        print(f"最大分配: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")
    
    # 检查是否为生成模式，如果是则跳过训练和评估
    if hasattr(a, 'generate_mode') and a.generate_mode:
        print("启用生成模式，跳过训练和评估阶段")
    else:
        # 在训练部分添加更详细的异常处理
        try:
            if not a.eval_only:
                print("开始训练模型...")
                trainer.fit(
                    plt_model, 
                    train_dataloaders=data_loader.train_dataloader,
                    val_dataloaders=data_loader.val_dataloader)
                print("训练完成")
        except Exception as e:
            print(f"训练过程中发生错误: {str(e)}")
            import traceback
            print(f"详细错误信息:\n{traceback.format_exc()}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print("已清理GPU缓存")

        # 保存困惑度
        if hasattr(a, 'saveperplexities') and a.saveperplexities:
            plt_model.tosave = True

        # 测试模型
        res = trainer.test(plt_model, dataloaders=data_loader.test_dataloader)
        
        # 保存结果
        if hasattr(a, 'saveperplexities') and a.saveperplexities is not None:
            os.makedirs(os.path.dirname(a.saveperplexities), exist_ok=True)
            with open(a.saveperplexities, "wb") as f:
                pickle.dump(plt_model.saved, f)

        if a.save_perplexity and not a.generate:
            os.makedirs(os.path.dirname(a.save_perplexity), exist_ok=True)
            test_perplexity = torch.exp(torch.tensor(res[0]['test_loss'])).item()
            with open(a.save_perplexity, "wb") as f:
                pickle.dump({
                    'test_perplexity': test_perplexity,
                    'timestamp': datetime.now().isoformat()
                }, f)

        if a.saveto is not None:
            os.makedirs(os.path.dirname(a.saveto), exist_ok=True)
            with open(a.saveto, "wb") as f:
                pickle.dump(res, f)
    
    # 生成数据集
    # 在生成数据阶段添加limit变量定义
    if a.generate is not None:
        try:
            print("\n" + "="*50)
            print("数据生成阶段")
            print("="*50)
            
            # 清空GPU缓存，确保生成阶段有足够内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print("已清理GPU缓存，准备开始生成数据")
                # 打印当前GPU内存使用情况
                allocated = torch.cuda.memory_allocated() / (1024 ** 2)
                cached = torch.cuda.memory_reserved() / (1024 ** 2)
                max_allocated = torch.cuda.max_memory_allocated() / (1024 ** 2)
                print(f"当前GPU内存使用情况:")
                print(f"分配: {allocated:.2f} MB")
                print(f"缓存: {cached:.2f} MB")
                print(f"最大分配: {max_allocated:.2f} MB")
            
            # 设置生成批次大小
            gen_batch_size = 1024  # 固定生成批次大小为1024
            if hasattr(a, 'gen_batch_size') and a.gen_batch_size is not None:
                print(f"注意：忽略命令行参数中的gen_batch_size，使用固定值1024")
            print(f"使用生成批次大小: {gen_batch_size}")
            
            # 设置数据加载线程数
            optimal_workers = min(a.num_workers, os.cpu_count() or 4)
            print(f"使用数据加载线程数: {optimal_workers}")
            
            # 设置生成数量限制
            limit = a.num_generations // gen_batch_size if hasattr(a, 'num_generations') else 100
            print(f"计划生成 {limit * gen_batch_size} 个样本 (批次大小 {gen_batch_size} × {limit} 批次)")
            
            # 启用困惑度计算
            calculate_perplexity = hasattr(a, 'calculate_perplexity') and a.calculate_perplexity
            if calculate_perplexity:
                print("已启用生成样本困惑度计算")
                perplexity_scores = []
            
            # 生成数据
            batches = []
            for batch_idx, batch in enumerate(tqdm(data_loader.test_dataloader(), desc="生成数据")):
                # 检查是否达到生成数量限制
                if batch_idx >= limit:
                    break
                
                input_ids = batch["input_ids"].to(plt_model.device)
                attention_mask = batch["attention_mask"].to(plt_model.device)
                
                # 使用半精度加速生成
                with torch.cuda.amp.autocast(enabled=hasattr(a, 'fp16') and a.fp16):
                    with torch.no_grad():
                        # 生成文本
                        generated_ids = model.generate(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            max_length=128,  # 设置生成的最大长度
                            do_sample=True,  # 使用采样而不是贪婪解码
                            top_p=0.95,      # 使用nucleus sampling
                            top_k=50,        # 使用top-k sampling
                            temperature=0.8, # 控制随机性
                            num_return_sequences=1,
                            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                        )
                        
                        # 如果启用了困惑度计算，计算生成文本的困惑度
                        if calculate_perplexity:
                            # 计算生成文本的困惑度
                            with torch.no_grad():
                                outputs = model(generated_ids, labels=generated_ids)
                                loss = outputs.loss
                                batch_perplexity = torch.exp(loss).item()
                                perplexity_scores.append(batch_perplexity)
                                
                                if batch_idx % a.log_interval == 0:
                                    print(f"批次 {batch_idx}/{limit}, 困惑度: {batch_perplexity:.4f}")
                
                # 保存生成的ID和输入ID
                batch_data = {
                    "input_ids": input_ids.cpu().numpy(),
                    "generated_ids": generated_ids.cpu().numpy(),
                }
                batches.append(batch_data)
                
                # 定期打印进度
                if batch_idx % a.log_interval == 0:
                    print(f"已完成: {batch_idx}/{limit} 批次")
            
            # 处理生成的数据
            print("生成完成，正在处理数据...")
            processed_data = []
            
            for item in tqdm(batches, desc="处理生成数据"):
                # 解码文本
                generated_text = tokenizer.decode(item["generated_ids"][0], skip_special_tokens=True)
                input_text = tokenizer.decode(item["input_ids"][0], skip_special_tokens=True)
                
                # 只保存文本数据
                processed_data.append({
                    "input_text": input_text,
                    "generated_text": generated_text,
                })
            
            # 保存生成的数据
            os.makedirs(os.path.dirname(a.generate), exist_ok=True)
            with open(a.generate, "wb") as f:
                pickle.dump(processed_data, f)
            print(f"已保存生成数据到: {a.generate}")
            
            # 如果计算了困惑度，保存困惑度
            if calculate_perplexity and hasattr(a, 'save_gen_perplexity') and a.save_gen_perplexity:
                avg_perplexity = sum(perplexity_scores) / len(perplexity_scores)
                perplexity_data = {
                    'perplexity_scores': perplexity_scores,
                    'avg_perplexity': avg_perplexity,
                    'timestamp': datetime.now().isoformat()
                }
                
                os.makedirs(os.path.dirname(a.save_gen_perplexity), exist_ok=True)
                with open(a.save_gen_perplexity, "wb") as f:
                    pickle.dump(perplexity_data, f)
                print(f"已保存生成数据困惑度到: {a.save_gen_perplexity}")
                print(f"生成数据平均困惑度: {avg_perplexity:.4f}")
            
            # 在生成完成后再次清理缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print("生成完成，已清理GPU缓存")
                
                # 打印最终内存状态
                allocated = torch.cuda.memory_allocated() / (1024 ** 2)
                cached = torch.cuda.memory_reserved() / (1024 ** 2)
                max_allocated = torch.cuda.max_memory_allocated() / (1024 ** 2)
                print(f"最终GPU内存使用情况:")
                print(f"分配: {allocated:.2f} MB")
                print(f"缓存: {cached:.2f} MB")
                print(f"最大分配: {max_allocated:.2f} MB")
                
                # 重置最大内存统计
                torch.cuda.reset_peak_memory_stats()
                print("已重置GPU峰值内存统计")
                
        except Exception as e:
            print(f"生成数据时出错: {str(e)}")
            import traceback
            print(traceback.format_exc())
    
