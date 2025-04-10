import os
import subprocess
import argparse
import time
from datetime import datetime
import logging
import sys

# 设置日志
def setup_logging(base_dir):
    log_dir = os.path.join(base_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'mixed_iteration_log.txt')),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def run_command(cmd, desc=None, logger=None):
    """运行命令并记录输出"""
    if desc and logger:
        logger.info(f"开始: {desc}")
    
    if logger:
        logger.info(f"执行命令: {cmd}")
    print(f"执行命令: {cmd}")
    start_time = time.time()
    
    try:
        process = subprocess.Popen(
            cmd, 
            shell=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )
        
        # 实时输出命令执行结果，优化进度条显示
        last_line = ""
        for line in process.stdout:
            line = line.strip()
            # 检测是否为进度信息行（包含百分比或epoch信息）
            if "%" in line or "Epoch" in line or "it/s" in line:
                # 在同一行更新进度信息
                print(f"\r{line}", end="", flush=True)
                last_line = line
            else:
                # 如果上一行是进度信息，先打印一个换行
                if last_line and ("%" in last_line or "Epoch" in last_line or "it/s" in last_line):
                    print()  # 打印换行
                # 正常打印非进度信息
                print(line)
                last_line = line
        
        # 确保最后有一个换行
        if last_line and ("%" in last_line or "Epoch" in last_line or "it/s" in last_line):
            print()
        
        process.wait()
        
        if process.returncode != 0:
            if logger:
                logger.error(f"命令执行失败，返回码: {process.returncode}")
            print(f"命令执行失败，返回码: {process.returncode}")
            return False
        
        elapsed_time = time.time() - start_time
        if logger:
            logger.info(f"完成: {desc if desc else cmd} (耗时: {elapsed_time:.2f}秒)")
        print(f"完成: {desc if desc else cmd} (耗时: {elapsed_time:.2f}秒)")
        return True
    
    except Exception as e:
        if logger:
            logger.error(f"执行命令时出错: {str(e)}")
        print(f"执行命令时出错: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description='模型崩溃分析迭代训练控制脚本 - 混合数据策略')
    parser.add_argument('-tag', '--model_tag', type=str, default='facebook/opt-125m', 
                        help='要使用的模型标签')
    parser.add_argument('-m', '--max_epochs', type=int, default=6, 
                        help='每次训练的最大轮数')
    parser.add_argument('-b', '--batch_size', type=int, default=256, 
                        help='训练和评估的批次大小')
    parser.add_argument('-gb', '--gen_batch_size', type=int, default=32, 
                        help='生成时的批次大小')
    parser.add_argument('-lr', '--learning_rate', type=float, default=2e-5, 
                        help='学习率')
    parser.add_argument('-n', '--num_iterations', type=int, default=15, 
                        help='总迭代次数（包括第一代）')
    parser.add_argument('-seed', '--seed', type=int, default=42, 
                        help='随机种子')
    parser.add_argument('-num_gen', '--num_generations', type=int, default=1000, 
                        help='每次生成的样本数量')
    parser.add_argument('-fp16', '--half_precision', action='store_true',
                        help='是否使用半精度(FP16)进行生成')
    parser.add_argument('-orig', '--original_data_ratio', type=float, default=0.1,
                        help='第二代开始使用的原始数据比例 (默认: 0.1，即10%)')
    parser.add_argument('-base_dir', '--base_directory', type=str, 
                        default='/root/autodl-tmp/model_collapse_analysis',
                        help='实验数据保存的基础目录')
    parser.add_argument('-resume', '--resume_from', type=int, default=None,
                        help='从指定代数继续实验 (例如: 3 表示从第3代开始)')
    parser.add_argument('-exp_id', '--experiment_id', type=str, default=None,
                        help='指定实验ID (用于恢复实验)')
    args = parser.parse_args()
    
    # 使用参数中的基础目录
    base_dir = args.base_directory
    
    # 设置日志
    logger = setup_logging(base_dir)
    
    # 确定实验ID
    if args.resume_from is not None and args.experiment_id is not None:
        experiment_id = args.experiment_id
        logger.info(f"恢复实验 ID: {experiment_id}, 从第{args.resume_from}代开始")
    else:
        experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        logger.info(f"开始新实验 ID: {experiment_id}, 模型: {args.model_tag}, 迭代次数: {args.num_iterations}")
    
    # 为本次实验创建一个集中的目录
    exp_dir = os.path.join(base_dir, "experiments", experiment_id)
    os.makedirs(exp_dir, exist_ok=True)
    
    # 在实验目录下创建子目录
    for subdir in ['models', 'generations', 'perplexities', 'logs']:
        os.makedirs(os.path.join(exp_dir, subdir), exist_ok=True)
    
    # 保存实验配置
    config_file = os.path.join(exp_dir, "config.txt")
    with open(config_file, "w") as f:
        f.write(f"实验ID: {experiment_id}\n")
        f.write(f"模型: {args.model_tag}\n")
        f.write(f"迭代次数: {args.num_iterations}\n")
        f.write(f"每代训练轮数: {args.max_epochs}\n")
        f.write(f"训练批次大小: {args.batch_size}\n")
        f.write(f"生成批次大小: {args.gen_batch_size}\n")
        f.write(f"学习率: {args.learning_rate}\n")
        f.write(f"随机种子: {args.seed}\n")
        f.write(f"第二代开始的原始数据比例: {args.original_data_ratio}\n")
        f.write(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    logger.info(f"实验配置已保存到 {config_file}")
    
    # 使用完整路径执行main.py
    # 假设main.py与run_mixed_iterations.py在同一目录
    main_py_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "main.py")
    
    # 确定起始代
    start_gen = 0
    if args.resume_from is not None:
        start_gen = args.resume_from - 1  # 转换为0-based索引
        
        # 验证恢复点
        if start_gen > 0:
            prev_gen = start_gen - 1
            prev_model_dir = os.path.join(exp_dir, "models", f"gen{prev_gen}")
            prev_data_file = os.path.join(exp_dir, "generations", f"gen{prev_gen}_data.pkl")
            
            if not os.path.exists(os.path.join(prev_model_dir, "best.ckpt")):
                logger.error(f"无法恢复实验: 上一代模型文件不存在 {os.path.join(prev_model_dir, 'best.ckpt')}")
                return
            
            if not os.path.exists(prev_data_file):
                logger.error(f"无法恢复实验: 上一代数据文件不存在 {prev_data_file}")
                return
            
            logger.info(f"验证通过，将从第{args.resume_from}代继续实验")
    
    # 如果是第一代，执行初始训练
    if start_gen == 0:
        logger.info("="*50)
        logger.info("开始第1代训练 (使用100%原始数据)")
        logger.info("="*50)
        
        gen_0_model_dir = os.path.join(exp_dir, "models", "gen0")
        gen_0_ppl_file = os.path.join(exp_dir, "perplexities", "gen0_ppl.pkl")
        
        # 训练第一代模型
        # 根据main.py的参数定义调整参数名
        cmd_train_gen0 = (
            f"python {main_py_path} "
            f"-tag {args.model_tag} "
            f"-save {gen_0_model_dir} "
            f"-save_ppl {gen_0_ppl_file} "
            f"-m {args.max_epochs} "
            f"-b {args.batch_size} "
            f"-lr {args.learning_rate} "
            f"-seed {args.seed} "
            f"-p "  # 使用预训练模型
            f"-original_percentage 1.0 "
            f"-version_name {experiment_id}_gen0"
        )
        
        if not run_command(cmd_train_gen0, "第1代模型训练", logger):
            logger.error("第1代模型训练失败，终止实验")
            return
        
        # 使用第一代模型生成数据
        gen_0_data_file = os.path.join(exp_dir, "generations", "gen0_data.pkl")
        gen_0_gen_ppl_file = os.path.join(exp_dir, "perplexities", "gen0_gen_ppl.pkl")
        
        # 生成命令
        cmd_gen_data = (
            f"python {main_py_path} "
            f"-tag {args.model_tag} "
            f"-load {os.path.join(gen_0_model_dir, 'best.ckpt')} "
            f"-generate {gen_0_data_file} "
            f"-save_ppl {gen_0_gen_ppl_file} "
            f"-b {args.gen_batch_size} "
            f"-seed {args.seed} "
            f"-p "
            f"-skip_preprocess "
            f"-generate_percentage {min(args.num_generations/1000, 1.0)} "
            f"-version_name {experiment_id}_gen0_gen"
        )
        
        # 如果启用半精度
        if args.half_precision:
            cmd_gen_data += " -fp16"
        
        # 在执行命令前打印完整命令进行检查
        logger.info(f"即将执行命令: {cmd_gen_data}")
        
        if not run_command(cmd_gen_data, "第1代数据生成", logger):
            logger.error("第1代数据生成失败，终止实验")
            return
        
        # 添加短暂延迟，确保文件系统完成写入
        time.sleep(2)
        
        # 检查生成的数据文件是否存在
        if not os.path.exists(gen_0_data_file):
            logger.error(f"生成的数据文件 {gen_0_data_file} 不存在，终止实验")
            return
        
        # 设置起始点为第一代
        prev_gen_model_dir = gen_0_model_dir
        prev_gen_data_file = gen_0_data_file
        start_gen = 1  # 从第二代开始迭代
    else:
        # 恢复实验时，设置上一代的引用
        prev_gen = start_gen - 1
        prev_gen_model_dir = os.path.join(exp_dir, "models", f"gen{prev_gen}")
        prev_gen_data_file = os.path.join(exp_dir, "generations", f"gen{prev_gen}_data.pkl")
    
    # 迭代训练后续代
    for gen in range(start_gen, args.num_iterations):
        logger.info("="*50)
        logger.info(f"开始第{gen+1}代训练 (混合数据策略: {args.original_data_ratio*100}%原始数据 + {(1-args.original_data_ratio)*100}%生成数据)")
        logger.info("="*50)
        
        current_gen_model_dir = os.path.join(exp_dir, "models", f"gen{gen}")
        current_gen_ppl_file = os.path.join(exp_dir, "perplexities", f"gen{gen}_ppl.pkl")
        
        # 使用混合数据策略训练当前代模型
        cmd_train_current_gen = (
            f"python {main_py_path} "
            f"-tag {args.model_tag} "
            f"-load {os.path.join(prev_gen_model_dir, 'best.ckpt')} "
            f"-save {current_gen_model_dir} "
            f"-save_ppl {current_gen_ppl_file} "
            f"-load-generate {prev_gen_data_file} "
            f"-m {args.max_epochs} "
            f"-b {args.batch_size} "
            f"-lr {args.learning_rate} "
            f"-seed {args.seed} "
            f"-p "
            f"-original_percentage {args.original_data_ratio} "
            f"-version_name {experiment_id}_gen{gen}"
        )
        
        if not run_command(cmd_train_current_gen, f"第{gen+1}代模型训练", logger):
            logger.error(f"第{gen+1}代模型训练失败，终止实验")
            break
        
        # 使用当前代模型生成下一代数据
        current_gen_data_file = os.path.join(exp_dir, "generations", f"gen{gen}_data.pkl")
        current_gen_gen_ppl_file = os.path.join(exp_dir, "perplexities", f"gen{gen}_gen_ppl.pkl")
        
        # 生成命令
        cmd_gen_current_data = (
            f"python {main_py_path} "
            f"-tag {args.model_tag} "
            f"-load {os.path.join(current_gen_model_dir, 'best.ckpt')} "
            f"-generate {current_gen_data_file} "
            f"-save_ppl {current_gen_gen_ppl_file} "
            f"-b {args.gen_batch_size} "
            f"-seed {args.seed} "
            f"-p "
            f"-skip_preprocess "
            f"-generate_percentage {min(args.num_generations/1000, 1.0)} "
            f"-version_name {experiment_id}_gen{gen}_gen"
        )
        
        # 如果启用半精度
        if args.half_precision:
            cmd_gen_current_data += " -fp16"
        
        if not run_command(cmd_gen_current_data, f"第{gen+1}代数据生成", logger):
            logger.error(f"第{gen+1}代数据生成失败，终止实验")
            break
        
        # 更新引用，为下一代做准备
        prev_gen_model_dir = current_gen_model_dir
        prev_gen_data_file = current_gen_data_file
        
        # 创建检查点文件，记录当前进度
        checkpoint_file = os.path.join(exp_dir, "checkpoint.txt")
        with open(checkpoint_file, "w") as f:
            f.write(f"experiment_id={experiment_id}\n")
            f.write(f"current_gen={gen+1}\n")
            f.write(f"timestamp={datetime.now().isoformat()}\n")
        
        logger.info(f"已创建检查点: 完成第{gen+1}代")
    
    # 实验完成
    logger.info("="*50)
    logger.info(f"实验 {experiment_id} 完成")
    logger.info("="*50)
    
    # 生成实验摘要
    summary = (
        f"实验ID: {experiment_id}\n"
        f"模型: {args.model_tag}\n"
        f"迭代次数: {args.num_iterations}\n"
        f"每代训练轮数: {args.max_epochs}\n"
        f"训练批次大小: {args.batch_size}\n"
        f"生成批次大小: {args.gen_batch_size}\n"
        f"学习率: {args.learning_rate}\n"
        f"随机种子: {args.seed}\n"
        f"第二代开始的原始数据比例: {args.original_data_ratio}\n"
        f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    )
    
    with open(os.path.join(exp_dir, "summary.txt"), "w") as f:
        f.write(summary)
    
    logger.info(f"实验摘要已保存到 {os.path.join(exp_dir, 'summary.txt')}")

if __name__ == "__main__":
    main()