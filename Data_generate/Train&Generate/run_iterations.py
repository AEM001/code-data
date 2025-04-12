import os
import subprocess
import argparse
import time
from datetime import datetime
import logging

# 设置日志
# 在run_iterations.py开头添加路径兼容性处理
import platform

# 根据操作系统设置基础目录
if platform.system() == 'Windows':
    base_dir = 'd:\\Code\\Zakahler-curse_recurse-b48c90a\\data'
else:
    base_dir = '/root/autodl-tmp/model_collapse_analysis'

# 使用os.path.join构建路径，而不是直接拼接
log_dir = os.path.join(base_dir, "logs")
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'{log_dir}/iteration_log.txt'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def run_command(cmd, desc=None):
    """运行命令并记录输出"""
    if desc:
        logger.info(f"开始: {desc}")
    
    logger.info(f"执行命令: {cmd}")
    start_time = time.time()
    
    try:
        # 在每次命令执行前尝试清理系统内存
        try:
            import gc
            gc.collect()
            logger.info("已执行垃圾回收，释放Python内存")
            
            # 如果是Linux系统，可以尝试清理系统缓存
            if os.name == 'posix':
                os.system('sync')  # 将缓存数据写入磁盘
                logger.info("已执行sync命令，将缓存数据写入磁盘")
        except Exception as e:
            logger.warning(f"内存清理尝试失败: {str(e)}")
        
        # 检查main.py文件是否存在
        if 'main.py' in cmd:
            main_path = cmd.split()[1]  # 获取main.py的路径
            if not os.path.exists(main_path):
                logger.error(f"主程序文件不存在: {main_path}")
                return False
            else:
                logger.info(f"确认主程序文件存在: {main_path}")
        
        process = subprocess.Popen(
            cmd, 
            shell=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )
        
        # 实时输出命令执行结果，优化进度条显示
        last_line = ""
        line_count = 0  # 添加行计数
        for line in process.stdout:
            line_count += 1  # 增加行计数
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
                # 在run_command函数中添加
                if "GPU" in line or "memory" in line or "error" in line.lower() or "exception" in line.lower():  # 新增错误监控
                    print(line)
                    logger.warning(line)  # 记录可能的错误信息
                
                print(line)
                last_line = line
        
        # 确保最后有一个换行
        if last_line and ("%" in last_line or "Epoch" in last_line or "it/s" in last_line):
            print()
        
        # 添加行数检查
        if line_count < 10:  # 如果输出行数太少，可能有问题
            logger.warning(f"命令输出行数过少({line_count}行)，可能执行不正确")
        
        process.wait()
        
        if process.returncode != 0:
            logger.error(f"命令执行失败，返回码: {process.returncode}")
            return False
        
        elapsed_time = time.time() - start_time
        logger.info(f"完成: {desc if desc else cmd} (耗时: {elapsed_time:.2f}秒)")
        
        # 添加时间检查
        if elapsed_time < 30 and ('train' in cmd or 'generate' in cmd):  # 训练或生成通常需要较长时间
            logger.warning(f"命令执行时间过短({elapsed_time:.2f}秒)，可能未正确执行")
        
        # 命令执行完成后，再次尝试清理内存
        try:
            import gc
            gc.collect()
            logger.info("命令执行完毕，已执行垃圾回收")
        except Exception:
            pass
            
        return True
    
    except Exception as e:
        logger.error(f"执行命令时出错: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())  # 添加详细错误信息
        return False

def main():
    parser = argparse.ArgumentParser(description='模型崩溃分析迭代训练控制脚本')
    parser.add_argument('--model_tag', type=str, default='facebook/opt-125m', 
                        help='要使用的模型标签')
    parser.add_argument('--max_epochs', type=int, default=6, 
                        help='每次训练的最大轮数')
    parser.add_argument('--batch_size', type=int, default=256, 
                        help='训练和评估的批次大小')
    parser.add_argument('--gen_batch_size', type=int, default=1024, 
                        help='生成时的批次大小')
    parser.add_argument('--learning_rate', type=float, default=2e-5, 
                        help='学习率')
    parser.add_argument('--num_iterations', type=int, default=12, 
                        help='总迭代次数（包括第一代）')
    parser.add_argument('--seed', type=int, default=42, 
                        help='随机种子')
    parser.add_argument('--num_generations', type=int, default=36000,
                        help='每次生成的样本数量，默认与原始训练数据规模相同')
    parser.add_argument('--half_precision', action='store_true',
                        help='是否使用半精度(FP16)进行生成')
    parser.add_argument('--base_directory', type=str, 
                        default='/root/autodl-tmp/model_collapse_analysis',
                        help='实验数据保存的基础目录')
    # 添加新参数 - 数据加载的工作线程数
    parser.add_argument('--num_workers', type=int, default=24, 
                        help='数据加载的工作线程数，建议设置为CPU核心数')
    
    # 删除部分生成相关参数
    # parser.add_argument('--partial_generation', action='store_true',
    #                    help='启用部分数据生成模式')
    # parser.add_argument('--partial_ratio', type=float, default=0.5,
    #                    help='部分生成模式下生成数据的比例 (0.0-1.0)')
    # parser.add_argument('--random_selection', action='store_true',
    #                    help='在部分生成模式下随机选择批次，而不是使用前N个批次')
    
    # 添加困惑度计算相关参数
    parser.add_argument('--calculate_perplexity', action='store_true',
                        help='计算生成文本的困惑度')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2,
                        help='梯度累积步数，用于减少内存使用')
    
    args = parser.parse_args()
    
    # 使用参数中的基础目录
    base_dir = args.base_directory
    
    # 使用固定的实验目录名称，而不是时间戳
    experiment_id = "fixed_experiment"  # 替换原来的时间戳生成
    logger.info(f"开始实验 ID: {experiment_id}, 模型: {args.model_tag}, 迭代次数: {args.num_iterations}")
    
    # 为本次实验创建一个固定的目录
    exp_dir = f"{base_dir}/experiments/{experiment_id}"
    os.makedirs(exp_dir, exist_ok=True)
    
    # 在实验目录下创建子目录
    for subdir in ['models', 'generations', 'perplexities', 'logs']:
        os.makedirs(f"{exp_dir}/{subdir}", exist_ok=True)
    
    # 第一代：使用100%原始数据训练
    logger.info("="*50)
    logger.info("开始第1代训练 (使用100%原始数据)")
    logger.info("="*50)
    
    gen_0_model_dir = f"{exp_dir}/models/gen0"
    gen_0_ppl_file = f"{exp_dir}/perplexities/gen0_ppl.pkl"
    gen_0_data_file = f"{exp_dir}/generations/gen0_data.pkl"
    gen_0_gen_ppl_file = f"{exp_dir}/perplexities/gen0_gen_ppl.pkl"
    
    # 使用完整路径执行main.py
    # 检查多个可能的路径
    possible_paths = [
        "/root/autodl-tmp/model_collapse_analysis/code/main.py",  # 原始路径
        "/root/autodl-tmp/model_collapse_analysis/main.py",       # 可能的替代路径
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "main.py")  # 当前目录
    ]
    
    main_py_path = None
    for path in possible_paths:
        if os.path.exists(path):
            main_py_path = path
            logger.info(f"找到main.py: {main_py_path}")
            break
    
    if main_py_path is None:
        logger.error("无法找到main.py文件，请检查路径")
        return
    
    # 简化第一代执行逻辑：训练并生成数据
    # 修改第一代训练命令，添加缺失的参数
    cmd_gen0 = (
        f"python {main_py_path} "
        f"--model_tag {args.model_tag} "
        f"--save-name {gen_0_model_dir} "
        f"--save_perplexity {gen_0_ppl_file} "
        f"--generate {gen_0_data_file} "
        f"--save_gen_perplexity {gen_0_gen_ppl_file} "
        f"--max-epochs {args.max_epochs} "
        f"--batch-size 256 "  # 固定训练批次大小为256
        f"--learning-rate {args.learning_rate} "
        f"--seed {args.seed} "
        f"--pretrained "  # 确保使用预训练模型
        f"--original_percentage 1.0 "  # 确保使用100%原始数据
        f"--num_generations {args.num_generations} "
        f"--version_name {experiment_id}_gen0 "
        f"--num_workers {args.num_workers} "  # 使用参数中的工作线程数
        f"--memory_limit 0.9 "  # 限制GPU内存使用
        f"--gradient_accumulation_steps {args.gradient_accumulation_steps} "
        f"--log_interval 50 "  # 降低日志频率以提高性能
    )
    
    # 添加半精度参数
    if args.half_precision:
        cmd_gen0 += " --fp16"
    
    # 添加困惑度计算参数
    if args.calculate_perplexity:
        cmd_gen0 += " --calculate_perplexity"
    
    # 打印完整命令以便调试
    logger.info(f"完整命令: {cmd_gen0}")
    
    # 尝试直接在当前目录执行命令
    logger.info("切换到main.py所在目录执行命令")
    main_dir = os.path.dirname(main_py_path)
    current_dir = os.getcwd()
    
    try:
        # 切换到main.py所在目录
        os.chdir(main_dir)
        logger.info(f"当前工作目录: {os.getcwd()}")
        
        # 修改命令使用相对路径
        local_cmd = cmd_gen0.replace(main_py_path, "./main.py")
        logger.info(f"修改后的命令: {local_cmd}")
        
        # 执行命令
        success = run_command(local_cmd, "第1代模型训练与数据生成")
        
        # 切回原目录
        os.chdir(current_dir)
        
        if not success:
            logger.error("第1代模型训练与数据生成失败，尝试使用绝对路径")
            # 如果失败，尝试使用原始命令
            if not run_command(cmd_gen0, "第1代模型训练与数据生成（使用绝对路径）"):
                logger.error("第1代模型训练与数据生成失败，终止实验")
                return
    except Exception as e:
        logger.error(f"执行命令时出错: {str(e)}")
        os.chdir(current_dir)  # 确保切回原目录
        return
    
    # 检查生成的数据文件
    if not os.path.exists(gen_0_data_file) or os.path.getsize(gen_0_data_file) < 1000:
        logger.error(f"第1代数据生成失败，文件不存在或过小")
        return
    
    # 打印文件大小信息
    file_size_mb = os.path.getsize(gen_0_data_file) / (1024 ** 2)  # MB
    logger.info(f"第1代生成数据文件大小: {file_size_mb:.2f} MB")
    
    # 迭代训练后续代
    prev_gen_model_dir = gen_0_model_dir
    prev_gen_data_file = gen_0_data_file
    
    # 在迭代训练后续代部分，每一代结束后添加额外清理
    for gen in range(1, args.num_iterations):
        logger.info("="*50)
        logger.info(f"开始第{gen+1}代训练与生成 (使用生成数据)")
        logger.info(f"加载模型: {prev_gen_model_dir}/best.ckpt")
        logger.info(f"加载数据: {prev_gen_data_file}")
        logger.info("="*50)
        
        current_gen_model_dir = f"{exp_dir}/models/gen{gen}"
        current_gen_ppl_file = f"{exp_dir}/perplexities/gen{gen}_ppl.pkl"
        current_gen_data_file = f"{exp_dir}/generations/gen{gen}_data.pkl"
        current_gen_gen_ppl_file = f"{exp_dir}/perplexities/gen{gen}_gen_ppl.pkl"  # 添加生成困惑度文件
        
        cmd_current_gen = (
            f"python {main_py_path} "
            f"--model_tag {args.model_tag} "
            f"--load-name {prev_gen_model_dir}/best.ckpt "
            f"--save-name {current_gen_model_dir} "
            f"--save_perplexity {current_gen_ppl_file} "
            f"--load-generate {prev_gen_data_file} "
            f"--generate {current_gen_data_file} "
            f"--save_gen_perplexity {current_gen_gen_ppl_file} "  # 添加生成困惑度保存路径
            f"--max-epochs {args.max_epochs} "
            f"--batch-size 256 "  # 固定训练批次大小为256
            f"--gen_batch_size 1024 "  # 固定生成批次大小为1024
            f"--learning-rate {args.learning_rate} "
            f"--seed {args.seed} "
            f"--pretrained "
            f"--generate_percentage 1.0 "  # 确保使用100%生成数据
            f"--original_percentage 0.0 "  # 确保不使用原始数据
            f"--num_generations {args.num_generations} "
            f"--log_interval 50 "  # 降低日志频率以提高性能
            f"--version_name {experiment_id}_gen{gen} "
            f"--text_data_format "  # 新增参数，指示数据是文本格式需要重新tokenize
            f"--num_workers {args.num_workers} "  # 使用参数中的工作线程数
            f"--memory_limit 0.9 "  # 降低内存使用限制，为系统预留更多空间
            f"--gradient_accumulation_steps {args.gradient_accumulation_steps}"  # 添加梯度累积
        )
        
        # 如果启用了半精度，添加相应参数
        if args.half_precision:
            cmd_current_gen += " --fp16"
        
        if not run_command(cmd_current_gen, f"第{gen+1}代模型训练与数据生成"):
            logger.error(f"第{gen+1}代模型训练与数据生成失败，终止实验")
            break
        
        # 检查生成的数据文件
        if not os.path.exists(current_gen_data_file) or os.path.getsize(current_gen_data_file) < 1000:
            logger.error(f"第{gen+1}代数据生成失败，文件不存在或过小")
            break
        
        # 打印文件大小信息
        file_size_mb = os.path.getsize(current_gen_data_file) / (1024 ** 2)  # MB
        logger.info(f"第{gen+1}代生成数据文件大小: {file_size_mb:.2f} MB")
        
        # 每一代结束后进行额外的内存清理
        try:
            # 强制执行Python垃圾回收
            import gc
            gc.collect()
            logger.info(f"第{gen+1}代完成，已执行Python垃圾回收")
            
            # 如果是Linux系统，可以尝试清理系统缓存
            if os.name == 'posix':
                # 同步文件系统缓存
                os.system('sync')
                logger.info("已执行sync命令，将缓存数据写入磁盘")
                
                # 可以考虑使用更激进的系统缓存清理（需要root权限）
                # 注意：这可能会影响系统性能，请谨慎使用
                # os.system('echo 3 > /proc/sys/vm/drop_caches')
                # logger.info("已尝试清理系统缓存")
        except Exception as e:
            logger.warning(f"内存清理尝试失败: {str(e)}")
        
        # 更新引用，为下一代做准备
        prev_gen_model_dir = current_gen_model_dir
        prev_gen_data_file = current_gen_data_file
        
        # 在每代之间添加短暂休息，让系统有时间完成清理
        time.sleep(5)
        logger.info(f"第{gen+1}代完成，休息5秒后继续下一代")
    
    # 实验完成
    logger.info("="*50)
    logger.info(f"实验 {experiment_id} 完成")
    logger.info("="*50)
    
    # 生成简单的实验摘要
    summary = (
        f"实验ID: {experiment_id}\n"
        f"模型: {args.model_tag}\n"
        f"迭代次数: {args.num_iterations}\n"
        f"每代训练轮数: {args.max_epochs}\n"
        f"批次大小: {args.batch_size}\n"
        f"生成批次大小: {args.gen_batch_size}\n"
        f"学习率: {args.learning_rate}\n"
        f"随机种子: {args.seed}\n"
        f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    )
    
    with open(f"{exp_dir}/summary.txt", "w") as f:
        f.write(summary)
    
    logger.info(f"实验摘要已保存到 {exp_dir}/summary.txt")

if __name__ == "__main__":
    main()