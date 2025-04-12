import os
import subprocess
import time
import sys

# 设置工作目录
data_dir = r"d:\Documents\100\action"
# 修改输出目录为results下的sem_analysis_results子目录
output_dir = r"d:\Documents\100\action\results\sem_analysis_results"
os.makedirs(output_dir, exist_ok=True)

# 设置matplotlib使用支持中文的字体
def setup_chinese_font():
    try:
        import matplotlib.pyplot as plt
        import matplotlib.font_manager as fm
        
        # 尝试设置微软雅黑字体（Windows系统常见字体）
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
        
        # 验证字体是否可用
        fonts = [f.name for f in fm.fontManager.ttflist]
        print("可用字体列表:")
        for font in ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']:
            if font in fonts:
                print(f"- {font} (可用)")
            else:
                print(f"- {font} (不可用)")
        
        # 创建字体配置文件
        with open(os.path.join(data_dir, "matplotlibrc"), "w", encoding="utf-8") as f:
            f.write("font.family: sans-serif\n")
            f.write("font.sans-serif: Microsoft YaHei, SimHei, Arial Unicode MS\n")
            f.write("axes.unicode_minus: False\n")
        
        print("已设置matplotlib使用中文字体")
    except Exception as e:
        print(f"设置中文字体时出错: {e}")

print("开始执行结构方程模型分析流程...")

# 设置中文字体
setup_chinese_font()

# 步骤1: 运行SEM模型构建和评估
print("\n步骤1: 运行SEM模型构建和评估...")
try:
    # 传递输出目录和字体配置作为环境变量
    env = os.environ.copy()
    env["SEM_OUTPUT_DIR"] = output_dir
    env["MATPLOTLIBRC"] = os.path.join(data_dir, "matplotlibrc")
    
    # 运行模型评估脚本
    result = subprocess.run(["python", os.path.join(data_dir, "sem_model_evaluation.py")], 
                           check=True, env=env, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("警告/错误信息:")
        print(result.stderr)
    print("SEM模型评估完成")
except subprocess.CalledProcessError as e:
    print(f"SEM模型评估过程中出错: {e}")
    print("错误输出:")
    print(e.stdout)
    print(e.stderr)
    if input("是否继续执行后续步骤? (y/n): ").lower() != 'y':
        sys.exit(1)
except Exception as e:
    print(f"SEM模型评估过程中出错: {e}")
    if input("是否继续执行后续步骤? (y/n): ").lower() != 'y':
        sys.exit(1)

# 步骤2: 运行交叉验证和多组分析
print("\n步骤2: 运行交叉验证和多组分析...")
try:
    # 传递输出目录和字体配置作为环境变量
    env = os.environ.copy()
    env["SEM_OUTPUT_DIR"] = output_dir
    env["MATPLOTLIBRC"] = os.path.join(data_dir, "matplotlibrc")
    
    # 运行交叉验证和多组分析脚本
    result = subprocess.run(["python", os.path.join(data_dir, "sem_cross_validation.py")], 
                           check=True, env=env, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("警告/错误信息:")
        print(result.stderr)
    print("交叉验证和多组分析完成")
except subprocess.CalledProcessError as e:
    print(f"交叉验证和多组分析过程中出错: {e}")
    print("错误输出:")
    print(e.stdout)
    print(e.stderr)
    if input("是否继续执行后续步骤? (y/n): ").lower() != 'y':
        sys.exit(1)
except Exception as e:
    print(f"交叉验证和多组分析过程中出错: {e}")
    if input("是否继续执行后续步骤? (y/n): ").lower() != 'y':
        sys.exit(1)

# 步骤3: 生成分析报告
print("\n步骤3: 生成分析报告...")
try:
    # 传递输出目录和字体配置作为环境变量
    env = os.environ.copy()
    env["SEM_OUTPUT_DIR"] = output_dir
    env["MATPLOTLIBRC"] = os.path.join(data_dir, "matplotlibrc")
    
    # 运行分析报告生成脚本
    result = subprocess.run(["python", os.path.join(data_dir, "sem_analysis_report.py")], 
                           check=True, env=env, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("警告/错误信息:")
        print(result.stderr)
    print("分析报告生成完成")
except subprocess.CalledProcessError as e:
    print(f"生成分析报告过程中出错: {e}")
    print("错误输出:")
    print(e.stdout)
    print(e.stderr)
except Exception as e:
    print(f"生成分析报告过程中出错: {e}")

print("\n分析流程完成！结果已保存到:", output_dir)
print("您可以查看以下关键文件:")
print(f"1. 模型比较结果: {os.path.join(output_dir, 'models_comparison.csv')}")
print(f"2. 效应分解结果: {os.path.join(output_dir, 'effects_decomposition.csv')}")
print(f"3. Bootstrap置信区间: {os.path.join(output_dir, 'bootstrap_confidence_intervals.csv')}")
print(f"4. 交叉验证结果: {os.path.join(output_dir, 'cross_validation_results.csv')}")
print(f"5. Run分组分析结果: {os.path.join(output_dir, 'run_comparison.csv')}")
print(f"6. 综合分析报告: {os.path.join(output_dir, 'sem_analysis_summary.md')}")