import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

# 蒙特卡洛检测内生性的代码
def monte_carlo_simulation(n=100, num_simulations=1000):
    """
    利用蒙特卡洛方法模拟内生性问题
    :param n: 每次模拟的数据点数量
    :param num_simulations: 模拟次数
    :return: 内生性检测结果
    """
    beta_true = 2  # 真正的参数值
    correlation_results = []  # 存储相关性结果

    for _ in range(num_simulations):
        np.random.seed()  # 每次模拟使用不同的随机种子
        
        # 随机生成变量 X
        X = np.random.normal(0, 1, n)
        
        # 随机生成误差项 e
        e = np.random.normal(0, 1, n)
        
        # 使用特定方式生成因变量 Y（这里引入内生性）
        # 令误差项与 X 存在某种关系来模拟内生性
        Y = beta_true * X + e + 0.5 * X  # 误差项 e 和 X 存在线性关系
        
        # 回归方程：Y = beta * X + u
        X_with_constant = sm.add_constant(X)  # 增加常数项（截距项）
        model = sm.OLS(Y, X_with_constant).fit()  # 使用OLS估计
        
        # 检查回归残差 u 是否与 X 存在相关性
        residuals = model.resid
        
        # 计算 X 和残差之间的相关性
        correlation = np.corrcoef(X, residuals)[0, 1]
        correlation_results.append(correlation)

    # 检测结果：观察相关性分布是否偏离零
    correlation_results = np.array(correlation_results)
    plt.hist(correlation_results, bins=30, edgecolor='k', alpha=0.7)
    plt.title("Correlation Between X and Residuals (Monte Carlo Simulation)")
    plt.xlabel("Correlation")
    plt.ylabel("Frequency")
    plt.show()

    print(f"平均相关性：{np.mean(correlation_results):.4f}")
    print(f"标准差：{np.std(correlation_results):.4f}")
    print("如果相关性显著偏离零，则可能存在内生性问题。")

# 运行蒙特卡洛检测
monte_carlo_simulation()
