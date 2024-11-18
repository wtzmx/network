# 公共物品博弈网络实验

本项目实现了论文中描述的全或无公共物品博弈（all-or-nothing public goods game）在网络环境下的四个主要实验。

## 环境配置

### 方法一：使用配置文件（推荐）

项目提供了完整的依赖配置文件，可以直接使用pip安装：

```bash
pip install -r requirements.txt
```

### 方法二：手动安装依赖

如果不想使用配置文件，也可以手动安装核心依赖：

```bash
pip install numpy networkx matplotlib seaborn tqdm
```

### Python版本要求
- 建议使用 Python >= 3.8

## 项目结构

```
.
├── requirements.txt        # 环境依赖配置文件
├── public_goods_game.py   # 核心博弈模型实现
├── example.py             # 基础示例运行脚本
├── experiments/
│   ├── convergence_experiment.py        # 实验一：基本收敛性实验
│   ├── geometric_network_experiment.py   # 实验二：随机几何网络实验
│   ├── regular_network_experiment.py     # 实验三：规则网络实验
│   └── stability_analysis.py            # 实验四：稳定性分析实验
└── results/                # 实验结果输出目录
```

## 实验内容

### 实验一：基本收敛性实验

目的：验证理论结果（系统最终会收敛到全合作或全不合作状态）

```bash
python experiments/convergence_experiment.py
```

**实验设置：**
- 总人数：50人
- 学习率：0.3
- 最大运行轮数：1000万轮
- 收敛阈值：10⁻⁴

**输出：**
- 收敛时间分布图
- 最终状态比例图
- 典型收敛轨迹图

### 实验二：随机几何网络实验

目的：研究不同网络密度对合作的影响

```bash
python experiments/geometric_network_experiment.py
```

**实验设置：**
- 网络类型：随机几何网络
- 网络密度参数(rg)：[0.15, 0.2, 0.25, 0.3]
- 每种参数运行500次

**输出：**
- 网络密度与合作率关系图
- 网络特征分析图
- 收敛时间统计

### 实验三：规则网络实验

目的：研究规则网络结构下的系统行为

```bash
python experiments/regular_network_experiment.py
```

**实验设置：**
- 网络类型：环形规则网络
- 邻居数量：[2, 4, 6, 8]
- 每种参数运行500次

**输出：**
- 收敛时间分布
- 最终状态分布
- 网络特征分析

### 实验四：稳定性分析实验

目的：研究网络结构对系统稳定性的影响

```bash
python experiments/stability_analysis.py
```

**实验设置：**
- 比较随机网络和规则网络
- 观察中间状态
- 分析收敛时间分布

**输出：**
- 收敛时间分布对比
- 信念演化轨迹
- 中间状态分析
- 网络结构影响分析

## 核心参数配置

```python
# 在 public_goods_game.py 中设置
PARAMS = {
    'N': 50,           # 智能体数量
    'alpha': 0.3,      # 学习率
    'max_rounds': 1e7, # 最大轮数
    'epsilon': 1e-4    # 收敛阈值
}
```

## 可视化结果

所有实验都会生成详细的可视化结果，包括：

1. 时间序列图
   - 合作率随时间变化
   - 信念值演化轨迹
   
2. 网络结构图
   - 节点颜色表示信念值
   - 边表示智能体间联系

3. 统计分布图
   - 收敛时间分布
   - 最终状态分布

## 数据分析输出

每个实验会生成以下统计结果：

```
results/
├── convergence/
│   ├── convergence_times.csv
│   ├── final_states.csv
│   └── evolution_trajectories.csv
├── geometric/
│   ├── cooperation_rates.csv
│   ├── network_metrics.csv
│   └── convergence_stats.csv
└── ...
```

## 注意事项

1. ⚠️ 实验运行时间可能较长，特别是收敛性实验
2. 💡 可以通过修改参数来调整实验规模
3. 📊 所有结果会自动保存为高质量图片

## 扩展建议

### 1. 自定义收益函数

```python
# in public_goods_game.py
def custom_payoff_function(cooperation_rate, threshold):
    """
    自定义收益生成函数
    Args:
        cooperation_rate: float, 合作率
        threshold: float, 阈值
    Returns:
        float: 收益值
    """
    if cooperation_rate >= threshold:
        return 1.0
    return 0.0
```

### 2. 添加新的网络密度参数

```python
# in geometric_network_experiment.py
DENSITY_PARAMS = [0.15, 0.2, 0.25, 0.3, 0.35]  # 添加新的密度参数
```

### 3. 扩展网络类型

```python
# in stability_analysis.py
def create_custom_network(N, param):
    """
    创建自定义网络结构
    """
    G = nx.Graph()
    # 添加自定义网络生成逻辑
    return G
```