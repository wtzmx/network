import networkx as nx
import matplotlib.pyplot as plt
from public_goods_game import PublicGoodsGame

def create_random_geometric_network(N: int, radius: float) -> nx.Graph:
    """创建随机几何网络"""
    while True:
        G = nx.random_geometric_graph(N, radius)
        if nx.is_connected(G):
            return G

def run_simulation(N: int = 50, radius: float = 0.25, num_rounds: int = 1000):
    """运行模拟实验"""
    print(f"\n{'='*50}")
    print(f"开始模拟实验：")
    print(f"智能体数量: {N}")
    print(f"网络半径: {radius}")
    print(f"模拟回合数: {num_rounds}")
    print(f"{'='*50}\n")
    
    # 创建网络
    network = create_random_geometric_network(N, radius)
    print(f"网络创建完成:")
    print(f"节点数: {network.number_of_nodes()}")
    print(f"边数: {network.number_of_edges()}")
    print(f"平均度: {sum(dict(network.degree()).values())/N:.2f}")
    
    # 创建博弈模型
    game = PublicGoodsGame(N, alpha=0.3)
    game.set_network(network)
    
    # 运行模拟
    print("\n开始运行博弈...")
    for i in range(num_rounds):
        if i % (num_rounds//10) == 0:
            print(f"已完成 {i/num_rounds*100:.1f}%")
        actions, payoffs = game.play_round()
    print("模拟完成！\n")
    
    # 输出统计结果
    stats = game.get_statistics()
    print("统计结果:")
    for key, value in stats.items():
        print(f"{key}: {value:.4f}")
    
    # 绘制结果
    print("\n正在生成可视化结果...")
    game.plot_results()

if __name__ == "__main__":
    run_simulation() 