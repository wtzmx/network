import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from public_goods_game import PublicGoodsGame
from typing import Dict, List, Tuple
import time
import seaborn as sns
from tqdm import tqdm

class GeometricNetworkExperiment:
    def __init__(self):
        self.N = 50  # 智能体数量
        self.alpha = 0.3  # 学习率
        self.max_rounds = 10_000_000  # 最大轮数
        self.epsilon = 1e-4  # 收敛阈值
        self.num_trials = 500  # 每个参数的实验次数
        self.radius_list = [0.15, 0.20, 0.25, 0.30]  # 网络密度参数
        
    def create_geometric_network(self, radius: float) -> nx.Graph:
        """创建连通的随机几何网络"""
        while True:
            G = nx.random_geometric_graph(self.N, radius)
            if nx.is_connected(G):
                return G
                
    def analyze_network(self, network: nx.Graph) -> Dict:
        """分析网络特征"""
        return {
            "avg_degree": np.mean([d for n, d in network.degree()]),
            "density": nx.density(network),
            "clustering": nx.average_clustering(network),
            "triangles": sum(nx.triangles(network).values()) / 3
        }
        
    def run_single_trial(self, radius: float) -> Dict:
        """运行单次实验"""
        # 创建网络
        network = self.create_geometric_network(radius)
        network_stats = self.analyze_network(network)
        
        # 初始化游戏
        game = PublicGoodsGame(self.N, self.alpha)
        game.set_network(network)
        
        # 记录数据
        belief_history = []
        cooperation_history = []
        final_state = "not_converged"
        
        # 运行博弈
        for t in range(self.max_rounds):
            actions, payoffs = game.play_round()
            belief_history.append(np.mean(game.beliefs))
            cooperation_rate = sum(1 for a in actions if a == 'C') / len(actions)
            cooperation_history.append(cooperation_rate)
            
            # 检查收敛
            if np.all(game.beliefs < self.epsilon):
                final_state = "defection"
                break
            elif np.all(game.beliefs > 1 - self.epsilon):
                final_state = "cooperation"
                break
                
        return {
            "radius": radius,
            "final_state": final_state,
            "network_stats": network_stats,
            "final_beliefs": game.beliefs.copy(),
            "belief_history": belief_history,
            "cooperation_history": cooperation_history,
            "convergence_time": len(belief_history)
        }
        
    def run_experiment(self):
        """运行完整实验"""
        results = []
        start_time = time.time()
        
        for radius in self.radius_list:
            print(f"\n运行网络密度参数 rg = {radius} 的实验...")
            for i in tqdm(range(self.num_trials)):
                result = self.run_single_trial(radius)
                results.append(result)
                
        print(f"\n实验完成，总耗时: {time.time() - start_time:.2f} 秒")
        return results
        
    def plot_results(self, results: List[Dict]):
        """可视化实验结果"""
        # 1. 创建结果汇总
        summary = {}
        for r in self.radius_list:
            r_results = [res for res in results if res['radius'] == r]
            summary[r] = {
                'cooperation': len([res for res in r_results if res['final_state'] == 'cooperation']) / self.num_trials,
                'defection': len([res for res in r_results if res['final_state'] == 'defection']) / self.num_trials,
                'not_converged': len([res for res in r_results if res['final_state'] == 'not_converged']) / self.num_trials
            }
            
        # 创建图表
        plt.figure(figsize=(20, 15))
        
        # 1. 最终状态分布
        plt.subplot(231)
        x = np.arange(len(self.radius_list))
        width = 0.25
        
        plt.bar(x - width, [summary[r]['cooperation'] for r in self.radius_list], width, label='合作')
        plt.bar(x, [summary[r]['defection'] for r in self.radius_list], width, label='背叛')
        plt.bar(x + width, [summary[r]['not_converged'] for r in self.radius_list], width, label='未收敛')
        
        plt.xlabel('网络密度参数 (rg)')
        plt.ylabel('比例')
        plt.title('不同网络密度下的最终状态分布')
        plt.xticks(x, self.radius_list)
        plt.legend()
        
        # 2. 平均度与最终信念的关系
        plt.subplot(232)
        for r in self.radius_list:
            r_results = [res for res in results if res['radius'] == r]
            degrees = [res['network_stats']['avg_degree'] for res in r_results]
            beliefs = [np.mean(res['final_beliefs']) for res in r_results]
            plt.scatter(degrees, beliefs, alpha=0.5, label=f'rg={r}')
        
        plt.xlabel('平均度')
        plt.ylabel('最终平均信念')
        plt.title('平均度与最终信念的关系')
        plt.legend()
        
        # 3. 聚类系数与最终信念的关系
        plt.subplot(233)
        for r in self.radius_list:
            r_results = [res for res in results if res['radius'] == r]
            clustering = [res['network_stats']['clustering'] for res in r_results]
            beliefs = [np.mean(res['final_beliefs']) for res in r_results]
            plt.scatter(clustering, beliefs, alpha=0.5, label=f'rg={r}')
            
        plt.xlabel('聚类系数')
        plt.ylabel('最终平均信念')
        plt.title('聚类系数与最终信念的关系')
        plt.legend()
        
        # 4. 收敛时间分布
        plt.subplot(234)
        for r in self.radius_list:
            r_results = [res for res in results if res['radius'] == r]
            conv_times = [res['convergence_time'] for res in r_results]
            plt.hist(conv_times, bins=50, alpha=0.5, label=f'rg={r}')
            
        plt.xlabel('收敛时间')
        plt.ylabel('频次')
        plt.title('收敛时间分布')
        plt.legend()
        
        # 5. 典型合作率演化轨迹
        plt.subplot(235)
        for r in self.radius_list:
            r_results = [res for res in results if res['radius'] == r]
            if r_results:
                # 选择一个典型样本
                sample = r_results[0]
                plt.plot(sample['cooperation_history'][:1000], label=f'rg={r}')
                
        plt.xlabel('回合数')
        plt.ylabel('合作率')
        plt.title('典型合作率演化轨迹')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('geometric_network_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 打印统计结果
        print("\n实验统计结果:")
        for r in self.radius_list:
            print(f"\n网络密度参数 rg = {r}:")
            print(f"合作比例: {summary[r]['cooperation']:.3f}")
            print(f"背叛比例: {summary[r]['defection']:.3f}")
            print(f"未收敛比例: {summary[r]['not_converged']:.3f}")

if __name__ == "__main__":
    experiment = GeometricNetworkExperiment()
    results = experiment.run_experiment()
    experiment.plot_results(results) 