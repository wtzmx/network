import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from public_goods_game import PublicGoodsGame
from typing import Dict, List, Tuple
import time
from tqdm import tqdm

class RegularNetworkExperiment:
    def __init__(self):
        self.N = 50  # 智能体数量
        self.alpha = 0.3  # 学习率
        self.max_rounds = 10_000_000  # 最大轮数
        self.epsilon = 1e-4  # 收敛阈值
        self.num_trials = 500  # 每个参数的实验次数
        self.neighbor_list = [2, 4, 6, 8]  # 邻居连接数
        
    def create_regular_network(self, num_neighbors: int) -> nx.Graph:
        """创建环形规则网络"""
        # 确保邻居数是偶数
        k = num_neighbors if num_neighbors % 2 == 0 else num_neighbors + 1
        return nx.circulant_graph(self.N, range(1, k//2 + 1))
        
    def analyze_network(self, network: nx.Graph) -> Dict:
        """分析网络特征"""
        return {
            "avg_degree": np.mean([d for n, d in network.degree()]),
            "density": nx.density(network),
            "clustering": nx.average_clustering(network),
            "diameter": nx.diameter(network)
        }
        
    def run_single_trial(self, num_neighbors: int) -> Dict:
        """运行单次实验"""
        # 创建网络
        network = self.create_regular_network(num_neighbors)
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
            "num_neighbors": num_neighbors,
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
        
        for k in self.neighbor_list:
            print(f"\n运行邻居数量 k = {k} 的实验...")
            for i in tqdm(range(self.num_trials)):
                result = self.run_single_trial(k)
                results.append(result)
                
        print(f"\n实验完成，总耗时: {time.time() - start_time:.2f} 秒")
        return results
        
    def plot_results(self, results: List[Dict]):
        """可视化实验结果"""
        plt.figure(figsize=(20, 15))
        
        # 1. 收敛时间分布
        plt.subplot(231)
        for k in self.neighbor_list:
            k_results = [res for res in results if res['num_neighbors'] == k]
            conv_times = [res['convergence_time'] for res in k_results]
            plt.hist(conv_times, bins=50, alpha=0.5, label=f'k={k}')
        plt.xlabel('收敛时间')
        plt.ylabel('频次')
        plt.title('收敛时间分布')
        plt.legend()
        
        # 2. 最终状态分布
        plt.subplot(232)
        states = ['cooperation', 'defection', 'not_converged']
        x = np.arange(len(self.neighbor_list))
        width = 0.25
        
        for i, state in enumerate(states):
            rates = []
            for k in self.neighbor_list:
                k_results = [res for res in results if res['num_neighbors'] == k]
                rate = len([res for res in k_results if res['final_state'] == state]) / self.num_trials
                rates.append(rate)
            plt.bar(x + i*width, rates, width, label=state)
            
        plt.xlabel('邻居数量')
        plt.ylabel('比例')
        plt.title('最终状态分布')
        plt.xticks(x + width, self.neighbor_list)
        plt.legend()
        
        # 3. 典型收敛轨迹
        plt.subplot(233)
        for k in self.neighbor_list:
            k_results = [res for res in results if res['num_neighbors'] == k]
            if k_results:
                sample = k_results[0]
                plt.plot(sample['cooperation_history'][:1000], label=f'k={k}')
        plt.xlabel('回合数')
        plt.ylabel('合作率')
        plt.title('典型合作率演化轨迹')
        plt.legend()
        
        # 4. 平均收敛时间与邻居数量的关系
        plt.subplot(234)
        avg_times = []
        std_times = []
        for k in self.neighbor_list:
            k_results = [res for res in results if res['num_neighbors'] == k]
            times = [res['convergence_time'] for res in k_results]
            avg_times.append(np.mean(times))
            std_times.append(np.std(times))
        plt.errorbar(self.neighbor_list, avg_times, yerr=std_times, fmt='o-')
        plt.xlabel('邻居数量')
        plt.ylabel('平均收敛时间')
        plt.title('收敛时间与邻居数量的关系')
        
        # 5. 网络特征分析
        plt.subplot(235)
        for k in self.neighbor_list:
            k_results = [res for res in results if res['num_neighbors'] == k]
            clustering = [res['network_stats']['clustering'] for res in k_results]
            beliefs = [np.mean(res['final_beliefs']) for res in k_results]
            plt.scatter(clustering, beliefs, alpha=0.5, label=f'k={k}')
        plt.xlabel('聚类系数')
        plt.ylabel('最终平均信念')
        plt.title('聚类系数与最终信念的关系')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('regular_network_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 打印统计结果
        print("\n实验统计结果:")
        for k in self.neighbor_list:
            k_results = [res for res in results if res['num_neighbors'] == k]
            coop_rate = len([res for res in k_results if res['final_state'] == 'cooperation']) / self.num_trials
            defect_rate = len([res for res in k_results if res['final_state'] == 'defection']) / self.num_trials
            not_conv_rate = len([res for res in k_results if res['final_state'] == 'not_converged']) / self.num_trials
            
            print(f"\n邻居数量 k = {k}:")
            print(f"合作比例: {coop_rate:.3f}")
            print(f"背叛比例: {defect_rate:.3f}")
            print(f"未收敛比例: {not_conv_rate:.3f}")
            print(f"平均收敛时间: {np.mean([res['convergence_time'] for res in k_results]):.1f}")

if __name__ == "__main__":
    experiment = RegularNetworkExperiment()
    results = experiment.run_experiment()
    experiment.plot_results(results) 