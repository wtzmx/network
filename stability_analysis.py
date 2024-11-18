import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from public_goods_game import PublicGoodsGame
from typing import Dict, List, Tuple
import time
from tqdm import tqdm
import seaborn as sns

class StabilityAnalysis:
    def __init__(self):
        self.N = 50  # 智能体数量
        self.alpha = 0.3  # 学习率
        self.max_rounds = 1_000_000  # 最大轮数（减少以观察中间状态）
        self.epsilon = 1e-4  # 收敛阈值
        self.num_trials = 100  # 每种网络类型的实验次数
        
    def create_geometric_network(self, radius: float) -> nx.Graph:
        """创建随机几何网络"""
        while True:
            G = nx.random_geometric_graph(self.N, radius)
            if nx.is_connected(G):
                return G
                
    def create_regular_network(self, k: int) -> nx.Graph:
        """创建规则网络"""
        return nx.circulant_graph(self.N, range(1, k//2 + 1))
        
    def analyze_network(self, network: nx.Graph) -> Dict:
        """分析网络特征"""
        return {
            "avg_degree": np.mean([d for n, d in network.degree()]),
            "density": nx.density(network),
            "clustering": nx.average_clustering(network),
            "diameter": nx.diameter(network)
        }
        
    def run_single_trial(self, network: nx.Graph, network_type: str) -> Dict:
        """运行单次实验"""
        game = PublicGoodsGame(self.N, self.alpha)
        game.set_network(network)
        
        # 记录数据
        belief_history = []
        cooperation_history = []
        total_belief_history = []
        final_state = "not_converged"
        
        # 运行博弈
        for t in range(self.max_rounds):
            actions, payoffs = game.play_round()
            belief_history.append(np.mean(game.beliefs))
            cooperation_rate = sum(1 for a in actions if a == 'C') / len(actions)
            cooperation_history.append(cooperation_rate)
            total_belief_history.append(np.sum(game.beliefs))
            
            # 检查收敛
            if np.all(game.beliefs < self.epsilon):
                final_state = "defection"
                break
            elif np.all(game.beliefs > 1 - self.epsilon):
                final_state = "cooperation"
                break
                
        return {
            "network_type": network_type,
            "final_state": final_state,
            "network_stats": self.analyze_network(network),
            "belief_history": belief_history,
            "cooperation_history": cooperation_history,
            "total_belief_history": total_belief_history,
            "final_beliefs": game.beliefs.copy(),
            "convergence_time": len(belief_history)
        }
        
    def run_experiment(self):
        """运行完整实验"""
        results = []
        start_time = time.time()
        
        # 随机几何网络实验
        print("\n运行随机几何网络实验...")
        for i in tqdm(range(self.num_trials)):
            network = self.create_geometric_network(0.15)  # 使用较小的半径以观察中间状态
            result = self.run_single_trial(network, "geometric")
            results.append(result)
            
        # 规则网络实验
        print("\n运行规则网络实验...")
        for i in tqdm(range(self.num_trials)):
            network = self.create_regular_network(4)  # 使用4邻居的规则网络
            result = self.run_single_trial(network, "regular")
            results.append(result)
            
        print(f"\n实验完成，总耗时: {time.time() - start_time:.2f} 秒")
        return results
        
    def plot_results(self, results: List[Dict]):
        """可视化实验结果"""
        # 创建子图
        fig = plt.figure(figsize=(20, 15))
        
        # 1. 收敛时间分布对比
        plt.subplot(231)
        for network_type in ["geometric", "regular"]:
            type_results = [r for r in results if r['network_type'] == network_type]
            conv_times = [r['convergence_time'] for r in type_results]
            sns.kdeplot(conv_times, label=network_type)
        plt.xlabel('收敛时间')
        plt.ylabel('密度')
        plt.title('收敛时间分布对比')
        plt.legend()
        
        # 2. 典型信念演化轨迹
        plt.subplot(232)
        for network_type in ["geometric", "regular"]:
            type_results = [r for r in results if r['network_type'] == network_type]
            sample = type_results[0]
            plt.plot(sample['belief_history'][:1000], 
                    label=f'{network_type}')
        plt.xlabel('回合数')
        plt.ylabel('平均信念')
        plt.title('典型信念演化轨迹')
        plt.legend()
        
        # 3. 总信念随时间变化
        plt.subplot(233)
        for network_type in ["geometric", "regular"]:
            type_results = [r for r in results if r['network_type'] == network_type]
            sample = type_results[0]
            plt.plot(sample['total_belief_history'][:1000], 
                    label=f'{network_type}')
        plt.xlabel('回合数')
        plt.ylabel('总信念')
        plt.title('总信念随时间变化')
        plt.legend()
        
        # 4. 最终状态分布
        plt.subplot(234)
        data = []
        labels = []
        for network_type in ["geometric", "regular"]:
            type_results = [r for r in results if r['network_type'] == network_type]
            for state in ["cooperation", "defection", "not_converged"]:
                count = len([r for r in type_results if r['final_state'] == state])
                data.append(count / len(type_results))
                labels.append(f"{network_type}-{state}")
        plt.bar(range(len(data)), data)
        plt.xticks(range(len(data)), labels, rotation=45)
        plt.ylabel('比例')
        plt.title('最终状态分布')
        
        # 5. 中间状态分析
        plt.subplot(235)
        for network_type in ["geometric", "regular"]:
            type_results = [r for r in results if r['network_type'] == network_type]
            metastable_results = [r for r in type_results 
                                if r['final_state'] == "not_converged"]
            if metastable_results:
                beliefs = np.array([r['belief_history'] for r in metastable_results])
                mean_belief = np.mean(beliefs, axis=0)
                std_belief = np.std(beliefs, axis=0)
                plt.plot(mean_belief[:1000], label=f'{network_type}')
                plt.fill_between(range(len(mean_belief[:1000])),
                               mean_belief[:1000] - std_belief[:1000],
                               mean_belief[:1000] + std_belief[:1000],
                               alpha=0.2)
        plt.xlabel('回合数')
        plt.ylabel('平均信念')
        plt.title('中间状态信念演化')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('stability_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 打印统计结果
        print("\n稳定性分析结果:")
        for network_type in ["geometric", "regular"]:
            print(f"\n{network_type}网络:")
            type_results = [r for r in results if r['network_type'] == network_type]
            
            # 收敛时间统计
            conv_times = [r['convergence_time'] for r in type_results]
            print(f"平均收敛时间: {np.mean(conv_times):.1f}")
            print(f"收敛时间标准差: {np.std(conv_times):.1f}")
            
            # 最终状态统计
            for state in ["cooperation", "defection", "not_converged"]:
                count = len([r for r in type_results if r['final_state'] == state])
                print(f"{state}比例: {count/len(type_results):.3f}")
            
            # 中间状态统计
            metastable_results = [r for r in type_results 
                                if r['final_state'] == "not_converged"]
            if metastable_results:
                beliefs = np.array([r['belief_history'] for r in metastable_results])
                print(f"中间状态平均信念: {np.mean(beliefs):.3f}")
                print(f"中间状态信念波动: {np.std(beliefs):.3f}")

if __name__ == "__main__":
    analysis = StabilityAnalysis()
    results = analysis.run_experiment()
    analysis.plot_results(results) 