import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from public_goods_game import PublicGoodsGame
from typing import Dict, List, Tuple
import time

class ConvergenceExperiment:
    def __init__(self):
        self.N = 50  # 智能体数量
        self.alpha = 0.3  # 学习率
        self.max_rounds = 10_000_000  # 最大轮数
        self.epsilon = 1e-4  # 收敛阈值
        self.num_trials = 500  # 实验重复次数
        
    def create_network(self, radius: float) -> nx.Graph:
        """创建随机几何网络，确保连通性"""
        while True:
            G = nx.random_geometric_graph(self.N, radius)
            if nx.is_connected(G):
                return G
    
    def check_convergence(self, beliefs: np.ndarray) -> Tuple[bool, str]:
        """检查是否收敛到全合作或全不合作状态"""
        if np.all(beliefs < self.epsilon):
            return True, "defection"
        elif np.all(beliefs > 1 - self.epsilon):
            return True, "cooperation"
        return False, "none"
    
    def run_single_trial(self, radius: float) -> Dict:
        """运行单次实验"""
        # 初始化游戏
        network = self.create_network(radius)
        game = PublicGoodsGame(self.N, self.alpha)
        game.set_network(network)
        
        # 记录数据
        convergence_time = self.max_rounds
        final_state = "not_converged"
        belief_history = []
        
        # 运行博弈
        for t in range(self.max_rounds):
            actions, payoffs = game.play_round()
            belief_history.append(np.mean(game.beliefs))
            
            # 检查收敛
            converged, state = self.check_convergence(game.beliefs)
            if converged:
                convergence_time = t
                final_state = state
                break
                
        return {
            "convergence_time": convergence_time,
            "final_state": final_state,
            "belief_history": belief_history,
            "network_stats": {
                "avg_degree": np.mean([d for n, d in network.degree()]),
                "density": nx.density(network),
                "triangles": sum(nx.triangles(network).values()) / 3
            }
        }
    
    def run_experiment(self, radius: float = 0.25) -> List[Dict]:
        """运行多次实验并收集数据"""
        results = []
        start_time = time.time()
        
        for i in range(self.num_trials):
            if i % 10 == 0:
                print(f"Running trial {i}/{self.num_trials}")
            result = self.run_single_trial(radius)
            results.append(result)
            
        print(f"Experiment completed in {time.time() - start_time:.2f} seconds")
        return results
    
    def analyze_results(self, results: List[Dict]):
        """分析实验结果"""
        # 统计最终状态
        final_states = [r["final_state"] for r in results]
        state_counts = {
            "cooperation": final_states.count("cooperation"),
            "defection": final_states.count("defection"),
            "not_converged": final_states.count("not_converged")
        }
        
        # 计算收敛时间统计
        conv_times = [r["convergence_time"] for r in results 
                     if r["final_state"] != "not_converged"]
        
        # 绘制结果
        plt.figure(figsize=(15, 10))
        
        # 1. 收敛时间分布
        plt.subplot(221)
        plt.hist(conv_times, bins=50)
        plt.title("Distribution of Convergence Times")
        plt.xlabel("Time steps")
        plt.ylabel("Frequency")
        
        # 2. 最终状态比例
        plt.subplot(222)
        plt.pie([v for v in state_counts.values()],
                labels=[k for k in state_counts.keys()],
                autopct='%1.1f%%')
        plt.title("Final States Distribution")
        
        # 3. 典型收敛轨迹
        plt.subplot(223)
        for i in range(min(5, len(results))):
            plt.plot(results[i]["belief_history"][:1000], 
                    label=f'Trial {i+1}')
        plt.title("Typical Convergence Trajectories")
        plt.xlabel("Time steps")
        plt.ylabel("Average Belief")
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('convergence_analysis.png')
        plt.show()
        
        # 打印统计信息
        print("\nExperiment Statistics:")
        print(f"Total trials: {self.num_trials}")
        print("\nFinal states distribution:")
        for state, count in state_counts.items():
            print(f"{state}: {count/self.num_trials*100:.1f}%")
        
        if conv_times:
            print("\nConvergence time statistics:")
            print(f"Mean: {np.mean(conv_times):.1f}")
            print(f"Median: {np.median(conv_times):.1f}")
            print(f"Std: {np.std(conv_times):.1f}")

if __name__ == "__main__":
    experiment = ConvergenceExperiment()
    results = experiment.run_experiment()
    experiment.analyze_results(results) 