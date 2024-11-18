import numpy as np
import networkx as nx
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt

class PublicGoodsGame:
    def __init__(self, N: int, alpha: float = 0.3):
        """
        初始化公共物品博弈模型
        
        参数:
        N: int - 智能体数量
        alpha: float - 学习率
        """
        self.N = N  
        self.alpha = alpha
        self.beliefs = np.ones(N) * 0.5  
        self.network = None
        self.history = {
            'beliefs': [],
            'actions': [],
            'payoffs': [],
            'cooperation_rates': []
        }
        
    def set_network(self, network: nx.Graph):
        """设置网络结构"""
        self.network = network
        
    def generate_payoff(self) -> float:
        """生成随机收益λ"""
        return np.random.uniform(1.0, 3.0)
        
    def agent_action(self, agent_id: int, group_size: int, lambda_i: float) -> str:
        """决定智能体的行动"""
        if group_size <= 1:
            return 'C'
            
        belief = self.beliefs[agent_id]
        u_c = lambda_i * (belief ** (group_size - 1))
        u_d = 1
        
        return 'C' if u_c >= u_d else 'D'
    
    def update_belief(self, agent_id: int, group: List[int], actions: List[str]):
        """更新智能体的信念"""
        if len(group) <= 1:
            return
            
        others_actions = [a for i, a in enumerate(actions) if group[i] != agent_id]
        if not others_actions:
            return
            
        cooperation_ratio = sum(1 for a in others_actions if a == 'C') / len(others_actions)
        self.beliefs[agent_id] = (1 - self.alpha) * self.beliefs[agent_id] + self.alpha * cooperation_ratio
        
    def play_round(self) -> Tuple[List[str], List[float]]:
        """进行一轮博弈"""
        focal_agent = np.random.randint(0, self.N)
        
        if self.network is None:
            group = list(range(self.N))
        else:
            neighbors = list(self.network.neighbors(focal_agent))
            group = neighbors + [focal_agent] if neighbors else [focal_agent]
            
        group_size = len(group)
        lambdas = [self.generate_payoff() for _ in group]
        actions = [self.agent_action(i, group_size, l) for i, l in zip(group, lambdas)]
        
        payoffs = []
        for i, action in enumerate(actions):
            if action == 'D':
                payoffs.append(1)
            else:
                if all(a == 'C' for a in actions):
                    payoffs.append(lambdas[i])
                else:
                    payoffs.append(0)
                    
        for i, agent_id in enumerate(group):
            self.update_belief(agent_id, group, actions)
            
        # 记录历史数据
        self.history['beliefs'].append(self.beliefs.copy())
        self.history['actions'].append(actions)
        self.history['payoffs'].append(payoffs)
        self.history['cooperation_rates'].append(sum(1 for a in actions if a == 'C') / len(actions))
            
        return actions, payoffs

    def plot_results(self):
        """绘制结果分析图表"""
        fig = plt.figure(figsize=(15, 10))
        
        # 绘制合作率变化
        ax1 = plt.subplot(221)
        ax1.plot(self.history['cooperation_rates'])
        ax1.set_title('合作率随时间变化')
        ax1.set_xlabel('回合数')
        ax1.set_ylabel('合作率')
        
        # 绘制平均信念变化
        ax2 = plt.subplot(222)
        beliefs_array = np.array(self.history['beliefs'])
        mean_beliefs = beliefs_array.mean(axis=1)
        ax2.plot(mean_beliefs)
        ax2.set_title('平均信念随时间变化')
        ax2.set_xlabel('回合数')
        ax2.set_ylabel('平均信念')
        
        # 绘制平均收益变化
        ax3 = plt.subplot(223)
        payoffs_array = np.array([np.mean(p) for p in self.history['payoffs']])
        ax3.plot(payoffs_array)
        ax3.set_title('平均收益随时间变化')
        ax3.set_xlabel('回合数')
        ax3.set_ylabel('平均收益')
        
        # 绘制最终网络状态
        if self.network is not None:
            ax4 = plt.subplot(224)
            pos = nx.spring_layout(self.network)
            
            # 创建一个颜色映射对象
            norm = plt.Normalize(vmin=0, vmax=1)
            sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlBu, norm=norm)
            sm.set_array([])
            
            # 绘制网络
            nx.draw(self.network, pos, 
                   node_color=[self.beliefs[i] for i in range(self.N)],
                   node_size=500,
                   cmap=plt.cm.RdYlBu,
                   with_labels=True,
                   ax=ax4)
            
            # 添加颜色条
            plt.colorbar(sm, ax=ax4, label='信念值')
            ax4.set_title('最终网络状态\n(节点颜色表示信念值)')
        
        plt.tight_layout()
        
        # 保存图片
        plt.savefig('simulation_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def get_statistics(self) -> Dict:
        """获取统计数据"""
        stats = {
            '平均合作率': np.mean(self.history['cooperation_rates']),
            '最终合作率': self.history['cooperation_rates'][-1],
            '平均信念': np.mean(self.beliefs),
            '信念标准差': np.std(self.beliefs),
            '平均收益': np.mean([np.mean(p) for p in self.history['payoffs']]),
            '收敛回合数': len(self.history['cooperation_rates'])
        }
        return stats