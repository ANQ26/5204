"""
智能体模块 - 定义与环境交互的智能体
"""

import random
import math
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from collections import defaultdict
import json
from abc import ABC, abstractmethod


@dataclass
class Action:
    tool: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    reasoning: str = ""


class Agent(ABC):
    """
    智能体基类
    
    核心功能：
    1. 观察环境状态
    2. 选择动作（工具调用）
    3. 学习从经验中改进
    """
    
    def __init__(self, name: str = "Agent"):
        self.name = name
        self.episode_rewards = []
        self.total_steps = 0
    
    @abstractmethod
    def act(self, observation: Dict[str, Any]) -> Optional[Action]:
        pass
    
    @abstractmethod
    def observe(
        self,
        observation: Dict[str, Any],
        action: Action,
        reward: float,
        next_observation: Dict[str, Any],
        done: bool
    ):
        pass
    
    def reset(self):
        self.episode_rewards = []
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "total_steps": self.total_steps,
            "episode_count": len(self.episode_rewards),
            "average_reward": sum(self.episode_rewards) / len(self.episode_rewards) if self.episode_rewards else 0.0,
            "max_reward": max(self.episode_rewards) if self.episode_rewards else 0.0,
            "min_reward": min(self.episode_rewards) if self.episode_rewards else 0.0
        }


class RandomAgent(Agent):
    """
    随机智能体 - 随机选择可用工具
    """
    
    def __init__(self, name: str = "RandomAgent"):
        super().__init__(name)
    
    def act(self, observation: Dict[str, Any]) -> Optional[Action]:
        available_tools = observation.get("available_tools", [])
        
        if not available_tools:
            return None
        
        selected_tool = random.choice(available_tools)
        
        return Action(
            tool=selected_tool,
            parameters={},
            confidence=0.5,
            reasoning="随机选择工具"
        )
    
    def observe(
        self,
        observation: Dict[str, Any],
        action: Action,
        reward: float,
        next_observation: Dict[str, Any],
        done: bool
    ):
        self.total_steps += 1
        if done:
            self.episode_rewards.append(reward)


class QLearningAgent(Agent):
    """
    Q学习智能体 - 基于表格的Q学习算法
    
    核心特点：
    1. 使用Q表存储状态-动作价值
    2. ε-greedy策略进行探索
    3. 从经验中学习最优策略
    """
    
    def __init__(
        self,
        name: str = "QLearningAgent",
        learning_rate: float = 0.1,
        discount_factor: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01
    ):
        super().__init__(name)
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        self.q_table: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.action_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    
    def _get_state_key(self, observation: Dict[str, Any]) -> str:
        state_vars = observation.get("state_variables", {})
        current_state = observation.get("current_state", "unknown")
        
        key_parts = [f"state={current_state}"]
        for var_name, value in sorted(state_vars.items()):
            if isinstance(value, (int, float, bool, str)):
                key_parts.append(f"{var_name}={value}")
        
        return "|".join(key_parts)
    
    def _get_available_actions(self, observation: Dict[str, Any]) -> List[str]:
        return observation.get("available_tools", [])
    
    def act(self, observation: Dict[str, Any]) -> Optional[Action]:
        state_key = self._get_state_key(observation)
        available_actions = self._get_available_actions(observation)
        
        if not available_actions:
            return None
        
        if random.random() < self.epsilon:
            selected_tool = random.choice(available_actions)
            reasoning = "探索：随机选择工具"
            confidence = 0.3
        else:
            q_values = self.q_table[state_key]
            if q_values:
                max_q = max(q_values.values())
                best_actions = [a for a in available_actions if q_values.get(a, 0) == max_q]
                selected_tool = random.choice(best_actions) if best_actions else random.choice(available_actions)
                reasoning = f"利用：选择Q值最高的工具 (Q={max_q:.2f})"
                confidence = 0.8
            else:
                selected_tool = random.choice(available_actions)
                reasoning = "无Q值信息，随机选择"
                confidence = 0.3
        
        return Action(
            tool=selected_tool,
            parameters={},
            confidence=confidence,
            reasoning=reasoning
        )
    
    def observe(
        self,
        observation: Dict[str, Any],
        action: Action,
        reward: float,
        next_observation: Dict[str, Any],
        done: bool
    ):
        state_key = self._get_state_key(observation)
        next_state_key = self._get_state_key(next_observation)
        action_tool = action.tool
        
        current_q = self.q_table[state_key].get(action_tool, 0.0)
        
        if next_state_key in self.q_table and self.q_table[next_state_key]:
            max_next_q = max(self.q_table[next_state_key].values())
        else:
            max_next_q = 0.0
        
        target = reward + self.discount_factor * max_next_q * (not done)
        new_q = current_q + self.learning_rate * (target - current_q)
        
        self.q_table[state_key][action_tool] = new_q
        
        self.action_counts[state_key][action_tool] += 1
        self.total_steps += 1
        
        if done:
            self.episode_rewards.append(reward)
            self._decay_epsilon()
    
    def _decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def get_q_table_size(self) -> int:
        return sum(len(actions) for actions in self.q_table.values())
    
    def save(self, filepath: str):
        data = {
            "name": self.name,
            "learning_rate": self.learning_rate,
            "discount_factor": self.discount_factor,
            "epsilon": self.epsilon,
            "epsilon_decay": self.epsilon_decay,
            "epsilon_min": self.epsilon_min,
            "q_table": dict(self.q_table),
            "action_counts": {k: dict(v) for k, v in self.action_counts.items()},
            "total_steps": self.total_steps,
            "episode_rewards": self.episode_rewards
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def load(self, filepath: str):
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.name = data.get("name", self.name)
        self.learning_rate = data.get("learning_rate", self.learning_rate)
        self.discount_factor = data.get("discount_factor", self.discount_factor)
        self.epsilon = data.get("epsilon", self.epsilon)
        self.epsilon_decay = data.get("epsilon_decay", self.epsilon_decay)
        self.epsilon_min = data.get("epsilon_min", self.epsilon_min)
        
        self.q_table = defaultdict(dict, data.get("q_table", {}))
        self.action_counts = defaultdict(
            lambda: defaultdict(int),
            {k: defaultdict(int, v) for k, v in data.get("action_counts", {}).items()}
        )
        self.total_steps = data.get("total_steps", 0)
        self.episode_rewards = data.get("episode_rewards", [])


class REINFORCEAgent(Agent):
    """
    REINFORCE智能体 - 策略梯度方法
    
    核心特点：
    1. 直接参数化策略函数
    2. 使用蒙特卡洛方法估计梯度
    3. 适合离散动作空间
    """
    
    def __init__(
        self,
        name: str = "REINFORCEAgent",
        learning_rate: float = 0.01,
        discount_factor: float = 0.99
    ):
        super().__init__(name)
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        
        self.policy: Dict[str, Dict[str, float]] = defaultdict(
            lambda: defaultdict(lambda: 1.0)
        )
        self.episode_buffer: List[Tuple[str, str, float]] = []
    
    def _get_state_key(self, observation: Dict[str, Any]) -> str:
        state_vars = observation.get("state_variables", {})
        current_state = observation.get("current_state", "unknown")
        
        key_parts = [f"state={current_state}"]
        for var_name, value in sorted(state_vars.items()):
            if isinstance(value, (int, float, bool, str)):
                key_parts.append(f"{var_name}={value}")
        
        return "|".join(key_parts)
    
    def _normalize_policy(self, state_key: str, available_actions: List[str]):
        policy_state = self.policy[state_key]
        
        for action in available_actions:
            if action not in policy_state:
                policy_state[action] = 1.0
        
        total = sum(policy_state.values())
        if total > 0:
            for action in policy_state:
                policy_state[action] /= total
    
    def act(self, observation: Dict[str, Any]) -> Optional[Action]:
        state_key = self._get_state_key(observation)
        available_actions = observation.get("available_tools", [])
        
        if not available_actions:
            return None
        
        self._normalize_policy(state_key, available_actions)
        
        actions = list(self.policy[state_key].keys())
        probabilities = list(self.policy[state_key].values())
        
        selected_tool = random.choices(actions, weights=probabilities, k=1)[0]
        probability = self.policy[state_key].get(selected_tool, 0.0)
        
        return Action(
            tool=selected_tool,
            parameters={},
            confidence=probability,
            reasoning=f"基于策略概率选择 (p={probability:.2f})"
        )
    
    def observe(
        self,
        observation: Dict[str, Any],
        action: Action,
        reward: float,
        next_observation: Dict[str, Any],
        done: bool
    ):
        state_key = self._get_state_key(observation)
        action_tool = action.tool
        
        self.episode_buffer.append((state_key, action_tool, reward))
        self.total_steps += 1
        
        if done:
            self._update_policy()
            self.episode_rewards.append(sum(r for _, _, r in self.episode_buffer))
            self.episode_buffer = []
    
    def _update_policy(self):
        returns = []
        G = 0
        
        for _, _, reward in reversed(self.episode_buffer):
            G = reward + self.discount_factor * G
            returns.insert(0, G)
        
        for i, (state_key, action, _) in enumerate(self.episode_buffer):
            G = returns[i]
            
            self.policy[state_key][action] += self.learning_rate * G * (1 - self.policy[state_key][action])
            
            for other_action in self.policy[state_key]:
                if other_action != action:
                    self.policy[state_key][other_action] -= self.learning_rate * G * self.policy[state_key][other_action]
    
    def get_policy_stats(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "policy_states": len(self.policy),
            "total_steps": self.total_steps,
            "episodes": len(self.episode_rewards)
        }


class HeuristicAgent(Agent):
    """
    启发式智能体 - 基于规则的智能体
    
    核心特点：
    1. 使用预定义的规则选择动作
    2. 适合作为基线比较
    3. 可以结合符号化反思进行改进
    """
    
    def __init__(self, name: str = "HeuristicAgent"):
        super().__init__(name)
        self.rules: List[Dict] = []
    
    def add_rule(self, condition: str, action: str, priority: int = 0):
        self.rules.append({
            "condition": condition,
            "action": action,
            "priority": priority
        })
        self.rules.sort(key=lambda x: x["priority"], reverse=True)
    
    def act(self, observation: Dict[str, Any]) -> Optional[Action]:
        available_actions = observation.get("available_tools", [])
        
        if not available_actions:
            return None
        
        for rule in self.rules:
            try:
                namespace = {
                    "observation": observation,
                    "state": observation.get("state_variables", {}),
                    "current_state": observation.get("current_state"),
                    "history": observation.get("history", [])
                }
                
                if eval(rule["condition"], namespace):
                    action_name = rule["action"]
                    if action_name in available_actions:
                        return Action(
                            tool=action_name,
                            parameters={},
                            confidence=0.9,
                            reasoning=f"触发规则: {rule['condition']}"
                        )
            except Exception:
                continue
        
        if available_actions:
            return Action(
                tool=available_actions[0],
                parameters={},
                confidence=0.3,
                reasoning="无规则匹配，选择默认动作"
            )
        
        return None
    
    def observe(
        self,
        observation: Dict[str, Any],
        action: Action,
        reward: float,
        next_observation: Dict[str, Any],
        done: bool
    ):
        self.total_steps += 1
        if done:
            self.episode_rewards.append(reward)
