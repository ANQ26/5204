"""
智能体模块 - 定义与环境交互的智能体

改进版本：
- 统一的 save/load 接口
- 更严格的类型注解
- 训练/评估模式分离
- 更好的异常处理
- 支持状态持久化
"""

import random
import math
import json
import os
from typing import Dict, List, Optional, Any, Tuple, Union, Callable, Type
from dataclasses import dataclass, field, asdict, fields
from collections import defaultdict
from abc import ABC, abstractmethod
from pathlib import Path
import logging


logger = logging.getLogger(__name__)


@dataclass
class Action:
    tool: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    reasoning: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool": self.tool,
            "parameters": self.parameters.copy(),
            "confidence": self.confidence,
            "reasoning": self.reasoning
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Action':
        return cls(
            tool=data.get("tool", ""),
            parameters=data.get("parameters", {}),
            confidence=data.get("confidence", 1.0),
            reasoning=data.get("reasoning", "")
        )


class Agent(ABC):
    """
    智能体基类
    
    核心功能：
    1. 观察环境状态
    2. 选择动作（工具调用）
    3. 学习从经验中改进
    4. 保存/加载状态
    """
    
    def __init__(self, name: str = "Agent"):
        self.name: str = name
        self.episode_rewards: List[float] = []
        self.total_steps: int = 0
        self.training: bool = True
        self._logger: Optional[logging.Logger] = None
    
    @abstractmethod
    def act(self, observation: Dict[str, Any]) -> Optional[Action]:
        """
        根据观察选择动作
        
        Args:
            observation: 环境观察
            
        Returns:
            动作对象，如果无法选择则返回 None
        """
        pass
    
    @abstractmethod
    def observe(
        self,
        observation: Dict[str, Any],
        action: Action,
        reward: float,
        next_observation: Dict[str, Any],
        done: bool
    ) -> None:
        """
        观察经验并学习
        
        Args:
            observation: 当前观察
            action: 执行的动作
            reward: 获得的奖励
            next_observation: 下一个观察
            done: 是否结束
        """
        pass
    
    def reset(self) -> None:
        """重置智能体状态"""
        self.episode_rewards = []
    
    def train_mode(self) -> None:
        """设置为训练模式"""
        self.training = True
    
    def eval_mode(self) -> None:
        """设置为评估模式"""
        self.training = False
    
    def get_stats(self) -> Dict[str, Any]:
        """获取智能体统计信息"""
        return {
            "name": self.name,
            "total_steps": self.total_steps,
            "episode_count": len(self.episode_rewards),
            "average_reward": sum(self.episode_rewards) / len(self.episode_rewards) if self.episode_rewards else 0.0,
            "max_reward": max(self.episode_rewards) if self.episode_rewards else 0.0,
            "min_reward": min(self.episode_rewards) if self.episode_rewards else 0.0,
            "training": self.training
        }
    
    def save(self, filepath: str) -> bool:
        """
        保存智能体状态
        
        Args:
            filepath: 保存路径
            
        Returns:
            是否保存成功
        """
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            data = {
                "agent_type": type(self).__name__,
                "name": self.name,
                "total_steps": self.total_steps,
                "episode_rewards": self.episode_rewards,
                "training": self.training,
                "timestamp": Path(filepath).stat().st_mtime if os.path.exists(filepath) else 0.0
            }
            
            extra_data = self._get_save_data()
            data.update(extra_data)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"智能体已保存: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"保存智能体失败 {filepath}: {e}")
            return False
    
    def load(self, filepath: str) -> bool:
        """
        加载智能体状态
        
        Args:
            filepath: 加载路径
            
        Returns:
            是否加载成功
        """
        try:
            if not os.path.exists(filepath):
                logger.warning(f"智能体文件不存在: {filepath}")
                return False
            
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.name = data.get("name", self.name)
            self.total_steps = data.get("total_steps", 0)
            self.episode_rewards = data.get("episode_rewards", [])
            self.training = data.get("training", True)
            
            self._load_from_data(data)
            
            logger.info(f"智能体已加载: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"加载智能体失败 {filepath}: {e}")
            return False
    
    def _get_save_data(self) -> Dict[str, Any]:
        """获取子类特定的保存数据"""
        return {}
    
    def _load_from_data(self, data: Dict[str, Any]) -> None:
        """从数据加载子类特定状态"""
        pass


class RandomAgent(Agent):
    """
    随机智能体 - 随机选择可用工具
    
    作为基线比较使用
    """
    
    def __init__(self, name: str = "RandomAgent"):
        super().__init__(name)
        self._action_count: int = 0
    
    def act(self, observation: Dict[str, Any]) -> Optional[Action]:
        available_tools = observation.get("available_tools", [])
        
        if not available_tools:
            logger.debug(f"随机智能体 {self.name}: 没有可用工具")
            return None
        
        selected_tool = random.choice(available_tools)
        self._action_count += 1
        
        return Action(
            tool=selected_tool,
            parameters={},
            confidence=0.5,
            reasoning=f"随机选择工具 (第 {self._action_count} 次选择)"
        )
    
    def observe(
        self,
        observation: Dict[str, Any],
        action: Action,
        reward: float,
        next_observation: Dict[str, Any],
        done: bool
    ) -> None:
        self.total_steps += 1
        if done:
            self.episode_rewards.append(reward)
    
    def _get_save_data(self) -> Dict[str, Any]:
        return {
            "_action_count": self._action_count
        }
    
    def _load_from_data(self, data: Dict[str, Any]) -> None:
        self._action_count = data.get("_action_count", 0)


class QLearningAgent(Agent):
    """
    Q学习智能体 - 基于表格的Q学习算法
    
    核心特点：
    1. 使用Q表存储状态-动作价值
    2. ε-greedy策略进行探索
    3. 从经验中学习最优策略
    4. 支持优先经验回放权重
    """
    
    def __init__(
        self,
        name: str = "QLearningAgent",
        learning_rate: float = 0.1,
        discount_factor: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
        initial_q_value: float = 0.0
    ):
        super().__init__(name)
        self.learning_rate: float = learning_rate
        self.discount_factor: float = discount_factor
        self.epsilon: float = epsilon
        self.epsilon_decay: float = epsilon_decay
        self.epsilon_min: float = epsilon_min
        self.initial_q_value: float = initial_q_value
        
        self.q_table: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.action_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.state_visits: Dict[str, int] = defaultdict(int)
        
        self._episode_count: int = 0
    
    def _get_state_key(self, observation: Dict[str, Any]) -> str:
        """生成状态键"""
        try:
            state_vars = observation.get("state_variables", {})
            current_state = observation.get("current_state", "unknown")
            
            key_parts: List[str] = [f"state={current_state}"]
            
            for var_name, value in sorted(state_vars.items()):
                if isinstance(value, (int, float, bool, str)):
                    key_parts.append(f"{var_name}={value}")
                elif isinstance(value, (list, tuple)) and len(value) < 10:
                    key_parts.append(f"{var_name}={str(value)}")
            
            return "|".join(key_parts)
        except Exception as e:
            logger.warning(f"生成状态键失败: {e}")
            return f"state_hash_{hash(str(observation)) % 1000000}"
    
    def _get_available_actions(self, observation: Dict[str, Any]) -> List[str]:
        """获取可用动作"""
        return observation.get("available_tools", [])
    
    def _get_q_value(self, state_key: str, action: str) -> float:
        """获取Q值"""
        return self.q_table[state_key].get(action, self.initial_q_value)
    
    def _update_q_value(
        self,
        state_key: str,
        action: str,
        reward: float,
        next_state_key: str,
        done: bool,
        importance_weight: float = 1.0
    ) -> float:
        """
        更新Q值
        
        Args:
            state_key: 当前状态键
            action: 动作
            reward: 奖励
            next_state_key: 下一个状态键
            done: 是否结束
            importance_weight: 重要性采样权重（用于优先经验回放）
            
        Returns:
            TD误差
        """
        current_q = self._get_q_value(state_key, action)
        
        if next_state_key in self.q_table and self.q_table[next_state_key]:
            max_next_q = max(self.q_table[next_state_key].values())
        else:
            max_next_q = self.initial_q_value
        
        target = reward + self.discount_factor * max_next_q * (not done)
        td_error = target - current_q
        
        new_q = current_q + self.learning_rate * importance_weight * td_error
        self.q_table[state_key][action] = new_q
        
        return td_error
    
    def act(self, observation: Dict[str, Any]) -> Optional[Action]:
        state_key = self._get_state_key(observation)
        available_actions = self._get_available_actions(observation)
        
        if not available_actions:
            logger.debug(f"Q学习智能体 {self.name}: 没有可用动作")
            return None
        
        self.state_visits[state_key] += 1
        
        if self.training and random.random() < self.epsilon:
            selected_tool = random.choice(available_actions)
            reasoning = f"探索(ε={self.epsilon:.3f})：随机选择工具"
            confidence = 0.3
        else:
            q_values = {a: self._get_q_value(state_key, a) for a in available_actions}
            
            if q_values:
                max_q = max(q_values.values())
                best_actions = [a for a, q in q_values.items() if abs(q - max_q) < 1e-8]
                
                if best_actions:
                    selected_tool = random.choice(best_actions)
                    reasoning = f"利用：选择Q值最高的工具 (Q={max_q:.4f})"
                    confidence = 0.7 + min(0.2, max_q / 10.0)
                else:
                    selected_tool = random.choice(available_actions)
                    reasoning = "无Q值信息，随机选择"
                    confidence = 0.3
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
    ) -> None:
        if not self.training:
            self.total_steps += 1
            if done:
                self.episode_rewards.append(reward)
            return
        
        state_key = self._get_state_key(observation)
        next_state_key = self._get_state_key(next_observation)
        action_tool = action.tool
        
        self._update_q_value(state_key, action_tool, reward, next_state_key, done)
        
        self.action_counts[state_key][action_tool] += 1
        self.total_steps += 1
        
        if done:
            self.episode_rewards.append(reward)
            self._episode_count += 1
            self._decay_epsilon()
    
    def _decay_epsilon(self) -> None:
        """衰减探索率"""
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def get_q_table_size(self) -> int:
        """获取Q表大小"""
        return sum(len(actions) for actions in self.q_table.values())
    
    def get_state_count(self) -> int:
        """获取访问过的状态数"""
        return len(self.q_table)
    
    def get_most_visited_states(self, top_n: int = 10) -> List[Tuple[str, int]]:
        """获取访问最多的状态"""
        sorted_items = sorted(self.state_visits.items(), key=lambda x: x[1], reverse=True)
        return sorted_items[:top_n]
    
    def _get_save_data(self) -> Dict[str, Any]:
        return {
            "learning_rate": self.learning_rate,
            "discount_factor": self.discount_factor,
            "epsilon": self.epsilon,
            "epsilon_decay": self.epsilon_decay,
            "epsilon_min": self.epsilon_min,
            "initial_q_value": self.initial_q_value,
            "q_table": dict(self.q_table),
            "action_counts": {k: dict(v) for k, v in self.action_counts.items()},
            "state_visits": dict(self.state_visits),
            "_episode_count": self._episode_count
        }
    
    def _load_from_data(self, data: Dict[str, Any]) -> None:
        self.learning_rate = data.get("learning_rate", self.learning_rate)
        self.discount_factor = data.get("discount_factor", self.discount_factor)
        self.epsilon = data.get("epsilon", self.epsilon)
        self.epsilon_decay = data.get("epsilon_decay", self.epsilon_decay)
        self.epsilon_min = data.get("epsilon_min", self.epsilon_min)
        self.initial_q_value = data.get("initial_q_value", self.initial_q_value)
        
        self.q_table = defaultdict(dict, data.get("q_table", {}))
        self.action_counts = defaultdict(
            lambda: defaultdict(int),
            {k: defaultdict(int, v) for k, v in data.get("action_counts", {}).items()}
        )
        self.state_visits = defaultdict(int, data.get("state_visits", {}))
        self._episode_count = data.get("_episode_count", 0)


class REINFORCEAgent(Agent):
    """
    REINFORCE智能体 - 策略梯度方法
    
    核心特点：
    1. 直接参数化策略函数
    2. 使用蒙特卡洛方法估计梯度
    3. 适合离散动作空间
    4. 支持基线减少方差
    """
    
    def __init__(
        self,
        name: str = "REINFORCEAgent",
        learning_rate: float = 0.01,
        discount_factor: float = 0.99,
        use_baseline: bool = True
    ):
        super().__init__(name)
        self.learning_rate: float = learning_rate
        self.discount_factor: float = discount_factor
        self.use_baseline: bool = use_baseline
        
        self.policy: Dict[str, Dict[str, float]] = defaultdict(
            lambda: defaultdict(lambda: 1.0)
        )
        self.episode_buffer: List[Tuple[str, str, float, Dict[str, Any], Dict[str, Any]]] = []
        self.baseline_values: Dict[str, float] = defaultdict(float)
        
        self._total_updates: int = 0
    
    def _get_state_key(self, observation: Dict[str, Any]) -> str:
        """生成状态键"""
        try:
            state_vars = observation.get("state_variables", {})
            current_state = observation.get("current_state", "unknown")
            
            key_parts: List[str] = [f"state={current_state}"]
            
            for var_name, value in sorted(state_vars.items()):
                if isinstance(value, (int, float, bool, str)):
                    key_parts.append(f"{var_name}={value}")
            
            return "|".join(key_parts)
        except Exception as e:
            logger.warning(f"REINFORCE 生成状态键失败: {e}")
            return f"reinforce_hash_{hash(str(observation)) % 1000000}"
    
    def _normalize_policy(self, state_key: str, available_actions: List[str]) -> None:
        """归一化策略"""
        try:
            policy_state = self.policy[state_key]
            
            for action in available_actions:
                if action not in policy_state:
                    policy_state[action] = 1.0
            
            total = sum(policy_state.values())
            if total > 0 and total != 1.0:
                for action in policy_state:
                    policy_state[action] /= total
        except Exception as e:
            logger.error(f"归一化策略失败: {e}")
    
    def act(self, observation: Dict[str, Any]) -> Optional[Action]:
        state_key = self._get_state_key(observation)
        available_actions = observation.get("available_tools", [])
        
        if not available_actions:
            logger.debug(f"REINFORCE智能体 {self.name}: 没有可用动作")
            return None
        
        if self.training:
            self._normalize_policy(state_key, available_actions)
            
            valid_actions = [a for a in available_actions if a in self.policy[state_key]]
            if not valid_actions:
                valid_actions = available_actions
            
            probabilities = [self.policy[state_key].get(a, 1.0 / len(valid_actions)) for a in valid_actions]
            
            total_prob = sum(probabilities)
            if total_prob > 0:
                probabilities = [p / total_prob for p in probabilities]
            else:
                probabilities = [1.0 / len(valid_actions)] * len(valid_actions)
            
            try:
                selected_tool = random.choices(valid_actions, weights=probabilities, k=1)[0]
                probability = self.policy[state_key].get(selected_tool, 0.0)
            except Exception as e:
                logger.warning(f"策略采样失败: {e}")
                selected_tool = random.choice(available_actions)
                probability = 0.5
            
            reasoning = f"基于策略概率选择 (p={probability:.4f})"
            confidence = probability
        else:
            self._normalize_policy(state_key, available_actions)
            policy_state = self.policy[state_key]
            
            best_action = max(
                available_actions,
                key=lambda a: policy_state.get(a, 0.0),
                default=None
            )
            
            if best_action is None:
                selected_tool = random.choice(available_actions)
                confidence = 0.5
                reasoning = "评估模式：无策略信息，随机选择"
            else:
                selected_tool = best_action
                confidence = policy_state.get(best_action, 0.5)
                reasoning = f"评估模式：选择概率最高的动作"
        
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
    ) -> None:
        if not self.training:
            self.total_steps += 1
            if done:
                self.episode_rewards.append(reward)
            return
        
        state_key = self._get_state_key(observation)
        action_tool = action.tool
        
        self.episode_buffer.append((state_key, action_tool, reward, observation, next_observation))
        self.total_steps += 1
        
        if done:
            total_reward = sum(r for _, _, r, _, _ in self.episode_buffer)
            self.episode_rewards.append(total_reward)
            self._update_policy()
            self.episode_buffer = []
    
    def _update_policy(self) -> None:
        """更新策略"""
        if not self.episode_buffer:
            return
        
        try:
            returns: List[float] = []
            G = 0.0
            
            for _, _, reward, _, _ in reversed(self.episode_buffer):
                G = reward + self.discount_factor * G
                returns.insert(0, G)
            
            for i, (state_key, action, _, _, _) in enumerate(self.episode_buffer):
                G = returns[i]
                
                if self.use_baseline:
                    baseline = self.baseline_values.get(state_key, 0.0)
                    delta = G - baseline
                    self.baseline_values[state_key] = (
                        0.9 * baseline + 0.1 * G
                    )
                else:
                    delta = G
                
                current_prob = self.policy[state_key].get(action, 0.0)
                
                self.policy[state_key][action] += self.learning_rate * delta * (1 - current_prob)
                
                for other_action in list(self.policy[state_key].keys()):
                    if other_action != action:
                        other_prob = self.policy[state_key][other_action]
                        self.policy[state_key][other_action] -= self.learning_rate * delta * other_prob
                
                for act in self.policy[state_key]:
                    self.policy[state_key][act] = max(1e-8, min(1.0 - 1e-8, self.policy[state_key][act]))
            
            self._total_updates += 1
            
        except Exception as e:
            logger.error(f"更新策略失败: {e}")
    
    def get_policy_stats(self) -> Dict[str, Any]:
        """获取策略统计"""
        return {
            "name": self.name,
            "policy_states": len(self.policy),
            "total_steps": self.total_steps,
            "episodes": len(self.episode_rewards),
            "total_updates": self._total_updates,
            "use_baseline": self.use_baseline
        }
    
    def _get_save_data(self) -> Dict[str, Any]:
        return {
            "learning_rate": self.learning_rate,
            "discount_factor": self.discount_factor,
            "use_baseline": self.use_baseline,
            "policy": {k: dict(v) for k, v in self.policy.items()},
            "baseline_values": dict(self.baseline_values),
            "_total_updates": self._total_updates
        }
    
    def _load_from_data(self, data: Dict[str, Any]) -> None:
        self.learning_rate = data.get("learning_rate", self.learning_rate)
        self.discount_factor = data.get("discount_factor", self.discount_factor)
        self.use_baseline = data.get("use_baseline", self.use_baseline)
        
        policy_data = data.get("policy", {})
        self.policy = defaultdict(
            lambda: defaultdict(lambda: 1.0),
            {k: defaultdict(lambda: 1.0, v) for k, v in policy_data.items()}
        )
        
        self.baseline_values = defaultdict(float, data.get("baseline_values", {}))
        self._total_updates = data.get("_total_updates", 0)


class HeuristicAgent(Agent):
    """
    启发式智能体 - 基于规则的智能体
    
    核心特点：
    1. 使用预定义的规则选择动作
    2. 适合作为基线比较
    3. 可以结合符号化反思进行改进
    4. 支持动态规则更新
    """
    
    def __init__(self, name: str = "HeuristicAgent"):
        super().__init__(name)
        self.rules: List[Dict[str, Any]] = []
        self._rule_counter: int = 0
    
    def add_rule(
        self,
        condition: Union[str, Callable[[Dict[str, Any]], bool]],
        action: str,
        priority: int = 0,
        parameters: Optional[Dict[str, Any]] = None,
        description: str = ""
    ) -> int:
        """
        添加规则
        
        Args:
            condition: 条件字符串（可eval）或可调用对象
            action: 动作名
            priority: 优先级（越大越优先）
            parameters: 动作参数
            description: 规则描述
            
        Returns:
            规则ID
        """
        rule_id = self._rule_counter
        self._rule_counter += 1
        
        rule = {
            "id": rule_id,
            "condition": condition,
            "action": action,
            "priority": priority,
            "parameters": parameters or {},
            "description": description,
            "trigger_count": 0
        }
        
        self.rules.append(rule)
        self.rules.sort(key=lambda x: x["priority"], reverse=True)
        
        logger.debug(f"添加规则 ID={rule_id}: {description or action}")
        return rule_id
    
    def remove_rule(self, rule_id: int) -> bool:
        """移除规则"""
        for i, rule in enumerate(self.rules):
            if rule["id"] == rule_id:
                self.rules.pop(i)
                logger.debug(f"移除规则 ID={rule_id}")
                return True
        return False
    
    def get_rules(self) -> List[Dict[str, Any]]:
        """获取所有规则"""
        return [r.copy() for r in self.rules]
    
    def _evaluate_condition(
        self,
        condition: Union[str, Callable],
        observation: Dict[str, Any]
    ) -> bool:
        """评估条件"""
        try:
            if callable(condition):
                return bool(condition(observation))
            
            namespace = {
                "observation": observation,
                "state": observation.get("state_variables", {}),
                "current_state": observation.get("current_state"),
                "history": observation.get("history", []),
                "abs": abs,
                "len": len,
                "min": min,
                "max": max,
                "sum": sum
            }
            
            result = eval(str(condition), namespace)
            return bool(result)
            
        except Exception as e:
            logger.debug(f"评估条件失败: {condition}, 错误: {e}")
            return False
    
    def act(self, observation: Dict[str, Any]) -> Optional[Action]:
        available_actions = observation.get("available_tools", [])
        
        if not available_actions:
            logger.debug(f"启发式智能体 {self.name}: 没有可用动作")
            return None
        
        for rule in self.rules:
            try:
                if self._evaluate_condition(rule["condition"], observation):
                    action_name = rule["action"]
                    
                    if action_name in available_actions:
                        rule["trigger_count"] += 1
                        
                        params = rule.get("parameters", {})
                        desc = rule.get("description", f"触发规则 ID={rule['id']}")
                        
                        return Action(
                            tool=action_name,
                            parameters=params.copy(),
                            confidence=0.9,
                            reasoning=desc
                        )
            except Exception as e:
                logger.warning(f"执行规则失败: {e}")
                continue
        
        if available_actions:
            default_action = available_actions[0]
            return Action(
                tool=default_action,
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
    ) -> None:
        self.total_steps += 1
        if done:
            self.episode_rewards.append(reward)
    
    def get_rule_stats(self) -> Dict[str, Any]:
        """获取规则统计"""
        trigger_counts = {r["id"]: r["trigger_count"] for r in self.rules}
        
        return {
            "name": self.name,
            "total_rules": len(self.rules),
            "trigger_counts": trigger_counts,
            "most_triggered": max(self.rules, key=lambda r: r["trigger_count"])["id"] if self.rules else None,
            "total_steps": self.total_steps
        }
    
    def _get_save_data(self) -> Dict[str, Any]:
        serializable_rules = []
        for rule in self.rules:
            serializable_rule = {
                "id": rule["id"],
                "action": rule["action"],
                "priority": rule["priority"],
                "parameters": rule.get("parameters", {}),
                "description": rule.get("description", ""),
                "trigger_count": rule.get("trigger_count", 0)
            }
            
            if callable(rule["condition"]):
                serializable_rule["condition_type"] = "callable"
                serializable_rule["condition"] = str(rule["condition"])
            else:
                serializable_rule["condition_type"] = "string"
                serializable_rule["condition"] = str(rule["condition"])
            
            serializable_rules.append(serializable_rule)
        
        return {
            "rules": serializable_rules,
            "_rule_counter": self._rule_counter
        }
    
    def _load_from_data(self, data: Dict[str, Any]) -> None:
        self.rules = []
        
        for rule_data in data.get("rules", []):
            if rule_data.get("condition_type") == "string":
                condition = rule_data.get("condition", "")
            else:
                condition = rule_data.get("condition", "")
            
            rule = {
                "id": rule_data.get("id", len(self.rules)),
                "condition": condition,
                "action": rule_data.get("action", ""),
                "priority": rule_data.get("priority", 0),
                "parameters": rule_data.get("parameters", {}),
                "description": rule_data.get("description", ""),
                "trigger_count": rule_data.get("trigger_count", 0)
            }
            
            self.rules.append(rule)
        
        self.rules.sort(key=lambda x: x["priority"], reverse=True)
        self._rule_counter = data.get("_rule_counter", len(self.rules))


def load_agent_from_file(filepath: str) -> Optional[Agent]:
    """
    从文件加载智能体（自动识别类型）
    
    Args:
        filepath: 智能体文件路径
        
    Returns:
        智能体实例，失败返回 None
    """
    try:
        if not os.path.exists(filepath):
            logger.warning(f"智能体文件不存在: {filepath}")
            return None
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        agent_type = data.get("agent_type", "")
        
        agent_classes: Dict[str, Type[Agent]] = {
            "QLearningAgent": QLearningAgent,
            "REINFORCEAgent": REINFORCEAgent,
            "HeuristicAgent": HeuristicAgent,
            "RandomAgent": RandomAgent,
            "Agent": Agent
        }
        
        agent_class = agent_classes.get(agent_type)
        
        if agent_class is None:
            logger.warning(f"未知的智能体类型: {agent_type}")
            return None
        
        agent = agent_class(name=data.get("name", "LoadedAgent"))
        agent.load(filepath)
        
        logger.info(f"成功加载智能体: {agent_type} from {filepath}")
        return agent
        
    except Exception as e:
        logger.error(f"加载智能体失败: {e}")
        return None
