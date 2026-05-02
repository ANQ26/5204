"""
经验回放缓冲区 - 用于存储和采样智能体的经验
"""

import random
import heapq
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from collections import deque
import json


@dataclass
class Experience:
    state: Dict[str, Any]
    action: Dict[str, Any]
    reward: float
    next_state: Dict[str, Any]
    done: bool
    priority: float = 1.0
    timestamp: float = field(default_factory=lambda: __import__('time').time())


class ExperienceBuffer:
    """
    经验回放缓冲区
    
    核心功能：
    1. 存储智能体的交互经验
    2. 支持批量采样用于训练
    3. 管理缓冲区大小和旧经验的淘汰
    """
    
    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.buffer: deque = deque(maxlen=capacity)
    
    def push(self, experience: Experience):
        self.buffer.append(experience)
    
    def push_transition(
        self,
        state: Dict[str, Any],
        action: Dict[str, Any],
        reward: float,
        next_state: Dict[str, Any],
        done: bool
    ):
        experience = Experience(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done
        )
        self.push(experience)
    
    def sample(self, batch_size: int) -> List[Experience]:
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)
        
        return random.sample(list(self.buffer), batch_size)
    
    def sample_batch(self, batch_size: int) -> Tuple[
        List[Dict], List[Dict], List[float], List[Dict], List[bool]
    ]:
        samples = self.sample(batch_size)
        
        states = [s.state for s in samples]
        actions = [s.action for s in samples]
        rewards = [s.reward for s in samples]
        next_states = [s.next_state for s in samples]
        dones = [s.done for s in samples]
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self) -> int:
        return len(self.buffer)
    
    def clear(self):
        self.buffer.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        if not self.buffer:
            return {
                "size": 0,
                "capacity": self.capacity,
                "utilization": 0.0
            }
        
        rewards = [e.reward for e in self.buffer]
        
        return {
            "size": len(self.buffer),
            "capacity": self.capacity,
            "utilization": len(self.buffer) / self.capacity,
            "avg_reward": sum(rewards) / len(rewards),
            "min_reward": min(rewards),
            "max_reward": max(rewards)
        }
    
    def save(self, filepath: str):
        data = {
            "capacity": self.capacity,
            "experiences": [
                {
                    "state": e.state,
                    "action": e.action,
                    "reward": e.reward,
                    "next_state": e.next_state,
                    "done": e.done,
                    "priority": e.priority,
                    "timestamp": e.timestamp
                }
                for e in self.buffer
            ]
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def load(self, filepath: str):
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.capacity = data.get("capacity", self.capacity)
        self.buffer.clear()
        
        for exp_data in data.get("experiences", []):
            experience = Experience(
                state=exp_data["state"],
                action=exp_data["action"],
                reward=exp_data["reward"],
                next_state=exp_data["next_state"],
                done=exp_data["done"],
                priority=exp_data.get("priority", 1.0),
                timestamp=exp_data.get("timestamp", 0.0)
            )
            self.push(experience)


class PrioritizedExperienceBuffer(ExperienceBuffer):
    """
    优先经验回放缓冲区
    
    核心特点：
    1. 根据TD误差优先级采样经验
    2. 更频繁地采样"重要"经验
    3. 支持重要性采样权重
    """
    
    def __init__(self, capacity: int = 10000, alpha: float = 0.6, beta: float = 0.4):
        super().__init__(capacity)
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = 0.001
        self.max_priority = 1.0
    
    def push(self, experience: Experience):
        experience.priority = self.max_priority
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> Tuple[List[Experience], List[float], List[int]]:
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)
        
        priorities = [e.priority for e in self.buffer]
        total_priority = sum(p ** self.alpha for p in priorities)
        
        probabilities = [(p ** self.alpha) / total_priority for p in priorities]
        
        indices = random.choices(range(len(self.buffer)), weights=probabilities, k=batch_size)
        
        samples = [self.buffer[i] for i in indices]
        
        weights = []
        N = len(self.buffer)
        for i in indices:
            prob = probabilities[i]
            weight = (N * prob) ** (-self.beta)
            weights.append(weight)
        
        max_weight = max(weights) if weights else 1.0
        weights = [w / max_weight for w in weights]
        
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        return samples, weights, indices
    
    def update_priorities(self, indices: List[int], td_errors: List[float]):
        for idx, td_error in zip(indices, td_errors):
            if 0 <= idx < len(self.buffer):
                priority = abs(td_error) + 1e-6
                self.buffer[idx].priority = priority
                self.max_priority = max(self.max_priority, priority)
    
    def sample_batch(self, batch_size: int) -> Tuple[
        List[Dict], List[Dict], List[float], List[Dict], List[bool], List[float], List[int]
    ]:
        samples, weights, indices = self.sample(batch_size)
        
        states = [s.state for s in samples]
        actions = [s.action for s in samples]
        rewards = [s.reward for s in samples]
        next_states = [s.next_state for s in samples]
        dones = [s.done for s in samples]
        
        return states, actions, rewards, next_states, dones, weights, indices


class EpisodeBuffer:
    """
    回合缓冲区 - 存储完整的回合经验
    
    核心特点：
    1. 按回合组织经验
    2. 支持蒙特卡洛方法
    3. 可用于策略梯度方法
    """
    
    def __init__(self):
        self.episodes: List[List[Experience]] = []
        self.current_episode: List[Experience] = []
    
    def start_episode(self):
        self.current_episode = []
    
    def add_step(
        self,
        state: Dict[str, Any],
        action: Dict[str, Any],
        reward: float,
        next_state: Dict[str, Any],
        done: bool
    ):
        experience = Experience(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done
        )
        self.current_episode.append(experience)
        
        if done:
            self.end_episode()
    
    def end_episode(self):
        if self.current_episode:
            self.episodes.append(self.current_episode)
            self.current_episode = []
    
    def get_latest_episode(self) -> Optional[List[Experience]]:
        if self.episodes:
            return self.episodes[-1]
        return None
    
    def get_all_episodes(self) -> List[List[Experience]]:
        return self.episodes
    
    def compute_returns(self, gamma: float = 0.99) -> List[List[float]]:
        all_returns = []
        
        for episode in self.episodes:
            returns = []
            G = 0
            
            for experience in reversed(episode):
                G = experience.reward + gamma * G
                returns.insert(0, G)
            
            all_returns.append(returns)
        
        return all_returns
    
    def get_stats(self) -> Dict[str, Any]:
        if not self.episodes:
            return {
                "episode_count": 0,
                "total_steps": 0
            }
        
        episode_lengths = [len(e) for e in self.episodes]
        total_rewards = [sum(exp.reward for exp in e) for e in self.episodes]
        
        return {
            "episode_count": len(self.episodes),
            "total_steps": sum(episode_lengths),
            "avg_episode_length": sum(episode_lengths) / len(episode_lengths),
            "avg_episode_reward": sum(total_rewards) / len(total_rewards),
            "max_episode_reward": max(total_rewards),
            "min_episode_reward": min(total_rewards)
        }
    
    def clear(self):
        self.episodes.clear()
        self.current_episode = []
