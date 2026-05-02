"""
训练器模块 - 用于协调强化学习训练流程
"""

import time
import json
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from abc import ABC, abstractmethod

from src.rl.agent import Agent, Action
from src.rl.experience_buffer import ExperienceBuffer, EpisodeBuffer


@dataclass
class TrainingConfig:
    max_episodes: int = 1000
    max_steps_per_episode: int = 100
    learning_rate: float = 0.1
    discount_factor: float = 0.99
    epsilon: float = 1.0
    epsilon_decay: float = 0.995
    epsilon_min: float = 0.01
    batch_size: int = 32
    replay_buffer_capacity: int = 10000
    target_update_freq: int = 100
    eval_freq: int = 100
    eval_episodes: int = 10
    save_freq: int = 500
    verbose: bool = True
    log_dir: str = "./logs"


@dataclass
class TrainingStats:
    episode_rewards: List[float] = field(default_factory=list)
    episode_lengths: List[int] = field(default_factory=list)
    eval_rewards: List[float] = field(default_factory=list)
    losses: List[float] = field(default_factory=list)
    epsilon_values: List[float] = field(default_factory=list)
    timestamps: List[float] = field(default_factory=list)
    
    def get_summary(self) -> Dict[str, Any]:
        if not self.episode_rewards:
            return {"status": "no_data"}
        
        return {
            "total_episodes": len(self.episode_rewards),
            "avg_reward": sum(self.episode_rewards) / len(self.episode_rewards),
            "max_reward": max(self.episode_rewards),
            "min_reward": min(self.episode_rewards),
            "avg_episode_length": sum(self.episode_lengths) / len(self.episode_lengths),
            "final_epsilon": self.epsilon_values[-1] if self.epsilon_values else None
        }


class BaseTrainer(ABC):
    """
    训练器基类
    """
    
    def __init__(self, config: TrainingConfig = None):
        self.config = config or TrainingConfig()
        self.stats = TrainingStats()
        self.start_time = None
    
    @abstractmethod
    def train(
        self,
        agent: Agent,
        environment,
        callbacks: Dict[str, Callable] = None
    ) -> TrainingStats:
        pass
    
    @abstractmethod
    def evaluate(
        self,
        agent: Agent,
        environment,
        num_episodes: int = None
    ) -> Dict[str, Any]:
        pass
    
    def _log(self, message: str):
        if self.config.verbose:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}] {message}")


class RLTrainer(BaseTrainer):
    """
    强化学习训练器
    
    核心功能：
    1. 协调智能体与环境的交互
    2. 管理训练流程和超参数
    3. 收集训练统计数据
    4. 支持回调函数
    """
    
    def __init__(self, config: TrainingConfig = None):
        super().__init__(config)
        self.replay_buffer = ExperienceBuffer(self.config.replay_buffer_capacity)
        self.episode_buffer = EpisodeBuffer()
    
    def train(
        self,
        agent: Agent,
        environment,
        callbacks: Dict[str, Callable] = None
    ) -> TrainingStats:
        callbacks = callbacks or {}
        self.start_time = time.time()
        
        self._log(f"开始训练，配置: {self.config.max_episodes} 回合, "
                  f"每回合最大步数: {self.config.max_steps_per_episode}")
        
        epsilon = self.config.epsilon
        
        for episode in range(self.config.max_episodes):
            observation = environment.reset()
            
            episode_reward = 0.0
            episode_steps = 0
            
            self.episode_buffer.start_episode()
            
            for step in range(self.config.max_steps_per_episode):
                action = agent.act(observation)
                
                if action is None:
                    break
                
                tool_name = action.tool
                parameters = action.parameters
                
                result, reward, done, info = environment.execute_tool(tool_name, **parameters)
                
                next_observation = environment.get_observation()
                
                agent.observe(observation, action, reward, next_observation, done)
                
                self.replay_buffer.push_transition(
                    state=observation,
                    action={"tool": tool_name, "parameters": parameters},
                    reward=reward,
                    next_state=next_observation,
                    done=done
                )
                
                self.episode_buffer.add_step(
                    state=observation,
                    action={"tool": tool_name, "parameters": parameters},
                    reward=reward,
                    next_state=next_observation,
                    done=done
                )
                
                episode_reward += reward
                episode_steps += 1
                
                if callbacks.get("on_step"):
                    callbacks["on_step"]({
                        "episode": episode,
                        "step": step,
                        "action": action,
                        "reward": reward,
                        "observation": observation,
                        "next_observation": next_observation,
                        "done": done
                    })
                
                observation = next_observation
                
                if done:
                    break
            
            self.stats.episode_rewards.append(episode_reward)
            self.stats.episode_lengths.append(episode_steps)
            self.stats.epsilon_values.append(epsilon)
            self.stats.timestamps.append(time.time() - self.start_time)
            
            epsilon = max(self.config.epsilon_min, epsilon * self.config.epsilon_decay)
            
            if callbacks.get("on_episode"):
                callbacks["on_episode"]({
                    "episode": episode,
                    "reward": episode_reward,
                    "length": episode_steps,
                    "epsilon": epsilon
                })
            
            if (episode + 1) % self.config.eval_freq == 0:
                eval_stats = self.evaluate(agent, environment)
                self.stats.eval_rewards.append(eval_stats["avg_reward"])
                self._log(f"回合 {episode + 1}: 平均奖励={eval_stats['avg_reward']:.2f}, "
                          f"步数={episode_steps}, epsilon={epsilon:.4f}")
            
            if (episode + 1) % 10 == 0:
                recent_avg = sum(self.stats.episode_rewards[-10:]) / 10
                self._log(f"回合 {episode + 1}: 最近10回合平均奖励={recent_avg:.2f}")
        
        self._log(f"训练完成，总回合数: {self.config.max_episodes}")
        return self.stats
    
    def evaluate(
        self,
        agent: Agent,
        environment,
        num_episodes: int = None
    ) -> Dict[str, Any]:
        num_episodes = num_episodes or self.config.eval_episodes
        
        all_rewards = []
        all_lengths = []
        
        original_epsilon = getattr(agent, 'epsilon', None)
        if original_epsilon is not None:
            agent.epsilon = 0.0
        
        for episode in range(num_episodes):
            observation = environment.reset()
            
            episode_reward = 0.0
            episode_steps = 0
            
            for step in range(self.config.max_steps_per_episode):
                action = agent.act(observation)
                
                if action is None:
                    break
                
                tool_name = action.tool
                parameters = action.parameters
                
                result, reward, done, info = environment.execute_tool(tool_name, **parameters)
                
                next_observation = environment.get_observation()
                
                episode_reward += reward
                episode_steps += 1
                
                observation = next_observation
                
                if done:
                    break
            
            all_rewards.append(episode_reward)
            all_lengths.append(episode_steps)
        
        if original_epsilon is not None:
            agent.epsilon = original_epsilon
        
        return {
            "avg_reward": sum(all_rewards) / len(all_rewards) if all_rewards else 0.0,
            "max_reward": max(all_rewards) if all_rewards else 0.0,
            "min_reward": min(all_rewards) if all_rewards else 0.0,
            "avg_length": sum(all_lengths) / len(all_lengths) if all_lengths else 0,
            "episodes": num_episodes
        }
    
    def save_stats(self, filepath: str):
        data = {
            "config": self.config.__dict__,
            "stats": {
                "episode_rewards": self.stats.episode_rewards,
                "episode_lengths": self.stats.episode_lengths,
                "eval_rewards": self.stats.eval_rewards,
                "losses": self.stats.losses,
                "epsilon_values": self.stats.epsilon_values,
                "timestamps": self.stats.timestamps
            },
            "summary": self.stats.get_summary()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


class CurriculumTrainer(RLTrainer):
    """
    课程学习训练器
    
    核心特点：
    1. 从简单环境逐步过渡到复杂环境
    2. 自动调整任务难度
    3. 提高训练效率和泛化能力
    """
    
    def __init__(self, config: TrainingConfig = None):
        super().__init__(config)
        self.curriculum_levels = []
        self.current_level = 0
    
    def add_curriculum_level(
        self,
        environment_creator: Callable,
        min_reward_threshold: float,
        consecutive_successes: int = 3
    ):
        self.curriculum_levels.append({
            "creator": environment_creator,
            "min_reward": min_reward_threshold,
            "consecutive_successes": consecutive_successes,
            "success_count": 0
        })
    
    def train(
        self,
        agent: Agent,
        initial_environment=None,
        callbacks: Dict[str, Callable] = None
    ) -> TrainingStats:
        if not self.curriculum_levels:
            raise ValueError("No curriculum levels defined")
        
        callbacks = callbacks or {}
        self.start_time = time.time()
        
        self.current_level = 0
        consecutive_successes = 0
        
        self._log(f"开始课程学习训练，共 {len(self.curriculum_levels)} 个难度级别")
        
        while self.current_level < len(self.curriculum_levels):
            level = self.curriculum_levels[self.current_level]
            environment = level["creator"]()
            
            self._log(f"当前难度级别: {self.current_level + 1}")
            
            for episode in range(self.config.max_episodes):
                observation = environment.reset()
                
                episode_reward = 0.0
                episode_steps = 0
                
                for step in range(self.config.max_steps_per_episode):
                    action = agent.act(observation)
                    
                    if action is None:
                        break
                    
                    result, reward, done, info = environment.execute_tool(
                        action.tool, **action.parameters
                    )
                    
                    next_observation = environment.get_observation()
                    agent.observe(observation, action, reward, next_observation, done)
                    
                    episode_reward += reward
                    episode_steps += 1
                    
                    observation = next_observation
                    
                    if done:
                        break
                
                self.stats.episode_rewards.append(episode_reward)
                self.stats.episode_lengths.append(episode_steps)
                self.stats.timestamps.append(time.time() - self.start_time)
                
                if episode_reward >= level["min_reward"]:
                    consecutive_successes += 1
                else:
                    consecutive_successes = 0
                
                if consecutive_successes >= level["consecutive_successes"]:
                    self._log(f"级别 {self.current_level + 1} 完成，连续 {consecutive_successes} 次达到奖励阈值")
                    self.current_level += 1
                    consecutive_successes = 0
                    
                    if callbacks.get("on_level_up"):
                        callbacks["on_level_up"]({
                            "level": self.current_level,
                            "episode": episode
                        })
                    
                    break
                
                if (episode + 1) % 10 == 0:
                    self._log(f"级别 {self.current_level + 1}, 回合 {episode + 1}: "
                              f"奖励={episode_reward:.2f}, 连续成功={consecutive_successes}")
        
        self._log(f"课程学习训练完成，已通过所有 {len(self.curriculum_levels)} 个级别")
        return self.stats
