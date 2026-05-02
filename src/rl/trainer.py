"""
训练器模块 - 用于协调强化学习训练流程

改进版本：
- 完整的异常处理机制
- 经验回放与智能体更新自动集成
- 模型保存/加载（Checkpoint 机制）
- 日志文件系统
- 进度展示
- 改进的课程学习设计
- 评估与训练环境隔离
- epsilon 同步更新
"""

import time
import json
import os
import logging
import sys
import traceback
from typing import Dict, List, Optional, Any, Callable, Tuple, Union, Type
from dataclasses import dataclass, field, asdict
from datetime import datetime
from abc import ABC, abstractmethod
from pathlib import Path
from enum import Enum, auto

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

from src.rl.agent import Agent, Action, QLearningAgent, REINFORCEAgent
from src.rl.experience_buffer import (
    ExperienceBuffer, PrioritizedExperienceBuffer, 
    EpisodeBuffer, Experience
)


class TrainingStage(Enum):
    INITIAL = auto()
    TRAINING = auto()
    EVALUATING = auto()
    SAVING = auto()
    PAUSED = auto()
    ERROR = auto()
    COMPLETED = auto()


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
    use_prioritized_replay: bool = True
    prioritized_replay_alpha: float = 0.6
    prioritized_replay_beta: float = 0.4
    replay_update_freq: int = 4
    min_replay_size: int = 100
    target_update_freq: int = 100
    eval_freq: int = 100
    eval_episodes: int = 10
    save_freq: int = 500
    verbose: bool = True
    log_dir: str = "./logs"
    log_level: int = logging.INFO
    log_to_console: bool = True
    log_to_file: bool = True
    show_progress: bool = True
    max_time_seconds: Optional[float] = None
    early_stop_patience: int = 50
    early_stop_min_improvement: float = 0.01
    resume_from_checkpoint: Optional[str] = None
    separate_eval_env: bool = True
    reward_shaping: Optional[Callable[[float, Dict], float]] = None
    gradient_clip: Optional[float] = None


@dataclass
class TrainingCheckpoint:
    episode: int
    total_steps: int
    timestamp: float
    config: Dict[str, Any]
    stats: Dict[str, Any]
    agent_name: str
    agent_type: str
    best_reward: float


@dataclass
class TrainingStats:
    episode_rewards: List[float] = field(default_factory=list)
    episode_lengths: List[int] = field(default_factory=list)
    eval_rewards: List[float] = field(default_factory=list)
    losses: List[float] = field(default_factory=list)
    epsilon_values: List[float] = field(default_factory=list)
    timestamps: List[float] = field(default_factory=list)
    learning_rates: List[float] = field(default_factory=list)
    best_eval_reward: float = -float('inf')
    best_training_reward: float = -float('inf')
    episodes_without_improvement: int = 0
    
    def get_summary(self) -> Dict[str, Any]:
        if not self.episode_rewards:
            return {"status": "no_data"}
        
        recent_rewards = self.episode_rewards[-100:] if len(self.episode_rewards) > 100 else self.episode_rewards
        
        return {
            "total_episodes": len(self.episode_rewards),
            "total_steps": sum(self.episode_lengths),
            "avg_reward": sum(self.episode_rewards) / len(self.episode_rewards),
            "recent_avg_reward": sum(recent_rewards) / len(recent_rewards),
            "max_reward": max(self.episode_rewards),
            "min_reward": min(self.episode_rewards),
            "avg_episode_length": sum(self.episode_lengths) / len(self.episode_lengths),
            "best_eval_reward": self.best_eval_reward,
            "best_training_reward": self.best_training_reward,
            "final_epsilon": self.epsilon_values[-1] if self.epsilon_values else None,
            "episodes_without_improvement": self.episodes_without_improvement
        }
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "episode_rewards": self.episode_rewards,
            "episode_lengths": self.episode_lengths,
            "eval_rewards": self.eval_rewards,
            "losses": self.losses,
            "epsilon_values": self.epsilon_values,
            "timestamps": self.timestamps,
            "learning_rates": self.learning_rates,
            "best_eval_reward": self.best_eval_reward,
            "best_training_reward": self.best_training_reward,
            "episodes_without_improvement": self.episodes_without_improvement
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrainingStats':
        stats = cls()
        stats.episode_rewards = data.get("episode_rewards", [])
        stats.episode_lengths = data.get("episode_lengths", [])
        stats.eval_rewards = data.get("eval_rewards", [])
        stats.losses = data.get("losses", [])
        stats.epsilon_values = data.get("epsilon_values", [])
        stats.timestamps = data.get("timestamps", [])
        stats.learning_rates = data.get("learning_rates", [])
        stats.best_eval_reward = data.get("best_eval_reward", -float('inf'))
        stats.best_training_reward = data.get("best_training_reward", -float('inf'))
        stats.episodes_without_improvement = data.get("episodes_without_improvement", 0)
        return stats


class TrainingError(Exception):
    """训练相关的自定义异常"""
    pass


class EnvironmentError(TrainingError):
    """环境执行错误"""
    pass


class AgentError(TrainingError):
    """智能体执行错误"""
    pass


class BaseTrainer(ABC):
    """
    训练器基类 - 抽象训练流程
    """
    
    def __init__(self, config: TrainingConfig = None):
        self.config = config or TrainingConfig()
        self.stats = TrainingStats()
        self.start_time: Optional[float] = None
        self.current_episode: int = 0
        self.total_steps: int = 0
        self.stage: TrainingStage = TrainingStage.INITIAL
        self._logger: Optional[logging.Logger] = None
        self._log_file: Optional[str] = None
        self._progress_bar: Optional[Any] = None
        self._is_running: bool = False
        self._should_stop: bool = False
        
        self._setup_logging()
    
    def _setup_logging(self):
        """设置日志系统"""
        if not self.config.log_to_file and not self.config.log_to_console:
            return
        
        log_dir = Path(self.config.log_dir)
        if self.config.log_to_file and not log_dir.exists():
            try:
                log_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                print(f"警告: 无法创建日志目录 {log_dir}: {e}")
                self.config.log_to_file = False
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        logger_name = f"trainer_{timestamp}"
        self._logger = logging.getLogger(logger_name)
        self._logger.setLevel(self.config.log_level)
        self._logger.propagate = False
        
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        if self.config.log_to_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(self.config.log_level)
            console_handler.setFormatter(formatter)
            self._logger.addHandler(console_handler)
        
        if self.config.log_to_file:
            self._log_file = str(log_dir / f"training_{timestamp}.log")
            try:
                file_handler = logging.FileHandler(self._log_file, encoding='utf-8')
                file_handler.setLevel(self.config.log_level)
                file_handler.setFormatter(formatter)
                self._logger.addHandler(file_handler)
            except Exception as e:
                print(f"警告: 无法创建日志文件: {e}")
    
    def _log(self, message: str, level: int = logging.INFO):
        """记录日志"""
        if self._logger:
            self._logger.log(level, message)
        elif self.config.verbose and self.config.log_to_console:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            level_name = logging.getLevelName(level)
            print(f"[{timestamp}] {level_name} | {message}")
    
    def _log_exception(self, message: str, exc: Exception):
        """记录异常"""
        if self._logger:
            self._logger.error(f"{message}: {str(exc)}")
            self._logger.debug(traceback.format_exc())
        else:
            print(f"错误: {message}: {str(exc)}")
    
    @abstractmethod
    def train(
        self,
        agent: Agent,
        environment,
        eval_environment=None,
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
    
    def save_checkpoint(
        self,
        agent: Agent,
        filepath: str,
        extra_info: Dict[str, Any] = None
    ) -> bool:
        """
        保存训练检查点
        
        Args:
            agent: 智能体实例
            filepath: 保存路径
            extra_info: 额外信息
        
        Returns:
            是否保存成功
        """
        try:
            checkpoint = TrainingCheckpoint(
                episode=self.current_episode,
                total_steps=self.total_steps,
                timestamp=time.time(),
                config=asdict(self.config),
                stats=self.stats.to_dict(),
                agent_name=agent.name,
                agent_type=type(agent).__name__,
                best_reward=self.stats.best_eval_reward
            )
            
            checkpoint_data = {
                "checkpoint": asdict(checkpoint),
                "extra_info": extra_info or {}
            }
            
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
            
            self._log(f"检查点已保存: {filepath}", logging.INFO)
            return True
            
        except Exception as e:
            self._log_exception(f"保存检查点失败 {filepath}", e)
            return False
    
    def load_checkpoint(
        self,
        filepath: str
    ) -> Tuple[Optional[TrainingCheckpoint], Optional[Dict]]:
        """
        加载训练检查点
        
        Args:
            filepath: 检查点文件路径
        
        Returns:
            (检查点对象, 额外信息)
        """
        try:
            if not os.path.exists(filepath):
                self._log(f"检查点文件不存在: {filepath}", logging.WARNING)
                return None, None
            
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            checkpoint_data = data.get("checkpoint", {})
            extra_info = data.get("extra_info", {})
            
            checkpoint = TrainingCheckpoint(
                episode=checkpoint_data.get("episode", 0),
                total_steps=checkpoint_data.get("total_steps", 0),
                timestamp=checkpoint_data.get("timestamp", time.time()),
                config=checkpoint_data.get("config", {}),
                stats=checkpoint_data.get("stats", {}),
                agent_name=checkpoint_data.get("agent_name", ""),
                agent_type=checkpoint_data.get("agent_type", ""),
                best_reward=checkpoint_data.get("best_reward", -float('inf'))
            )
            
            self._log(f"检查点已加载: {filepath}", logging.INFO)
            return checkpoint, extra_info
            
        except Exception as e:
            self._log_exception(f"加载检查点失败 {filepath}", e)
            return None, None
    
    def pause(self):
        """暂停训练"""
        self._should_stop = True
        self.stage = TrainingStage.PAUSED
        self._log("训练已暂停", logging.INFO)
    
    def resume(self):
        """恢复训练"""
        self._should_stop = False
        self.stage = TrainingStage.TRAINING
        self._log("训练已恢复", logging.INFO)
    
    def stop(self):
        """停止训练"""
        self._should_stop = True
        self._is_running = False
        self.stage = TrainingStage.ERROR
        self._log("训练已停止", logging.INFO)
    
    def get_stats(self) -> TrainingStats:
        """获取训练统计"""
        return self.stats
    
    def get_current_episode(self) -> int:
        """获取当前回合数"""
        return self.current_episode
    
    def get_elapsed_time(self) -> float:
        """获取已用时间"""
        if self.start_time is None:
            return 0.0
        return time.time() - self.start_time
    
    def get_log_file(self) -> Optional[str]:
        """获取日志文件路径"""
        return self._log_file


class RLTrainer(BaseTrainer):
    """
    强化学习训练器 - 增强版
    
    核心改进：
    1. 完整的异常处理
    2. 经验回放自动采样学习
    3. 模型保存/加载（Checkpoint）
    4. 日志文件系统
    5. 进度条展示
    6. 评估与训练环境隔离
    7. epsilon 同步更新
    8. 早停机制
    """
    
    def __init__(self, config: TrainingConfig = None):
        super().__init__(config)
        
        if self.config.use_prioritized_replay:
            self.replay_buffer = PrioritizedExperienceBuffer(
                capacity=self.config.replay_buffer_capacity,
                alpha=self.config.prioritized_replay_alpha,
                beta=self.config.prioritized_replay_beta
            )
        else:
            self.replay_buffer = ExperienceBuffer(
                capacity=self.config.replay_buffer_capacity
            )
        
        self.episode_buffer = EpisodeBuffer()
        self._best_eval_reward = -float('inf')
        self._last_save_episode = 0
    
    def _safe_reset_environment(self, environment) -> Dict[str, Any]:
        """安全重置环境"""
        try:
            observation = environment.reset()
            if observation is None:
                observation = {}
            return observation
        except Exception as e:
            raise EnvironmentError(f"环境重置失败: {str(e)}") from e
    
    def _safe_execute_action(
        self, 
        environment, 
        tool_name: str, 
        parameters: Dict[str, Any]
    ) -> Tuple[Any, float, bool, Dict[str, Any]]:
        """安全执行动作"""
        try:
            result, reward, done, info = environment.execute_tool(tool_name, **parameters)
            
            if self.config.reward_shaping:
                shaped_reward = self.config.reward_shaping(reward, info)
                reward = shaped_reward
            
            return result, reward, done, info
        except Exception as e:
            raise EnvironmentError(f"执行工具 {tool_name} 失败: {str(e)}") from e
    
    def _safe_get_observation(self, environment) -> Dict[str, Any]:
        """安全获取观察"""
        try:
            observation = environment.get_observation()
            if observation is None:
                return {}
            return observation
        except Exception as e:
            self._log_exception("获取观察失败", e)
            return {}
    
    def _safe_agent_act(self, agent: Agent, observation: Dict[str, Any]) -> Optional[Action]:
        """安全调用智能体 act"""
        try:
            action = agent.act(observation)
            return action
        except Exception as e:
            raise AgentError(f"智能体 act 失败: {str(e)}") from e
    
    def _safe_agent_observe(
        self,
        agent: Agent,
        observation: Dict[str, Any],
        action: Action,
        reward: float,
        next_observation: Dict[str, Any],
        done: bool
    ):
        """安全调用智能体 observe"""
        try:
            agent.observe(observation, action, reward, next_observation, done)
        except Exception as e:
            self._log_exception("智能体 observe 失败", e)
    
    def _update_from_replay(
        self,
        agent: Agent,
        batch_size: Optional[int] = None
    ) -> Optional[float]:
        """
        从经验回放缓冲区采样并更新智能体
        
        这是关键改进：将经验回放与智能体更新集成
        """
        batch_size = batch_size or self.config.batch_size
        
        if len(self.replay_buffer) < self.config.min_replay_size:
            return None
        
        try:
            if self.config.use_prioritized_replay:
                states, actions, rewards, next_states, dones, weights, indices = \
                    self.replay_buffer.sample_batch(batch_size)
            else:
                states, actions, rewards, next_states, dones = \
                    self.replay_buffer.sample_batch(batch_size)
                weights = [1.0] * len(states)
                indices = None
            
            total_loss = 0.0
            td_errors = []
            
            for i, (state, action_dict, reward, next_state, done, weight) in enumerate(
                zip(states, actions, rewards, next_states, dones, weights)
            ):
                tool_name = action_dict.get("tool", "")
                parameters = action_dict.get("parameters", {})
                action = Action(tool=tool_name, parameters=parameters)
                
                if isinstance(agent, QLearningAgent):
                    state_key = agent._get_state_key(state)
                    next_state_key = agent._get_state_key(next_state)
                    
                    current_q = agent.q_table[state_key].get(tool_name, 0.0)
                    
                    if next_state_key in agent.q_table and agent.q_table[next_state_key]:
                        max_next_q = max(agent.q_table[next_state_key].values())
                    else:
                        max_next_q = 0.0
                    
                    target = reward + agent.discount_factor * max_next_q * (not done)
                    td_error = target - current_q
                    
                    new_q = current_q + agent.learning_rate * weight * td_error
                    agent.q_table[state_key][tool_name] = new_q
                    
                    td_errors.append(abs(td_error))
                    total_loss += td_error ** 2
                
                elif isinstance(agent, REINFORCEAgent):
                    pass
            
            if self.config.use_prioritized_replay and indices and td_errors:
                self.replay_buffer.update_priorities(indices, td_errors)
            
            avg_loss = total_loss / len(states) if states else 0.0
            self.stats.losses.append(avg_loss)
            
            return avg_loss
            
        except Exception as e:
            self._log_exception("从回放缓冲区学习失败", e)
            return None
    
    def _sync_agent_epsilon(self, agent: Agent, epsilon: float):
        """同步智能体的 epsilon 值"""
        if hasattr(agent, 'epsilon'):
            agent.epsilon = max(agent.epsilon_min if hasattr(agent, 'epsilon_min') else 0.01, epsilon)
    
    def train(
        self,
        agent: Agent,
        environment,
        eval_environment=None,
        callbacks: Dict[str, Callable] = None
    ) -> TrainingStats:
        """
        训练主循环
        
        Args:
            agent: 智能体实例
            environment: 训练环境
            eval_environment: 评估环境（可选，默认使用训练环境的副本）
            callbacks: 回调函数字典
        
        Returns:
            训练统计信息
        """
        callbacks = callbacks or {}
        self.start_time = time.time()
        self.stage = TrainingStage.TRAINING
        self._is_running = True
        self._should_stop = False
        
        epsilon = self.config.epsilon
        start_episode = 0
        
        if self.config.resume_from_checkpoint:
            checkpoint, extra = self.load_checkpoint(self.config.resume_from_checkpoint)
            if checkpoint:
                start_episode = checkpoint.episode
                self.total_steps = checkpoint.total_steps
                self.stats = TrainingStats.from_dict(checkpoint.stats)
                self._best_eval_reward = checkpoint.best_reward
                epsilon = self.stats.epsilon_values[-1] if self.stats.epsilon_values else epsilon
                self._log(f"从检查点恢复训练，起始回合: {start_episode}")
        
        actual_eval_env = eval_environment
        if self.config.separate_eval_env and actual_eval_env is None:
            from src.sandbox.environment_manager import EnvironmentManager
            manager = EnvironmentManager()
            instance = manager.create_environment(
                name="EvalEnv",
                type="base",
                config={}
            )
            actual_eval_env = manager.load_environment(instance)
            self._log("已创建独立的评估环境")
        
        self._log(f"开始训练，配置: {self.config.max_episodes} 回合, "
                  f"每回合最大步数: {self.config.max_steps_per_episode}")
        
        if self.config.show_progress and HAS_TQDM:
            self._progress_bar = tqdm(
                total=self.config.max_episodes,
                initial=start_episode,
                desc="训练进度",
                unit="episode"
            )
        
        for episode in range(start_episode, self.config.max_episodes):
            if self._should_stop:
                self._log(f"训练在回合 {episode} 被手动停止", logging.WARNING)
                break
            
            if self.config.max_time_seconds:
                elapsed = self.get_elapsed_time()
                if elapsed >= self.config.max_time_seconds:
                    self._log(f"达到最大训练时间限制: {self.config.max_time_seconds}s", logging.INFO)
                    break
            
            self.current_episode = episode
            
            try:
                observation = self._safe_reset_environment(environment)
            except EnvironmentError as e:
                self._log_exception(f"回合 {episode} 环境重置失败，跳过", e)
                continue
            
            episode_reward = 0.0
            episode_steps = 0
            episode_losses = []
            
            try:
                self.episode_buffer.start_episode()
            except Exception as e:
                self._log_exception("启动回合缓冲区失败", e)
            
            for step in range(self.config.max_steps_per_episode):
                if self._should_stop:
                    break
                
                try:
                    action = self._safe_agent_act(agent, observation)
                except AgentError as e:
                    self._log_exception(f"回合 {episode}, 步骤 {step} 智能体决策失败", e)
                    break
                
                if action is None:
                    self._log(f"回合 {episode}, 步骤 {step}: 智能体返回 None，结束回合", logging.DEBUG)
                    break
                
                tool_name = action.tool
                parameters = action.parameters
                
                try:
                    result, reward, done, info = self._safe_execute_action(
                        environment, tool_name, parameters
                    )
                except EnvironmentError as e:
                    self._log_exception(f"回合 {episode}, 步骤 {step} 执行动作失败", e)
                    reward = -10.0
                    done = True
                    info = {"error": str(e)}
                
                try:
                    next_observation = self._safe_get_observation(environment)
                except Exception as e:
                    next_observation = observation.copy() if observation else {}
                
                self._safe_agent_observe(
                    agent, observation, action, reward, next_observation, done
                )
                
                try:
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
                except Exception as e:
                    self._log_exception("添加经验到缓冲区失败", e)
                
                self.total_steps += 1
                episode_steps += 1
                episode_reward += reward
                
                if self.total_steps % self.config.replay_update_freq == 0:
                    loss = self._update_from_replay(agent)
                    if loss is not None:
                        episode_losses.append(loss)
                
                if callbacks.get("on_step"):
                    try:
                        callbacks["on_step"]({
                            "episode": episode,
                            "step": step,
                            "action": action,
                            "reward": reward,
                            "observation": observation,
                            "next_observation": next_observation,
                            "done": done,
                            "total_steps": self.total_steps
                        })
                    except Exception as e:
                        self._log_exception("执行 on_step 回调失败", e)
                
                observation = next_observation
                
                if done:
                    break
            
            self.stats.episode_rewards.append(episode_reward)
            self.stats.episode_lengths.append(episode_steps)
            self.stats.epsilon_values.append(epsilon)
            self.stats.timestamps.append(self.get_elapsed_time())
            self.stats.learning_rates.append(
                agent.learning_rate if hasattr(agent, 'learning_rate') else self.config.learning_rate
            )
            
            epsilon = max(self.config.epsilon_min, epsilon * self.config.epsilon_decay)
            self._sync_agent_epsilon(agent, epsilon)
            
            if episode_reward > self.stats.best_training_reward:
                self.stats.best_training_reward = episode_reward
                self.stats.episodes_without_improvement = 0
            else:
                self.stats.episodes_without_improvement += 1
            
            if callbacks.get("on_episode"):
                try:
                    callbacks["on_episode"]({
                        "episode": episode,
                        "reward": episode_reward,
                        "length": episode_steps,
                        "epsilon": epsilon,
                        "losses": episode_losses
                    })
                except Exception as e:
                    self._log_exception("执行 on_episode 回调失败", e)
            
            if (episode + 1) % self.config.eval_freq == 0 and actual_eval_env:
                try:
                    eval_stats = self.evaluate(agent, actual_eval_env)
                    self.stats.eval_rewards.append(eval_stats["avg_reward"])
                    
                    if eval_stats["avg_reward"] > self._best_eval_reward:
                        self._best_eval_reward = eval_stats["avg_reward"]
                        self.stats.best_eval_reward = self._best_eval_reward
                        self.stats.episodes_without_improvement = 0
                        
                        best_path = os.path.join(self.config.log_dir, "best_model_checkpoint.json")
                        if hasattr(agent, 'save'):
                            try:
                                agent.save(best_path.replace("checkpoint", "agent"))
                            except Exception:
                                pass
                    
                    self._log(f"回合 {episode + 1}: 评估平均奖励={eval_stats['avg_reward']:.2f}, "
                              f"最佳={self._best_eval_reward:.2f}, "
                              f"训练奖励={episode_reward:.2f}, epsilon={epsilon:.4f}")
                except Exception as e:
                    self._log_exception("评估失败", e)
            
            if self.config.early_stop_patience > 0:
                if self.stats.episodes_without_improvement >= self.config.early_stop_patience:
                    self._log(f"早停触发: 连续 {self.config.early_stop_patience} 回合没有改进", logging.INFO)
                    break
            
            if (episode + 1) % self.config.save_freq == 0 or \
               (episode + 1) == self.config.max_episodes:
                try:
                    checkpoint_path = os.path.join(
                        self.config.log_dir, 
                        f"checkpoint_episode_{episode + 1}.json"
                    )
                    self.save_checkpoint(agent, checkpoint_path)
                    self._last_save_episode = episode
                except Exception as e:
                    self._log_exception("保存检查点失败", e)
            
            if (episode + 1) % 10 == 0 or episode == start_episode:
                recent_avg = sum(self.stats.episode_rewards[-10:]) / min(10, len(self.stats.episode_rewards))
                self._log(f"回合 {episode + 1}: 最近10回合平均奖励={recent_avg:.2f}, "
                          f"epsilon={epsilon:.4f}, 缓冲区大小={len(self.replay_buffer)}")
            
            if self._progress_bar:
                self._progress_bar.update(1)
                self._progress_bar.set_postfix({
                    'reward': f'{episode_reward:.1f}',
                    'epsilon': f'{epsilon:.3f}',
                    'steps': episode_steps
                })
        
        self._is_running = False
        self.stage = TrainingStage.COMPLETED
        
        if self._progress_bar:
            self._progress_bar.close()
        
        final_checkpoint = os.path.join(self.config.log_dir, "final_checkpoint.json")
        self.save_checkpoint(agent, final_checkpoint, {
            "completed": True,
            "elapsed_time": self.get_elapsed_time()
        })
        
        summary = self.stats.get_summary()
        self._log(f"训练完成！总回合数: {self.current_episode + 1}")
        self._log(f"  总步数: {summary['total_steps']}")
        self._log(f"  平均奖励: {summary['avg_reward']:.2f}")
        self._log(f"  最佳训练奖励: {summary['best_training_reward']:.2f}")
        self._log(f"  最佳评估奖励: {summary['best_eval_reward']:.2f}")
        self._log(f"  耗时: {self.get_elapsed_time():.1f} 秒")
        
        return self.stats
    
    def evaluate(
        self,
        agent: Agent,
        environment,
        num_episodes: int = None
    ) -> Dict[str, Any]:
        """
        评估智能体性能
        
        Args:
            agent: 智能体实例
            environment: 评估环境
            num_episodes: 评估回合数
        
        Returns:
            评估统计信息
        """
        num_episodes = num_episodes or self.config.eval_episodes
        
        all_rewards: List[float] = []
        all_lengths: List[int] = []
        all_successes: List[bool] = []
        
        original_epsilon = getattr(agent, 'epsilon', None)
        if original_epsilon is not None:
            agent.epsilon = 0.0
        
        original_training = getattr(agent, 'training', True)
        if hasattr(agent, 'training'):
            agent.training = False
        
        self.stage = TrainingStage.EVALUATING
        
        for episode in range(num_episodes):
            try:
                observation = self._safe_reset_environment(environment)
            except EnvironmentError as e:
                self._log_exception(f"评估回合 {episode} 环境重置失败", e)
                continue
            
            episode_reward = 0.0
            episode_steps = 0
            episode_success = False
            
            for step in range(self.config.max_steps_per_episode):
                try:
                    action = self._safe_agent_act(agent, observation)
                except AgentError as e:
                    self._log_exception(f"评估回合 {episode}, 步骤 {step} 智能体决策失败", e)
                    break
                
                if action is None:
                    break
                
                try:
                    result, reward, done, info = self._safe_execute_action(
                        environment, action.tool, action.parameters
                    )
                except EnvironmentError as e:
                    self._log(f"评估回合 {episode}, 步骤 {step} 执行失败: {e}", logging.DEBUG)
                    reward = 0.0
                    done = True
                
                episode_reward += reward
                episode_steps += 1
                
                try:
                    observation = self._safe_get_observation(environment)
                except Exception:
                    pass
                
                if done:
                    episode_success = info.get('success', reward > 0)
                    break
            
            all_rewards.append(episode_reward)
            all_lengths.append(episode_steps)
            all_successes.append(episode_success)
        
        if original_epsilon is not None:
            agent.epsilon = original_epsilon
        
        if hasattr(agent, 'training'):
            agent.training = original_training
        
        self.stage = TrainingStage.TRAINING
        
        success_rate = sum(all_successes) / len(all_successes) if all_successes else 0.0
        
        return {
            "avg_reward": sum(all_rewards) / len(all_rewards) if all_rewards else 0.0,
            "max_reward": max(all_rewards) if all_rewards else 0.0,
            "min_reward": min(all_rewards) if all_rewards else 0.0,
            "avg_length": sum(all_lengths) / len(all_lengths) if all_lengths else 0,
            "success_rate": success_rate,
            "episodes": num_episodes,
            "all_rewards": all_rewards,
            "all_lengths": all_lengths
        }


class CurriculumLevel:
    """课程学习级别配置"""
    
    def __init__(
        self,
        level_id: str,
        difficulty: int,
        environment_creator: Callable,
        min_success_rate: float = 0.7,
        min_avg_reward: Optional[float] = None,
        min_consecutive_success: int = 3,
        max_episodes_per_level: int = 100,
        description: str = ""
    ):
        self.level_id = level_id
        self.difficulty = difficulty
        self.environment_creator = environment_creator
        self.min_success_rate = min_success_rate
        self.min_avg_reward = min_avg_reward
        self.min_consecutive_success = min_consecutive_success
        self.max_episodes_per_level = max_episodes_per_level
        self.description = description
        self.environment = None
        
        self.episodes_trained = 0
        self.consecutive_successes = 0
        self.recent_rewards: List[float] = []
        self.recent_successes: List[bool] = []
        self.completed = False
    
    def get_environment(self):
        """获取该级别的环境实例"""
        if self.environment is None:
            try:
                self.environment = self.environment_creator()
            except Exception as e:
                raise EnvironmentError(f"创建课程级别 {self.level_id} 环境失败: {e}")
        return self.environment
    
    def check_advancement(
        self,
        episode_reward: float,
        episode_success: bool
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        检查是否满足升级条件
        
        Returns:
            (是否可以升级, 详细信息)
        """
        self.episodes_trained += 1
        self.recent_rewards.append(episode_reward)
        self.recent_successes.append(episode_success)
        
        if len(self.recent_rewards) > 20:
            self.recent_rewards = self.recent_rewards[-20:]
            self.recent_successes = self.recent_successes[-20:]
        
        if episode_success:
            self.consecutive_successes += 1
        else:
            self.consecutive_successes = 0
        
        can_advance = False
        reasons = []
        
        if self.consecutive_successes >= self.min_consecutive_success:
            can_advance = True
            reasons.append(f"连续成功 {self.consecutive_successes} 次")
        
        if self.recent_successes:
            recent_success_rate = sum(self.recent_successes) / len(self.recent_successes)
            if recent_success_rate >= self.min_success_rate and len(self.recent_successes) >= 10:
                can_advance = True
                reasons.append(f"最近成功率 {recent_success_rate:.2f} >= {self.min_success_rate}")
        
        if self.min_avg_reward is not None and self.recent_rewards:
            recent_avg = sum(self.recent_rewards) / len(self.recent_rewards)
            if recent_avg >= self.min_avg_reward:
                can_advance = True
                reasons.append(f"最近平均奖励 {recent_avg:.2f} >= {self.min_avg_reward}")
        
        if self.episodes_trained >= self.max_episodes_per_level:
            can_advance = True
            reasons.append(f"达到该级别最大回合数 {self.max_episodes_per_level}")
        
        info = {
            "level_id": self.level_id,
            "episodes_trained": self.episodes_trained,
            "consecutive_successes": self.consecutive_successes,
            "recent_success_rate": sum(self.recent_successes) / len(self.recent_successes) if self.recent_successes else 0,
            "recent_avg_reward": sum(self.recent_rewards) / len(self.recent_rewards) if self.recent_rewards else 0
        }
        
        return can_advance, info
    
    def reset(self):
        """重置该级别状态"""
        self.episodes_trained = 0
        self.consecutive_successes = 0
        self.recent_rewards = []
        self.recent_successes = []
        self.completed = False
        if self.environment is not None:
            try:
                self.environment.reset()
            except Exception:
                pass


class CurriculumTrainer(RLTrainer):
    """
    课程学习训练器 - 改进版
    
    核心改进：
    1. 结构化的课程级别配置
    2. 灵活的升级条件（成功率、平均奖励、连续成功）
    3. 每个级别独立的环境
    4. 进度追踪和日志
    5. 异常处理
    """
    
    def __init__(self, config: TrainingConfig = None):
        super().__init__(config)
        self.curriculum_levels: List[CurriculumLevel] = []
        self.current_level_index: int = 0
        self._level_environments: Dict[str, Any] = {}
    
    def add_level(
        self,
        level_id: str,
        difficulty: int,
        environment_creator: Callable,
        min_success_rate: float = 0.7,
        min_avg_reward: Optional[float] = None,
        min_consecutive_success: int = 3,
        max_episodes_per_level: int = 100,
        description: str = ""
    ):
        """
        添加课程级别
        
        Args:
            level_id: 级别唯一标识
            difficulty: 难度值（用于排序）
            environment_creator: 环境创建函数
            min_success_rate: 最小成功率要求
            min_avg_reward: 最小平均奖励要求（可选）
            min_consecutive_success: 最小连续成功次数
            max_episodes_per_level: 每级别最大回合数
            description: 级别描述
        """
        level = CurriculumLevel(
            level_id=level_id,
            difficulty=difficulty,
            environment_creator=environment_creator,
            min_success_rate=min_success_rate,
            min_avg_reward=min_avg_reward,
            min_consecutive_success=min_consecutive_success,
            max_episodes_per_level=max_episodes_per_level,
            description=description
        )
        
        self.curriculum_levels.append(level)
        self.curriculum_levels.sort(key=lambda x: x.difficulty)
        
        self._log(f"已添加课程级别: {level_id} (难度: {difficulty})")
    
    def train(
        self,
        agent: Agent,
        initial_environment=None,
        eval_environment=None,
        callbacks: Dict[str, Callable] = None
    ) -> TrainingStats:
        """
        课程学习训练主循环
        
        Args:
            agent: 智能体实例
            initial_environment: 初始环境（可选，使用第一级别环境）
            eval_environment: 评估环境
            callbacks: 回调函数
        
        Returns:
            训练统计信息
        """
        if not self.curriculum_levels:
            raise ValueError("没有定义任何课程级别，请先使用 add_level() 添加")
        
        callbacks = callbacks or {}
        self.start_time = time.time()
        self.stage = TrainingStage.TRAINING
        self._is_running = True
        self._should_stop = False
        
        epsilon = self.config.epsilon
        
        self._log(f"开始课程学习训练，共 {len(self.curriculum_levels)} 个级别")
        
        for level_idx, level in enumerate(self.curriculum_levels):
            if self._should_stop:
                break
            
            self.current_level_index = level_idx
            level.reset()
            
            self._log(f"\n{'='*60}")
            self._log(f"进入级别 {level_idx + 1}: {level.level_id}")
            self._log(f"  描述: {level.description}")
            self._log(f"  难度: {level.difficulty}")
            self._log(f"  升级条件:")
            self._log(f"    - 连续成功: {level.min_consecutive_success} 次")
            self._log(f"    - 成功率: {level.min_success_rate:.0%}")
            if level.min_avg_reward:
                self._log(f"    - 平均奖励: {level.min_avg_reward}")
            self._log(f"{'='*60}\n")
            
            try:
                environment = level.get_environment()
            except EnvironmentError as e:
                self._log_exception(f"无法创建级别 {level.level_id} 的环境，跳过", e)
                continue
            
            for episode in range(level.max_episodes_per_level):
                if self._should_stop:
                    break
                
                self.current_episode = episode
                
                try:
                    observation = self._safe_reset_environment(environment)
                except EnvironmentError as e:
                    self._log_exception(f"回合 {episode} 环境重置失败", e)
                    continue
                
                episode_reward = 0.0
                episode_steps = 0
                episode_success = False
                
                for step in range(self.config.max_steps_per_episode):
                    if self._should_stop:
                        break
                    
                    try:
                        action = self._safe_agent_act(agent, observation)
                    except AgentError as e:
                        self._log_exception(f"智能体决策失败", e)
                        break
                    
                    if action is None:
                        break
                    
                    try:
                        result, reward, done, info = self._safe_execute_action(
                            environment, action.tool, action.parameters
                        )
                    except EnvironmentError as e:
                        reward = -5.0
                        done = True
                        info = {"error": str(e)}
                    
                    try:
                        next_observation = self._safe_get_observation(environment)
                    except Exception:
                        next_observation = observation.copy()
                    
                    self._safe_agent_observe(
                        agent, observation, action, reward, next_observation, done
                    )
                    
                    try:
                        self.replay_buffer.push_transition(
                            state=observation,
                            action={"tool": action.tool, "parameters": action.parameters},
                            reward=reward,
                            next_state=next_observation,
                            done=done
                        )
                    except Exception:
                        pass
                    
                    if self.total_steps % self.config.replay_update_freq == 0:
                        self._update_from_replay(agent)
                    
                    self.total_steps += 1
                    episode_steps += 1
                    episode_reward += reward
                    
                    if callbacks.get("on_step"):
                        try:
                            callbacks["on_step"]({
                                "level": level_idx,
                                "episode": episode,
                                "step": step,
                                "reward": reward
                            })
                        except Exception:
                            pass
                    
                    observation = next_observation
                    
                    if done:
                        episode_success = info.get('success', reward > 0)
                        break
                
                self.stats.episode_rewards.append(episode_reward)
                self.stats.episode_lengths.append(episode_steps)
                self.stats.epsilon_values.append(epsilon)
                self.stats.timestamps.append(self.get_elapsed_time())
                
                epsilon = max(self.config.epsilon_min, epsilon * self.config.epsilon_decay)
                self._sync_agent_epsilon(agent, epsilon)
                
                can_advance, advance_info = level.check_advancement(episode_reward, episode_success)
                
                if (episode + 1) % 10 == 0:
                    self._log(
                        f"级别 {level_idx + 1}, 回合 {episode + 1}: "
                        f"奖励={episode_reward:.2f}, "
                        f"连续成功={advance_info['consecutive_successes']}, "
                        f"成功率={advance_info['recent_success_rate']:.0%}, "
                        f"epsilon={epsilon:.4f}"
                    )
                
                if can_advance:
                    level.completed = True
                    self._log(f"\n✓ 级别 {level_idx + 1} 完成！")
                    self._log(f"  连续成功: {advance_info['consecutive_successes']} 次")
                    self._log(f"  训练回合数: {level.episodes_trained}")
                    self._log(f"  平均奖励: {advance_info['recent_avg_reward']:.2f}")
                    
                    if callbacks.get("on_level_up"):
                        try:
                            callbacks["on_level_up"]({
                                "from_level": level_idx,
                                "to_level": level_idx + 1,
                                "level_id": level.level_id,
                                "info": advance_info
                            })
                        except Exception:
                            pass
                    
                    try:
                        checkpoint_path = os.path.join(
                            self.config.log_dir,
                            f"level_{level_idx + 1}_complete_checkpoint.json"
                        )
                        self.save_checkpoint(agent, checkpoint_path, {
                            "completed_level": level.level_id,
                            "level_index": level_idx
                        })
                    except Exception:
                        pass
                    
                    break
        
        all_completed = all(level.completed for level in self.curriculum_levels)
        
        self._is_running = False
        self.stage = TrainingStage.COMPLETED
        
        self._log(f"\n{'='*60}")
        self._log("课程学习训练完成！")
        self._log(f"  通过级别数: {sum(1 for l in self.curriculum_levels if l.completed)} / {len(self.curriculum_levels)}")
        self._log(f"  总回合数: {len(self.stats.episode_rewards)}")
        self._log(f"  总步数: {self.total_steps}")
        self._log(f"  耗时: {self.get_elapsed_time():.1f} 秒")
        self._log(f"{'='*60}")
        
        try:
            final_path = os.path.join(self.config.log_dir, "curriculum_final_checkpoint.json")
            self.save_checkpoint(agent, final_path, {
                "levels_completed": [l.level_id for l in self.curriculum_levels if l.completed],
                "all_completed": all_completed
            })
        except Exception:
            pass
        
        return self.stats
    
    def get_current_level(self) -> Optional[CurriculumLevel]:
        """获取当前课程级别"""
        if 0 <= self.current_level_index < len(self.curriculum_levels):
            return self.curriculum_levels[self.current_level_index]
        return None
    
    def get_level_progress(self) -> List[Dict[str, Any]]:
        """获取所有级别的进度"""
        return [
            {
                "level_id": level.level_id,
                "difficulty": level.difficulty,
                "episodes_trained": level.episodes_trained,
                "consecutive_successes": level.consecutive_successes,
                "completed": level.completed,
                "description": level.description
            }
            for level in self.curriculum_levels
        ]
