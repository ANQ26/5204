"""
训练器模块 - 用于协调强化学习训练流程

改进版本：
- 完整的异常处理机制
- 经验回放与智能体更新自动集成
- 模型保存/加载（Checkpoint 机制）
- 断点续训：中断后可加载模型接续训练
- 自动保存最优模型：留存训练过程中效果最好的权重文件
- 简易控制台进度条：实时展示单回合与整体训练进度
- 运行异常捕获模块：报错自动记录日志且程序不会直接终止
- 早停判定机制：连续多轮收益无提升时自动终止无效训练
- 日志文件系统
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
import signal
from typing import Dict, List, Optional, Any, Callable, Tuple, Union, Type
from dataclasses import dataclass, field, asdict
from datetime import datetime
from abc import ABC, abstractmethod
from pathlib import Path
from enum import Enum, auto
from contextlib import contextmanager

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


class ConsoleProgressBar:
    """
    简易控制台实时进度条 - 不依赖第三方库

    多层信息展示:
    - 总体训练进度 (回合完成比例、ETA、速度)
    - 当前阶段/级别标注
    - 实时指标面板 (奖励、epsilon、最佳记录等)
    - 单回合步骤子进度条 (可选)

    用户全程直观掌握训练状态。
    """

    def __init__(self, total: int, bar_width: int = 40, desc: str = ""):
        self.total = total
        self.bar_width = bar_width
        self.desc = desc
        self.current = 0
        self._start_time = time.time()
        self._last_print_time = 0.0
        self._min_interval = 0.08
        self._fill_char = "#"
        self._empty_char = "."
        self._lines_printed = 0
        self._stage_desc: str = ""
        self._sub_progress: Optional[Tuple[int, int]] = None
        self._metrics: Dict[str, str] = {}
        self._speed_window: List[float] = []

    def set_stage(self, desc: str):
        """设置当前阶段描述（如课程级别名称）"""
        self._stage_desc = desc

    def set_sub_progress(self, current_step: int, max_steps: int):
        """设置单回合内的步骤子进度"""
        self._sub_progress = (current_step, max_steps)

    def clear_sub_progress(self):
        """清除子进度（回合结束时调用）"""
        self._sub_progress = None

    def set_metrics(self, metrics: Dict[str, str]):
        """设置实时指标面板内容"""
        self._metrics = metrics

    def update(self, n: int = 1, postfix: Optional[Dict[str, str]] = None,
               step_info: Optional[str] = None):
        """更新进度并刷新显示"""
        self.current += n
        now = time.time()

        self._speed_window.append(now)
        if len(self._speed_window) > 20:
            self._speed_window = self._speed_window[-20:]

        if now - self._last_print_time < self._min_interval and self.current < self.total:
            return
        self._last_print_time = now

        if postfix:
            self._metrics.update(postfix)

        self._render(step_info)

    def _render(self, step_info: Optional[str] = None):
        """渲染多行进度显示"""
        if self._lines_printed > 0:
            sys.stdout.write(f"\033[{self._lines_printed}A\033[J")

        lines = []

        progress = self.current / self.total if self.total > 0 else 0
        filled = int(self.bar_width * progress)
        bar = self._fill_char * filled + self._empty_char * (self.bar_width - filled)

        elapsed = time.time() - self._start_time
        speed = self._calc_speed()
        if self.current > 0 and self.current < self.total:
            eta = elapsed / self.current * (self.total - self.current)
            eta_str = self._format_time(eta)
        elif self.current >= self.total:
            eta_str = "0s"
        else:
            eta_str = "??:??"

        elapsed_str = self._format_time(elapsed)
        speed_str = f"{speed:.1f}ep/s" if speed > 0 else ""

        header = self.desc
        if self._stage_desc:
            header = f"{self.desc} [{self._stage_desc}]"

        line1 = (f"  {header} |{bar}| {self.current}/{self.total} "
                 f"{progress*100:.0f}% [{elapsed_str}<{eta_str}] {speed_str}")
        lines.append(line1)

        if self._metrics:
            metric_parts = [f"{k}={v}" for k, v in self._metrics.items()]
            line2 = "  " + " | ".join(metric_parts)
            lines.append(line2)

        if self._sub_progress:
            sub_cur, sub_max = self._sub_progress
            sub_progress = sub_cur / sub_max if sub_max > 0 else 0
            sub_width = 20
            sub_filled = int(sub_width * sub_progress)
            sub_bar = self._fill_char * sub_filled + self._empty_char * (sub_width - sub_filled)
            line3 = f"  step |{sub_bar}| {sub_cur}/{sub_max}"
            if step_info:
                line3 += f"  {step_info}"
            lines.append(line3)
        elif step_info:
            lines.append(f"  {step_info}")

        output = "\n".join(lines)
        sys.stdout.write(output + "\n")
        sys.stdout.flush()
        self._lines_printed = len(lines)

    def _calc_speed(self) -> float:
        """计算最近的训练速度 (episodes/sec)"""
        if len(self._speed_window) < 2:
            return 0.0
        duration = self._speed_window[-1] - self._speed_window[0]
        if duration <= 0:
            return 0.0
        return (len(self._speed_window) - 1) / duration

    def close(self):
        """关闭进度条，打印最终状态"""
        self._sub_progress = None
        if self.current < self.total:
            self.current = self.total
        self._render()
        self._lines_printed = 0

    @staticmethod
    def _format_time(seconds: float) -> str:
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            m, s = divmod(int(seconds), 60)
            return f"{m}m{s:02d}s"
        else:
            h, remainder = divmod(int(seconds), 3600)
            m, s = divmod(remainder, 60)
            return f"{h}h{m:02d}m"


class ExceptionGuard:
    """
    运行异常捕获模块

    捕获训练过程中的各类异常，自动记录日志且程序不会直接终止。
    支持异常分类统计、错误恢复建议。
    """

    def __init__(self, logger_instance: Optional[logging.Logger] = None,
                 log_dir: str = "./logs"):
        self._logger = logger_instance
        self._log_dir = Path(log_dir)
        self._error_counts: Dict[str, int] = {}
        self._error_history: List[Dict[str, Any]] = []
        self._max_consecutive_errors = 10
        self._consecutive_errors = 0
        self._crash_log_path: Optional[str] = None

        self._ensure_log_dir()

    def _ensure_log_dir(self):
        try:
            self._log_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self._crash_log_path = str(self._log_dir / f"crash_log_{timestamp}.jsonl")
        except Exception:
            self._crash_log_path = None

    def catch(self, context: str, exc: Exception, critical: bool = False) -> bool:
        """
        捕获并记录异常

        Args:
            context: 异常发生的上下文描述
            exc: 异常实例
            critical: 是否为致命错误

        Returns:
            True 表示可以继续执行, False 表示应当终止
        """
        error_type = type(exc).__name__
        self._error_counts[error_type] = self._error_counts.get(error_type, 0) + 1
        self._consecutive_errors += 1

        error_record = {
            "timestamp": datetime.now().isoformat(),
            "context": context,
            "error_type": error_type,
            "message": str(exc),
            "traceback": traceback.format_exc(),
            "critical": critical,
            "consecutive_count": self._consecutive_errors
        }
        self._error_history.append(error_record)

        self._write_crash_log(error_record)

        if self._logger:
            level = logging.CRITICAL if critical else logging.ERROR
            self._logger.log(level, f"[异常捕获] {context}: {error_type}: {exc}")
            self._logger.debug(f"详细追踪:\n{traceback.format_exc()}")

        if critical:
            return False
        if self._consecutive_errors >= self._max_consecutive_errors:
            if self._logger:
                self._logger.critical(
                    f"连续错误达到 {self._max_consecutive_errors} 次上限，建议终止训练"
                )
            return False

        return True

    def reset_consecutive(self):
        self._consecutive_errors = 0

    def _write_crash_log(self, record: Dict[str, Any]):
        if not self._crash_log_path:
            return
        try:
            with open(self._crash_log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def get_summary(self) -> Dict[str, Any]:
        return {
            "total_errors": len(self._error_history),
            "error_types": dict(self._error_counts),
            "crash_log_path": self._crash_log_path,
            "consecutive_errors": self._consecutive_errors
        }


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
    early_stop_metric: str = "eval_reward"
    resume_from_checkpoint: Optional[str] = None
    resume_agent_path: Optional[str] = None
    separate_eval_env: bool = True
    reward_shaping: Optional[Callable[[float, Dict], float]] = None
    gradient_clip: Optional[float] = None
    best_model_dir: Optional[str] = None
    auto_save_best: bool = True
    checkpoint_on_interrupt: bool = True


class EarlyStopMonitor:
    """
    早停判定机制

    连续多轮收益无提升时自动终止无效训练。
    支持多种指标监控（评估奖励、训练奖励、损失），
    可配置最小改进阈值和耐心轮数。
    """

    def __init__(self, patience: int = 50, min_improvement: float = 0.01,
                 metric: str = "eval_reward"):
        """
        Args:
            patience: 连续无改进回合数上限
            min_improvement: 被视为有效改进的最小增量
            metric: 监控指标 ("eval_reward", "train_reward", "loss")
        """
        self.patience = patience
        self.min_improvement = min_improvement
        self.metric = metric
        self._best_value: Optional[float] = None
        self._rounds_without_improvement: int = 0
        self._history: List[float] = []
        self._triggered: bool = False
        self._trigger_reason: str = ""

    def step(self, value: float) -> bool:
        """
        报告新一轮的指标值

        Args:
            value: 当前轮的指标值

        Returns:
            True 表示应当继续训练，False 表示应当触发早停
        """
        self._history.append(value)

        if self._best_value is None:
            self._best_value = value
            self._rounds_without_improvement = 0
            return True

        if self.metric == "loss":
            improved = (self._best_value - value) > self.min_improvement
        else:
            improved = (value - self._best_value) > self.min_improvement

        if improved:
            self._best_value = value
            self._rounds_without_improvement = 0
        else:
            self._rounds_without_improvement += 1

        if self._rounds_without_improvement >= self.patience:
            self._triggered = True
            self._trigger_reason = (
                f"连续 {self._rounds_without_improvement} 轮无有效改进 "
                f"(最佳={self._best_value:.4f}, 当前={value:.4f}, "
                f"最小改进阈值={self.min_improvement})"
            )
            return False

        return True

    @property
    def triggered(self) -> bool:
        return self._triggered

    @property
    def reason(self) -> str:
        return self._trigger_reason

    @property
    def rounds_without_improvement(self) -> int:
        return self._rounds_without_improvement

    @property
    def best_value(self) -> Optional[float]:
        return self._best_value

    def reset(self):
        self._best_value = None
        self._rounds_without_improvement = 0
        self._history = []
        self._triggered = False
        self._trigger_reason = ""

    def get_state(self) -> Dict[str, Any]:
        return {
            "patience": self.patience,
            "min_improvement": self.min_improvement,
            "metric": self.metric,
            "best_value": self._best_value,
            "rounds_without_improvement": self._rounds_without_improvement,
            "triggered": self._triggered,
            "history_length": len(self._history)
        }

    def load_state(self, state: Dict[str, Any]):
        self._best_value = state.get("best_value")
        self._rounds_without_improvement = state.get("rounds_without_improvement", 0)
        self._triggered = state.get("triggered", False)


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
    early_stop_state: Optional[Dict[str, Any]] = None
    agent_save_path: Optional[str] = None


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

    集成功能：
    - 断点续训（checkpoint resume）
    - 自动保存最优模型
    - 控制台进度条
    - 异常捕获模块
    - 早停判定
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

        self._exception_guard: Optional[ExceptionGuard] = None
        self._early_stop_monitor: Optional[EarlyStopMonitor] = None
        self._best_model_path: Optional[str] = None
        self._interrupt_checkpoint_saved: bool = False

        self._setup_logging()
        self._setup_exception_guard()
        self._setup_early_stop()
        self._setup_signal_handlers()
    
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
        if self._exception_guard:
            can_continue = self._exception_guard.catch(message, exc)
            if not can_continue:
                self._log(f"致命异常累积过多，标记停止: {message}", logging.CRITICAL)
                self._should_stop = True
        if self._logger:
            self._logger.error(f"{message}: {str(exc)}")
            self._logger.debug(traceback.format_exc())
        else:
            print(f"错误: {message}: {str(exc)}")

    def _setup_exception_guard(self):
        """初始化异常捕获模块"""
        self._exception_guard = ExceptionGuard(
            logger_instance=self._logger,
            log_dir=self.config.log_dir
        )

    def _setup_early_stop(self):
        """初始化早停监控器"""
        if self.config.early_stop_patience > 0:
            self._early_stop_monitor = EarlyStopMonitor(
                patience=self.config.early_stop_patience,
                min_improvement=self.config.early_stop_min_improvement,
                metric=self.config.early_stop_metric
            )

    def _setup_signal_handlers(self):
        """设置信号处理，确保中断时保存检查点"""
        if not self.config.checkpoint_on_interrupt:
            return
        try:
            original_sigint = signal.getsignal(signal.SIGINT)

            def _interrupt_handler(signum, frame):
                self._log("收到中断信号，正在保存当前状态...", logging.WARNING)
                self._should_stop = True
                if original_sigint and callable(original_sigint):
                    signal.signal(signal.SIGINT, original_sigint)

            signal.signal(signal.SIGINT, _interrupt_handler)
        except (OSError, ValueError):
            pass
    
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
        保存训练检查点（含智能体权重）

        Args:
            agent: 智能体实例
            filepath: 保存路径
            extra_info: 额外信息

        Returns:
            是否保存成功
        """
        try:
            agent_save_path = filepath.replace(".json", "_agent.json")
            agent_saved = False
            if hasattr(agent, 'save'):
                agent_saved = agent.save(agent_save_path)

            early_stop_state = None
            if self._early_stop_monitor:
                early_stop_state = self._early_stop_monitor.get_state()

            checkpoint = TrainingCheckpoint(
                episode=self.current_episode,
                total_steps=self.total_steps,
                timestamp=time.time(),
                config=asdict(self.config),
                stats=self.stats.to_dict(),
                agent_name=agent.name,
                agent_type=type(agent).__name__,
                best_reward=self.stats.best_eval_reward,
                early_stop_state=early_stop_state,
                agent_save_path=agent_save_path if agent_saved else None
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

    def save_best_model(self, agent: Agent, metric_value: float) -> bool:
        """
        自动保存最优模型

        在训练过程中，当检测到新的最佳性能时自动调用。
        保存智能体权重和对应的性能指标。

        Args:
            agent: 智能体实例
            metric_value: 当前最优指标值

        Returns:
            是否保存成功
        """
        try:
            best_dir = self.config.best_model_dir or self.config.log_dir
            os.makedirs(best_dir, exist_ok=True)

            best_agent_path = os.path.join(best_dir, "best_model_agent.json")
            best_meta_path = os.path.join(best_dir, "best_model_meta.json")

            agent_saved = False
            if hasattr(agent, 'save'):
                agent_saved = agent.save(best_agent_path)

            meta = {
                "metric_value": metric_value,
                "metric_name": self.config.early_stop_metric,
                "episode": self.current_episode,
                "total_steps": self.total_steps,
                "timestamp": datetime.now().isoformat(),
                "agent_name": agent.name,
                "agent_type": type(agent).__name__,
                "agent_path": best_agent_path if agent_saved else None
            }
            with open(best_meta_path, 'w', encoding='utf-8') as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)

            self._best_model_path = best_agent_path
            self._log(
                f"最优模型已保存: {best_agent_path} (指标={metric_value:.4f})",
                logging.INFO
            )
            return True

        except Exception as e:
            self._log_exception("保存最优模型失败", e)
            return False
    
    def load_checkpoint(
        self,
        filepath: str,
        agent: Optional[Agent] = None
    ) -> Tuple[Optional[TrainingCheckpoint], Optional[Dict]]:
        """
        加载训练检查点（断点续训）

        支持加载完整训练状态，包括智能体权重、训练统计、早停状态等，
        实现中断后无缝接续训练。

        Args:
            filepath: 检查点文件路径
            agent: 可选，传入智能体时自动加载其权重

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
                best_reward=checkpoint_data.get("best_reward", -float('inf')),
                early_stop_state=checkpoint_data.get("early_stop_state"),
                agent_save_path=checkpoint_data.get("agent_save_path")
            )

            if agent and checkpoint.agent_save_path:
                if os.path.exists(checkpoint.agent_save_path):
                    if hasattr(agent, 'load'):
                        agent.load(checkpoint.agent_save_path)
                        self._log(f"智能体权重已从检查点恢复: {checkpoint.agent_save_path}")

            if checkpoint.early_stop_state and self._early_stop_monitor:
                self._early_stop_monitor.load_state(checkpoint.early_stop_state)

            self._log(f"检查点已加载: {filepath} (回合={checkpoint.episode}, "
                      f"步数={checkpoint.total_steps})", logging.INFO)
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

    def get_best_model_path(self) -> Optional[str]:
        """获取最优模型保存路径"""
        return self._best_model_path

    def get_exception_summary(self) -> Optional[Dict[str, Any]]:
        """获取异常统计摘要"""
        if self._exception_guard:
            return self._exception_guard.get_summary()
        return None

    def get_early_stop_state(self) -> Optional[Dict[str, Any]]:
        """获取早停监控器状态"""
        if self._early_stop_monitor:
            return self._early_stop_monitor.get_state()
        return None


class RLTrainer(BaseTrainer):
    """
    强化学习训练器 - 增强版

    核心改进：
    1. 断点续训：中断后自动保存，重启后加载模型接续训练
    2. 自动保存最优模型：留存训练过程中效果最好的权重文件
    3. 简易控制台进度条：实时展示单回合与整体训练进度
    4. 运行异常捕获模块：报错自动记录日志且程序不会直接终止
    5. 早停判定机制：连续多轮收益无提升时自动终止无效训练
    6. 经验回放自动采样学习
    7. 评估与训练环境隔离
    8. epsilon 同步更新
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

        增强功能：
        - 断点续训：从检查点恢复训练状态和智能体权重
        - 自动保存最优模型：每次评估后若有改进则保存
        - 控制台进度条：单回合步骤进度 + 总体训练进度
        - 异常捕获：所有异常自动记录日志，不终止程序
        - 早停：连续无改进时自动终止

        Args:
            agent: 智能体实例
            environment: 训练环境
            eval_environment: 评估环境（可选）
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

        # === 断点续训：加载检查点恢复训练状态 ===
        if self.config.resume_from_checkpoint:
            checkpoint, extra = self.load_checkpoint(
                self.config.resume_from_checkpoint, agent=agent
            )
            if checkpoint:
                start_episode = checkpoint.episode + 1
                self.total_steps = checkpoint.total_steps
                self.stats = TrainingStats.from_dict(checkpoint.stats)
                self._best_eval_reward = checkpoint.best_reward
                epsilon = self.stats.epsilon_values[-1] if self.stats.epsilon_values else epsilon
                self._log(f"断点续训: 从回合 {start_episode} 继续 "
                          f"(已完成步数={self.total_steps}, 最佳奖励={self._best_eval_reward:.2f})")
        elif self.config.resume_agent_path:
            if hasattr(agent, 'load') and os.path.exists(self.config.resume_agent_path):
                agent.load(self.config.resume_agent_path)
                self._log(f"已加载预训练智能体权重: {self.config.resume_agent_path}")

        actual_eval_env = eval_environment
        if self.config.separate_eval_env and actual_eval_env is None:
            try:
                from src.sandbox.environment_manager import EnvironmentManager
                manager = EnvironmentManager()
                instance = manager.create_environment(
                    name="EvalEnv",
                    type="base",
                    config={}
                )
                actual_eval_env = manager.load_environment(instance)
                self._log("已创建独立的评估环境")
            except Exception as e:
                self._log_exception("创建评估环境失败，使用训练环境", e)
                actual_eval_env = None

        self._log(f"开始训练: 目标{self.config.max_episodes}回合, "
                  f"每回合最大{self.config.max_steps_per_episode}步, "
                  f"早停耐心={self.config.early_stop_patience}")

        # === 控制台进度条初始化 ===
        use_tqdm = self.config.show_progress and HAS_TQDM
        use_console_bar = self.config.show_progress and not HAS_TQDM

        if use_tqdm:
            self._progress_bar = tqdm(
                total=self.config.max_episodes,
                initial=start_episode,
                desc="训练进度",
                unit="ep"
            )
        elif use_console_bar:
            self._progress_bar = ConsoleProgressBar(
                total=self.config.max_episodes - start_episode,
                desc="训练进度"
            )

        # === 主训练循环 ===
        stop_reason = "completed"
        try:
            for episode in range(start_episode, self.config.max_episodes):
                if self._should_stop:
                    stop_reason = "interrupted"
                    self._log(f"训练在回合 {episode} 被中断", logging.WARNING)
                    break

                if self.config.max_time_seconds:
                    elapsed = self.get_elapsed_time()
                    if elapsed >= self.config.max_time_seconds:
                        stop_reason = "time_limit"
                        self._log(f"达到最大训练时间: {self.config.max_time_seconds}s")
                        break

                self.current_episode = episode

                # 重置环境
                try:
                    observation = self._safe_reset_environment(environment)
                except EnvironmentError as e:
                    self._log_exception(f"回合 {episode} 环境重置失败，跳过", e)
                    if self._exception_guard:
                        self._exception_guard.reset_consecutive()
                    continue

                if self._exception_guard:
                    self._exception_guard.reset_consecutive()

                episode_reward = 0.0
                episode_steps = 0
                episode_losses = []

                try:
                    self.episode_buffer.start_episode()
                except Exception as e:
                    self._log_exception("启动回合缓冲区失败", e)

                # === 单回合步骤循环 ===
                for step in range(self.config.max_steps_per_episode):
                    if self._should_stop:
                        break

                    try:
                        action = self._safe_agent_act(agent, observation)
                    except AgentError as e:
                        self._log_exception(f"回合{episode}步骤{step}决策失败", e)
                        break

                    if action is None:
                        break

                    tool_name = action.tool
                    parameters = action.parameters

                    try:
                        result, reward, done, info = self._safe_execute_action(
                            environment, tool_name, parameters
                        )
                    except EnvironmentError as e:
                        self._log_exception(f"回合{episode}步骤{step}执行失败", e)
                        reward = -10.0
                        done = True
                        info = {"error": str(e)}

                    try:
                        next_observation = self._safe_get_observation(environment)
                    except Exception as e:
                        self._log_exception("获取下一观察失败", e)
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

                # === 回合结束统计 ===
                self.stats.episode_rewards.append(episode_reward)
                self.stats.episode_lengths.append(episode_steps)
                self.stats.epsilon_values.append(epsilon)
                self.stats.timestamps.append(self.get_elapsed_time())
                self.stats.learning_rates.append(
                    agent.learning_rate if hasattr(agent, 'learning_rate')
                    else self.config.learning_rate
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

                # === 定期评估 + 自动保存最优模型 ===
                if (episode + 1) % self.config.eval_freq == 0 and actual_eval_env:
                    try:
                        self.stage = TrainingStage.EVALUATING
                        eval_stats = self.evaluate(agent, actual_eval_env)
                        self.stage = TrainingStage.TRAINING
                        self.stats.eval_rewards.append(eval_stats["avg_reward"])

                        if eval_stats["avg_reward"] > self._best_eval_reward:
                            self._best_eval_reward = eval_stats["avg_reward"]
                            self.stats.best_eval_reward = self._best_eval_reward
                            self.stats.episodes_without_improvement = 0

                            if self.config.auto_save_best:
                                self.save_best_model(agent, self._best_eval_reward)

                        self._log(
                            f"回合 {episode+1}: 评估奖励={eval_stats['avg_reward']:.2f}, "
                            f"最佳={self._best_eval_reward:.2f}, "
                            f"训练奖励={episode_reward:.2f}, ε={epsilon:.4f}"
                        )

                        # 早停监控器更新（基于评估奖励）
                        if self._early_stop_monitor and \
                                self.config.early_stop_metric == "eval_reward":
                            should_continue = self._early_stop_monitor.step(
                                eval_stats["avg_reward"]
                            )
                            if not should_continue:
                                stop_reason = "early_stop"
                                self._log(
                                    f"早停触发: {self._early_stop_monitor.reason}",
                                    logging.INFO
                                )
                                break

                    except Exception as e:
                        self._log_exception("评估阶段异常", e)
                        self.stage = TrainingStage.TRAINING

                # === 早停判定（基于训练奖励） ===
                if self._early_stop_monitor and \
                        self.config.early_stop_metric == "train_reward":
                    should_continue = self._early_stop_monitor.step(episode_reward)
                    if not should_continue:
                        stop_reason = "early_stop"
                        self._log(
                            f"早停触发: {self._early_stop_monitor.reason}",
                            logging.INFO
                        )
                        break
                elif not self._early_stop_monitor and self.config.early_stop_patience > 0:
                    if self.stats.episodes_without_improvement >= self.config.early_stop_patience:
                        stop_reason = "early_stop"
                        self._log(
                            f"早停触发: 连续 {self.config.early_stop_patience} 回合无改进",
                            logging.INFO
                        )
                        break

                # === 定期保存检查点 ===
                if (episode + 1) % self.config.save_freq == 0:
                    try:
                        self.stage = TrainingStage.SAVING
                        checkpoint_path = os.path.join(
                            self.config.log_dir,
                            f"checkpoint_episode_{episode + 1}.json"
                        )
                        self.save_checkpoint(agent, checkpoint_path)
                        self._last_save_episode = episode
                        self.stage = TrainingStage.TRAINING
                    except Exception as e:
                        self._log_exception("保存检查点失败", e)
                        self.stage = TrainingStage.TRAINING

                # === 周期性日志 ===
                if (episode + 1) % 10 == 0 or episode == start_episode:
                    recent_rewards = self.stats.episode_rewards[-10:]
                    recent_avg = sum(recent_rewards) / len(recent_rewards)
                    self._log(
                        f"回合 {episode+1}: 近10回合均奖={recent_avg:.2f}, "
                        f"ε={epsilon:.4f}, 缓冲区={len(self.replay_buffer)}"
                    )

                # === 更新进度条 ===
                if self._progress_bar:
                    postfix = {
                        'R': f'{episode_reward:.1f}',
                        'ε': f'{epsilon:.3f}',
                        'best': f'{self.stats.best_training_reward:.1f}'
                    }
                    if isinstance(self._progress_bar, ConsoleProgressBar):
                        step_info = f"步={episode_steps}"
                        self._progress_bar.update(1, postfix=postfix,
                                                  step_info=step_info)
                    else:
                        self._progress_bar.update(1)
                        self._progress_bar.set_postfix({
                            'reward': f'{episode_reward:.1f}',
                            'epsilon': f'{epsilon:.3f}',
                            'best': f'{self.stats.best_training_reward:.1f}',
                            'steps': episode_steps
                        })

        except KeyboardInterrupt:
            stop_reason = "keyboard_interrupt"
            self._log("收到 KeyboardInterrupt，正在安全退出...", logging.WARNING)
            if self.config.checkpoint_on_interrupt:
                try:
                    interrupt_checkpoint = os.path.join(
                        self.config.log_dir,
                        f"interrupt_checkpoint_ep{self.current_episode}.json"
                    )
                    self.save_checkpoint(agent, interrupt_checkpoint, {
                        "stop_reason": stop_reason,
                        "interrupted": True,
                        "current_episode": self.current_episode,
                        "total_steps": self.total_steps,
                        "elapsed_time": self.get_elapsed_time()
                    })
                    self._log(f"中断检查点已保存: {interrupt_checkpoint}", logging.INFO)
                except Exception as save_err:
                    self._log_exception("中断时保存检查点失败", save_err)
        except Exception as e:
            stop_reason = "unhandled_exception"
            self._log_exception("训练循环发生未处理异常", e)

        # === 训练结束收尾 ===
        self._is_running = False
        self.stage = TrainingStage.COMPLETED

        if self._progress_bar:
            if isinstance(self._progress_bar, ConsoleProgressBar):
                self._progress_bar.close()
            else:
                self._progress_bar.close()

        # 保存最终检查点（含智能体权重，支持后续断点续训）
        try:
            final_checkpoint = os.path.join(self.config.log_dir, "final_checkpoint.json")
            self.save_checkpoint(agent, final_checkpoint, {
                "completed": stop_reason == "completed",
                "stop_reason": stop_reason,
                "elapsed_time": self.get_elapsed_time()
            })
        except Exception as e:
            self._log_exception("保存最终检查点失败", e)

        # 输出训练总结
        summary = self.stats.get_summary()
        self._log(f"训练结束 (原因: {stop_reason})")
        self._log(f"  总回合: {self.current_episode + 1}, 总步数: {summary.get('total_steps', 0)}")
        self._log(f"  平均奖励: {summary.get('avg_reward', 0):.2f}, "
                  f"最佳训练奖励: {summary.get('best_training_reward', 0):.2f}")
        self._log(f"  最佳评估奖励: {summary.get('best_eval_reward', -float('inf')):.2f}")
        self._log(f"  耗时: {self.get_elapsed_time():.1f}s")
        if self._best_model_path:
            self._log(f"  最优模型路径: {self._best_model_path}")
        if self._exception_guard:
            err_summary = self._exception_guard.get_summary()
            if err_summary["total_errors"] > 0:
                self._log(f"  异常统计: 共{err_summary['total_errors']}次, "
                          f"类型={err_summary['error_types']}")
                self._log(f"  异常日志: {err_summary['crash_log_path']}")

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
                except Exception as e:
                    self._log_exception("评估中获取观察失败", e)
                
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
            except Exception as e:
                self.environment = None

    def can_progress(self) -> bool:
        """检查是否满足升级条件（基于当前累积的统计）"""
        if self.completed:
            return True
        if self.consecutive_successes >= self.min_consecutive_success:
            return True
        if self.recent_successes and len(self.recent_successes) >= 10:
            recent_success_rate = sum(self.recent_successes) / len(self.recent_successes)
            if recent_success_rate >= self.min_success_rate:
                return True
        if self.min_avg_reward is not None and self.recent_rewards:
            recent_avg = sum(self.recent_rewards) / len(self.recent_rewards)
            if recent_avg >= self.min_avg_reward:
                return True
        if self.episodes_trained >= self.max_episodes_per_level:
            return True
        return False

    def get_state(self) -> Dict[str, Any]:
        """序列化级别训练状态，用于断点续训"""
        return {
            "level_id": self.level_id,
            "episodes_trained": self.episodes_trained,
            "consecutive_successes": self.consecutive_successes,
            "recent_rewards": self.recent_rewards[-20:],
            "recent_successes": self.recent_successes[-20:],
            "completed": self.completed
        }

    def load_state(self, state: Dict[str, Any]):
        """从序列化数据恢复级别训练状态"""
        if state.get("level_id") != self.level_id:
            return
        self.episodes_trained = state.get("episodes_trained", 0)
        self.consecutive_successes = state.get("consecutive_successes", 0)
        self.recent_rewards = state.get("recent_rewards", [])
        self.recent_successes = state.get("recent_successes", [])
        self.completed = state.get("completed", False)


class CurriculumTrainer(RLTrainer):
    """
    课程学习训练器 - 改进版

    核心改进：
    1. 结构化的课程级别配置
    2. 灵活的升级条件（成功率、平均奖励、连续成功）
    3. 每个级别独立的环境
    4. 断点续训：中断后可加载检查点恢复课程进度和智能体权重
    5. 自动保存最优模型：训练过程中效果最好的权重文件自动留存
    6. 双进度条：总级别进度 + 单级别回合进度，实时展示性能指标
    7. 早停判定机制：级别内连续多轮无提升时自动终止无效训练
    8. 中断自动保存：KeyboardInterrupt 时保存课程状态检查点
    9. 完善的异常处理：所有异常均记录日志，程序不会直接终止
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

    def _build_curriculum_extra_info(self, stop_reason: str = "") -> Dict[str, Any]:
        """构建课程进度状态信息，用于检查点保存"""
        return {
            "stop_reason": stop_reason,
            "completed_levels": [l.level_id for l in self.curriculum_levels if l.completed],
            "curriculum_progress": {
                "current_level_index": self.current_level_index,
                "all_levels": [
                    {
                        "level_id": l.level_id,
                        "completed": l.completed,
                        "episodes_trained": l.episodes_trained,
                        "consecutive_successes": l.consecutive_successes,
                        "recent_rewards": l.recent_rewards[-20:],
                        "recent_successes": l.recent_successes[-20:]
                    }
                    for l in self.curriculum_levels
                ]
            }
        }

    def _restore_curriculum_progress(self, checkpoint, extra_info: Dict[str, Any]) -> Tuple[int, float]:
        """
        从检查点恢复课程进度状态

        Returns:
            (start_level_idx, epsilon)
        """
        start_level_idx = 0
        epsilon = self.config.epsilon

        self.stats = TrainingStats.from_dict(checkpoint.stats)
        self.total_steps = checkpoint.total_steps
        self._best_eval_reward = checkpoint.best_reward

        if self.stats.epsilon_values:
            epsilon = self.stats.epsilon_values[-1]

        curriculum_progress = extra_info.get("curriculum_progress", {})
        completed_levels = extra_info.get("completed_levels", [])

        if curriculum_progress:
            saved_levels = curriculum_progress.get("all_levels", [])
            for saved_level in saved_levels:
                level_id = saved_level.get("level_id")
                for level in self.curriculum_levels:
                    if level.level_id == level_id:
                        level.completed = saved_level.get("completed", False)
                        level.episodes_trained = saved_level.get("episodes_trained", 0)
                        level.consecutive_successes = saved_level.get("consecutive_successes", 0)
                        if saved_level.get("recent_rewards"):
                            level.recent_rewards = saved_level["recent_rewards"]
                        if saved_level.get("recent_successes"):
                            level.recent_successes = saved_level["recent_successes"]
                        break

            start_level_idx = curriculum_progress.get("current_level_index", 0)
            if start_level_idx < len(self.curriculum_levels) and self.curriculum_levels[start_level_idx].completed:
                start_level_idx += 1
        elif completed_levels:
            for level in self.curriculum_levels:
                if level.level_id in completed_levels:
                    level.completed = True
            for i, level in enumerate(self.curriculum_levels):
                if not level.completed:
                    start_level_idx = i
                    break
            else:
                start_level_idx = len(self.curriculum_levels)

        completed_count = sum(1 for l in self.curriculum_levels if l.completed)
        self._log(
            f"课程学习续训: 从第 {start_level_idx + 1} 级开始 "
            f"(已完成 {completed_count}/{len(self.curriculum_levels)} 级, "
            f"步数={self.total_steps}, 最佳奖励={self._best_eval_reward:.2f})"
        )
        return start_level_idx, epsilon

    def train(
        self,
        agent: Agent,
        initial_environment=None,
        eval_environment=None,
        callbacks: Dict[str, Callable] = None
    ) -> TrainingStats:
        """
        课程学习训练主循环

        增强功能：
        - 断点续训：从检查点恢复课程进度和智能体权重
        - 自动保存最优模型：级别完成时评估并保存最佳模型
        - 双进度条：总级别进度 + 单级别回合进度
        - 早停判定：级别内连续无提升时自动终止
        - 中断保存：KeyboardInterrupt 时自动保存课程状态

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
        start_level_idx = 0

        # ========== 断点续训：加载检查点恢复课程进度 ==========
        if self.config.resume_from_checkpoint:
            checkpoint, extra_info = self.load_checkpoint(
                self.config.resume_from_checkpoint, agent=agent
            )
            if checkpoint:
                start_level_idx, epsilon = self._restore_curriculum_progress(
                    checkpoint, extra_info or {}
                )
        elif self.config.resume_agent_path:
            if hasattr(agent, 'load') and os.path.exists(self.config.resume_agent_path):
                agent.load(self.config.resume_agent_path)
                self._log(f"已加载预训练智能体权重: {self.config.resume_agent_path}")

        # ========== 初始化早停监控器（级别内） ==========
        level_early_stop: Optional[EarlyStopMonitor] = None
        if self.config.early_stop_patience > 0:
            level_early_stop = EarlyStopMonitor(
                patience=self.config.early_stop_patience,
                min_improvement=self.config.early_stop_min_improvement,
                metric=self.config.early_stop_metric
            )

        # ========== 初始化双进度条 ==========
        use_tqdm = self.config.show_progress and HAS_TQDM
        use_console_bar = self.config.show_progress and not HAS_TQDM

        level_progress_bar = None
        episode_progress_bar = None

        remaining_levels = len(self.curriculum_levels) - start_level_idx
        if use_tqdm:
            level_progress_bar = tqdm(
                total=len(self.curriculum_levels),
                initial=start_level_idx,
                desc="课程进度",
                unit="级",
                position=0,
                leave=True
            )
        elif use_console_bar:
            level_progress_bar = ConsoleProgressBar(
                total=remaining_levels,
                desc="课程进度"
            )

        self._log(f"开始课程学习训练，共 {len(self.curriculum_levels)} 个级别")

        stop_reason = "completed"

        try:
            for level_idx in range(start_level_idx, len(self.curriculum_levels)):
                if self._should_stop:
                    stop_reason = "interrupted"
                    break

                self.current_level_index = level_idx
                level = self.curriculum_levels[level_idx]

                if not level.completed:
                    level.reset()

                # 重置级别内早停监控器
                if level_early_stop:
                    level_early_stop.reset()

                self._log(f"\n{'='*60}")
                self._log(f"进入级别 {level_idx + 1}/{len(self.curriculum_levels)}: {level.level_id}")
                self._log(f"  描述: {level.description}")
                self._log(f"  难度: {level.difficulty}")
                self._log(f"  升级条件:")
                self._log(f"    - 连续成功: {level.min_consecutive_success} 次")
                self._log(f"    - 成功率: {level.min_success_rate:.0%}")
                if level.min_avg_reward:
                    self._log(f"    - 平均奖励: {level.min_avg_reward}")
                if level_early_stop:
                    self._log(f"    - 早停耐心: {level_early_stop.patience} 回合")
                self._log(f"{'='*60}\n")

                # ========== 初始化单级别回合进度条 ==========
                if use_tqdm:
                    episode_progress_bar = tqdm(
                        total=level.max_episodes_per_level,
                        initial=level.episodes_trained,
                        desc=f"级别 {level_idx + 1}",
                        unit="回合",
                        position=1,
                        leave=False
                    )
                elif use_console_bar:
                    remaining_episodes = level.max_episodes_per_level - level.episodes_trained
                    episode_progress_bar = ConsoleProgressBar(
                        total=remaining_episodes,
                        desc=f"级别 {level_idx + 1}"
                    )

                try:
                    environment = level.get_environment()
                except EnvironmentError as e:
                    self._log_exception(f"无法创建级别 {level.level_id} 的环境，跳过", e)
                    if level_progress_bar:
                        if isinstance(level_progress_bar, ConsoleProgressBar):
                            level_progress_bar.update(1, step_info="跳过")
                        else:
                            level_progress_bar.update(1)
                    continue

                start_episode_in_level = level.episodes_trained

                for episode in range(start_episode_in_level, level.max_episodes_per_level):
                    if self._should_stop:
                        stop_reason = "interrupted"
                        break

                    self.current_episode = episode

                    try:
                        observation = self._safe_reset_environment(environment)
                    except EnvironmentError as e:
                        self._log_exception(f"回合 {episode} 环境重置失败", e)
                        if self._exception_guard:
                            self._exception_guard.reset_consecutive()
                        continue

                    if self._exception_guard:
                        self._exception_guard.reset_consecutive()

                    episode_reward = 0.0
                    episode_steps = 0
                    episode_success = False

                    for step in range(self.config.max_steps_per_episode):
                        if self._should_stop:
                            break

                        try:
                            action = self._safe_agent_act(agent, observation)
                        except AgentError as e:
                            self._log_exception("智能体决策失败", e)
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
                        except Exception as e:
                            self._log_exception("获取观察失败", e)
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
                        except Exception as e:
                            self._log_exception("添加经验到回放缓冲区失败", e)

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
                            except Exception as e:
                                self._log_exception("执行 on_step 回调失败", e)

                        observation = next_observation

                        if done:
                            episode_success = info.get('success', reward > 0)
                            break

                    # === 回合结束统计 ===
                    self.stats.episode_rewards.append(episode_reward)
                    self.stats.episode_lengths.append(episode_steps)
                    self.stats.epsilon_values.append(epsilon)
                    self.stats.timestamps.append(self.get_elapsed_time())

                    epsilon = max(self.config.epsilon_min, epsilon * self.config.epsilon_decay)
                    self._sync_agent_epsilon(agent, epsilon)

                    if episode_reward > self.stats.best_training_reward:
                        self.stats.best_training_reward = episode_reward

                    can_advance, advance_info = level.check_advancement(episode_reward, episode_success)

                    # ========== 早停判定 ==========
                    if level_early_stop:
                        should_continue = level_early_stop.step(episode_reward)
                        if not should_continue:
                            self._log(
                                f"级别 {level.level_id} 早停触发: {level_early_stop.reason}",
                                logging.INFO
                            )
                            break

                    # ========== 更新回合进度条 ==========
                    if episode_progress_bar:
                        postfix = {
                            '奖励': f'{episode_reward:.1f}',
                            '连胜': str(advance_info['consecutive_successes']),
                            '成功率': f"{advance_info['recent_success_rate']:.0%}"
                        }
                        if isinstance(episode_progress_bar, ConsoleProgressBar):
                            step_info = f"步骤={episode_steps}"
                            episode_progress_bar.update(1, postfix=postfix, step_info=step_info)
                        else:
                            episode_progress_bar.update(1)
                            episode_progress_bar.set_postfix(postfix, refresh=False)

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
                        self._log(f"\n级别 {level_idx + 1} 完成!")
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
                            except Exception as e:
                                self._log_exception("执行 on_level_up 回调失败", e)

                        # ========== 级别完成时保存检查点（含课程进度） ==========
                        try:
                            self.stage = TrainingStage.SAVING
                            checkpoint_path = os.path.join(
                                self.config.log_dir,
                                f"level_{level_idx + 1}_complete_checkpoint.json"
                            )
                            self.save_checkpoint(agent, checkpoint_path,
                                                 self._build_curriculum_extra_info("level_completed"))
                            self.stage = TrainingStage.TRAINING
                        except Exception as e:
                            self._log_exception("保存级别完成检查点失败", e)
                            self.stage = TrainingStage.TRAINING

                        # ========== 自动保存最优模型 ==========
                        if self.config.auto_save_best and eval_environment:
                            try:
                                self.stage = TrainingStage.EVALUATING
                                eval_stats = self.evaluate(agent, eval_environment)
                                self.stage = TrainingStage.TRAINING
                                self.stats.eval_rewards.append(eval_stats["avg_reward"])

                                if eval_stats["avg_reward"] > self._best_eval_reward:
                                    self._best_eval_reward = eval_stats["avg_reward"]
                                    self.stats.best_eval_reward = self._best_eval_reward
                                    self.save_best_model(agent, self._best_eval_reward)

                                self._log(
                                    f"级别 {level_idx + 1} 完成评估: "
                                    f"评估奖励={eval_stats['avg_reward']:.2f}, "
                                    f"最佳={self._best_eval_reward:.2f}"
                                )
                            except Exception as e:
                                self._log_exception("级别完成时评估失败", e)
                                self.stage = TrainingStage.TRAINING

                        break

                # ========== 关闭单级别进度条并更新总进度 ==========
                if episode_progress_bar:
                    if isinstance(episode_progress_bar, ConsoleProgressBar):
                        episode_progress_bar.close()
                    else:
                        episode_progress_bar.close()
                    episode_progress_bar = None

                if level_progress_bar:
                    if level.completed:
                        status_info = f"完成 {level.level_id}"
                    else:
                        status_info = f"未完成 {level.level_id}"
                    if isinstance(level_progress_bar, ConsoleProgressBar):
                        level_progress_bar.update(1, step_info=status_info)
                    else:
                        level_progress_bar.update(1)

        except KeyboardInterrupt:
            stop_reason = "keyboard_interrupt"
            self._log("收到 KeyboardInterrupt，正在保存课程状态...", logging.WARNING)
            # ========== 中断时自动保存课程状态检查点 ==========
            if self.config.checkpoint_on_interrupt:
                try:
                    interrupt_checkpoint = os.path.join(
                        self.config.log_dir,
                        f"curriculum_interrupt_ep{self.current_episode}.json"
                    )
                    self.save_checkpoint(agent, interrupt_checkpoint,
                                         self._build_curriculum_extra_info(stop_reason))
                    self._log(f"中断检查点已保存: {interrupt_checkpoint}", logging.INFO)
                except Exception as e:
                    self._log_exception("中断时保存检查点失败", e)
        except Exception as e:
            stop_reason = "unhandled_exception"
            self._log_exception("课程学习训练循环发生未处理异常", e)

        # ========== 训练结束收尾 ==========
        self._is_running = False
        self.stage = TrainingStage.COMPLETED

        if level_progress_bar:
            if isinstance(level_progress_bar, ConsoleProgressBar):
                level_progress_bar.close()
            else:
                level_progress_bar.close()

        all_completed = all(level.completed for level in self.curriculum_levels)

        self._log(f"\n{'='*60}")
        self._log("课程学习训练结束!")
        self._log(f"  通过级别数: {sum(1 for l in self.curriculum_levels if l.completed)} / {len(self.curriculum_levels)}")
        self._log(f"  总回合数: {len(self.stats.episode_rewards)}")
        self._log(f"  总步数: {self.total_steps}")
        self._log(f"  最佳训练奖励: {self.stats.best_training_reward:.2f}")
        self._log(f"  最佳评估奖励: {self.stats.best_eval_reward:.2f}")
        self._log(f"  耗时: {self.get_elapsed_time():.1f} 秒")
        if self._best_model_path:
            self._log(f"  最优模型路径: {self._best_model_path}")
        if self._exception_guard:
            err_summary = self._exception_guard.get_summary()
            if err_summary["total_errors"] > 0:
                self._log(f"  异常统计: 共{err_summary['total_errors']}次, "
                          f"类型={err_summary['error_types']}")
        self._log(f"{'='*60}")

        try:
            self.stage = TrainingStage.SAVING
            final_path = os.path.join(self.config.log_dir, "curriculum_final_checkpoint.json")
            self.save_checkpoint(agent, final_path,
                                 self._build_curriculum_extra_info(stop_reason))
            self.stage = TrainingStage.COMPLETED
        except Exception as e:
            self._log_exception("保存最终检查点失败", e)
            self.stage = TrainingStage.COMPLETED

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
