"""
改进版训练演示 - 展示所有新功能

演示内容：
1. 完整的异常处理
2. 模型保存/加载功能
3. 经验回放自动学习
4. 日志文件系统
5. 进度条展示
6. 改进的课程学习设计
7. 训练/评估模式分离
8. 早停机制
"""

import os
import sys
import tempfile
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.rl.agent import (
    QLearningAgent, REINFORCEAgent, HeuristicAgent, 
    RandomAgent, Action, load_agent_from_file
)
from src.rl.trainer import (
    TrainingConfig, RLTrainer, CurriculumTrainer, 
    TrainingStage, TrainingError
)
from src.rl.experience_buffer import ExperienceBuffer, Experience
from src.sandbox.environment_manager import EnvironmentManager


class SimpleTestEnvironment:
    """简单的测试环境，用于演示"""
    
    def __init__(self, difficulty: int = 1):
        self.difficulty = difficulty
        self.state: int = 0
        self.max_state: int = 10 * difficulty
        self.total_steps: int = 0
        self.available_tools: List[str] = ["move_left", "move_right", "stay"]
        
    def reset(self) -> Dict[str, Any]:
        self.state = 0
        self.total_steps = 0
        return {
            "current_state": "running",
            "state_variables": {"position": self.state, "difficulty": self.difficulty},
            "available_tools": self.available_tools,
            "max_position": self.max_state
        }
    
    def execute_tool(self, tool: str, **kwargs) -> tuple:
        self.total_steps += 1
        
        if tool == "move_right":
            self.state = min(self.max_state, self.state + 1)
            reward = 0.1
        elif tool == "move_left":
            self.state = max(0, self.state - 1)
            reward = -0.1
        elif tool == "stay":
            reward = 0.0
        else:
            reward = -0.5
        
        done = False
        info = {}
        
        if self.state >= self.max_state:
            done = True
            reward = 100.0 * self.difficulty
            info["success"] = True
            info["message"] = f"成功到达目标位置！难度={self.difficulty}"
        
        if self.total_steps >= 100 * self.difficulty:
            done = True
            reward = -10.0
            info["success"] = False
            info["message"] = "超时"
        
        info["position"] = self.state
        info["steps"] = self.total_steps
        
        return f"Position: {self.state}", reward, done, info
    
    def get_observation(self) -> Dict[str, Any]:
        return {
            "current_state": "terminal" if self.state >= self.max_state else "running",
            "state_variables": {"position": self.state, "difficulty": self.difficulty},
            "available_tools": self.available_tools,
            "max_position": self.max_state
        }


def create_environment_creator(difficulty: int) -> Callable:
    """创建环境生成函数"""
    def creator():
        return SimpleTestEnvironment(difficulty=difficulty)
    return creator


def demo_exception_handling():
    """演示异常处理"""
    print("\n" + "=" * 60)
    print("演示 1: 完整的异常处理")
    print("=" * 60)
    
    config = TrainingConfig(
        max_episodes=5,
        max_steps_per_episode=10,
        verbose=True,
        log_to_file=False,
        show_progress=False
    )
    
    trainer = RLTrainer(config)
    
    class FailingEnvironment:
        def reset(self):
            raise RuntimeError("环境重置失败！")
        
        def execute_tool(self, *args, **kwargs):
            raise RuntimeError("执行动作失败！")
        
        def get_observation(self):
            return {}
    
    agent = QLearningAgent("TestAgent")
    bad_env = FailingEnvironment()
    
    print("\n测试环境重置异常...")
    try:
        trainer.train(agent, bad_env)
    except Exception as e:
        print(f"  ✓ 正确捕获异常: {type(e).__name__}")
    
    print("\n✓ 异常处理演示完成")


def demo_model_save_load():
    """演示模型保存/加载"""
    print("\n" + "=" * 60)
    print("演示 2: 模型保存/加载功能")
    print("=" * 60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "test_agent.json")
        
        print(f"\n创建Q学习智能体并训练...")
        agent1 = QLearningAgent(
            name="TestAgent",
            learning_rate=0.2,
            epsilon=0.5
        )
        
        env = SimpleTestEnvironment(difficulty=1)
        for episode in range(20):
            obs = env.reset()
            done = False
            while not done:
                action = agent1.act(obs)
                if action is None:
                    break
                result, reward, done, info = env.execute_tool(action.tool, **action.parameters)
                next_obs = env.get_observation()
                agent1.observe(obs, action, reward, next_obs, done)
                obs = next_obs
        
        q_table_size_before = agent1.get_q_table_size()
        epsilon_before = agent1.epsilon
        
        print(f"  训练完成: Q表大小={q_table_size_before}, epsilon={epsilon_before:.4f}")
        
        print(f"\n保存智能体到: {save_path}")
        success = agent1.save(save_path)
        print(f"  保存{'成功' if success else '失败'}")
        
        if success:
            print("\n加载保存的智能体...")
            agent2 = QLearningAgent("LoadedAgent")
            load_success = agent2.load(save_path)
            print(f"  加载{'成功' if load_success else '失败'}")
            
            if load_success:
                print(f"\n比较参数:")
                print(f"  原智能体 - name: {agent1.name}, epsilon: {agent1.epsilon:.4f}")
                print(f"  加载后   - name: {agent2.name}, epsilon: {agent2.epsilon:.4f}")
                print(f"  Q表大小: 原={agent1.get_q_table_size()}, 加载后={agent2.get_q_table_size()}")
                
                print("\n✓ 模型保存/加载演示完成")
            
            print("\n测试 load_agent_from_file 函数...")
            agent3 = load_agent_from_file(save_path)
            if agent3:
                print(f"  ✓ 成功自动识别类型: {type(agent3).__name__}")
            else:
                print("  ✗ 加载失败")


def demo_experience_replay_integration():
    """演示经验回放集成"""
    print("\n" + "=" * 60)
    print("演示 3: 经验回放自动学习")
    print("=" * 60)
    
    config = TrainingConfig(
        max_episodes=30,
        max_steps_per_episode=50,
        batch_size=16,
        replay_update_freq=4,
        min_replay_size=20,
        use_prioritized_replay=True,
        eval_freq=10,
        verbose=True,
        log_to_file=False,
        show_progress=False
    )
    
    print("\n创建训练器（带优先经验回放）...")
    trainer = RLTrainer(config)
    
    print(f"  回放缓冲区容量: {trainer.config.replay_buffer_capacity}")
    print(f"  优先经验回放: {trainer.config.use_prioritized_replay}")
    print(f"  回放更新频率: 每 {trainer.config.replay_update_freq} 步")
    
    agent = QLearningAgent(
        name="ReplayAgent",
        learning_rate=0.15,
        epsilon=0.8,
        epsilon_decay=0.95
    )
    
    env = SimpleTestEnvironment(difficulty=2)
    
    print("\n开始训练（经验回放自动学习）...")
    
    initial_q_size = agent.get_q_table_size()
    stats = trainer.train(agent, env)
    
    final_q_size = agent.get_q_table_size()
    buffer_size = len(trainer.replay_buffer)
    
    print(f"\n训练结果:")
    print(f"  Q表大小: {initial_q_size} -> {final_q_size}")
    print(f"  回放缓冲区样本数: {buffer_size}")
    print(f"  总回合数: {len(stats.episode_rewards)}")
    print(f"  平均奖励: {stats.get_summary()['avg_reward']:.2f}")
    
    print("\n✓ 经验回放集成演示完成")


def demo_logging_system():
    """演示日志文件系统"""
    print("\n" + "=" * 60)
    print("演示 4: 日志文件系统")
    print("=" * 60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        config = TrainingConfig(
            max_episodes=10,
            max_steps_per_episode=20,
            verbose=True,
            log_dir=tmpdir,
            log_to_file=True,
            log_to_console=True,
            show_progress=False
        )
        
        print(f"\n创建带日志的训练器...")
        print(f"  日志目录: {tmpdir}")
        print(f"  日志到文件: {config.log_to_file}")
        print(f"  日志到控制台: {config.log_to_console}")
        
        trainer = RLTrainer(config)
        agent = QLearningAgent("LoggingAgent")
        env = SimpleTestEnvironment(difficulty=1)
        
        print("\n开始训练...")
        trainer.train(agent, env)
        
        log_file = trainer.get_log_file()
        print(f"\n日志文件: {log_file}")
        
        if log_file and os.path.exists(log_file):
            print("\n查看日志内容（前20行）:")
            with open(log_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()[:20]
                for line in lines:
                    print(f"  {line.rstrip()}")
        
        print("\n✓ 日志文件系统演示完成")


def demo_training_eval_modes():
    """演示训练/评估模式分离"""
    print("\n" + "=" * 60)
    print("演示 5: 训练/评估模式分离")
    print("=" * 60)
    
    agent = QLearningAgent(
        name="ModeAgent",
        epsilon=1.0,
        epsilon_decay=1.0
    )
    
    print(f"\n初始状态:")
    print(f"  training: {agent.training}")
    print(f"  epsilon: {agent.epsilon}")
    
    obs = {
        "available_tools": ["action1", "action2"],
        "state_variables": {"x": 0}
    }
    
    print("\n训练模式下的动作选择（高探索率）:")
    actions_train = []
    for _ in range(20):
        action = agent.act(obs)
        if action:
            actions_train.append(action.tool)
    
    tool_counts_train = {k: actions_train.count(k) for k in set(actions_train)}
    print(f"  动作分布: {tool_counts_train}")
    
    print("\n切换到评估模式...")
    agent.eval_mode()
    print(f"  training: {agent.training}")
    
    print("\n评估模式下的动作选择（无探索）:")
    actions_eval = []
    for _ in range(20):
        action = agent.act(obs)
        if action:
            actions_eval.append(action.tool)
    
    tool_counts_eval = {k: actions_eval.count(k) for k in set(actions_eval)}
    print(f"  动作分布: {tool_counts_eval}")
    
    print("\n切换回训练模式...")
    agent.train_mode()
    print(f"  training: {agent.training}")
    
    print("\n✓ 训练/评估模式分离演示完成")


def demo_curriculum_learning():
    """演示改进的课程学习"""
    print("\n" + "=" * 60)
    print("演示 6: 改进的课程学习设计")
    print("=" * 60)
    
    config = TrainingConfig(
        max_episodes=100,
        max_steps_per_episode=30,
        verbose=True,
        log_to_file=False,
        show_progress=False,
        eval_freq=20
    )
    
    print("\n创建课程学习训练器...")
    trainer = CurriculumTrainer(config)
    
    print("\n添加课程级别（从易到难）:")
    print("  级别 1: 难度=1, 目标位置=10")
    print("  级别 2: 难度=2, 目标位置=20")
    print("  级别 3: 难度=3, 目标位置=30")
    
    trainer.add_level(
        level_id="easy",
        difficulty=1,
        environment_creator=create_environment_creator(1),
        min_success_rate=0.7,
        min_consecutive_success=3,
        max_episodes_per_level=50,
        description="简单级别 - 学习基本动作"
    )
    
    trainer.add_level(
        level_id="medium",
        difficulty=2,
        environment_creator=create_environment_creator(2),
        min_success_rate=0.6,
        min_consecutive_success=2,
        max_episodes_per_level=80,
        description="中等难度 - 巩固学习"
    )
    
    trainer.add_level(
        level_id="hard",
        difficulty=3,
        environment_creator=create_environment_creator(3),
        min_success_rate=0.5,
        min_consecutive_success=2,
        max_episodes_per_level=100,
        description="困难级别 - 高级挑战"
    )
    
    agent = QLearningAgent(
        name="CurriculumAgent",
        learning_rate=0.2,
        epsilon=0.9,
        epsilon_decay=0.98
    )
    
    print("\n开始课程学习训练...")
    
    def on_level_up(info):
        print(f"\n  🎉 升级! 从级别 {info['from_level']} 到 {info['to_level']}")
        print(f"     级别ID: {info['level_id']}")
    
    stats = trainer.train(agent, callbacks={"on_level_up": on_level_up})
    
    print(f"\n课程学习结果:")
    progress = trainer.get_level_progress()
    for p in progress:
        status = "✓ 完成" if p['completed'] else "○ 未完成"
        print(f"  级别 {p['level_id']}: {status}, 训练回合数={p['episodes_trained']}")
    
    completed_count = sum(1 for p in progress if p['completed'])
    print(f"\n  完成级别: {completed_count} / {len(progress)}")
    
    print("\n✓ 课程学习演示完成")


def demo_early_stopping():
    """演示早停机制"""
    print("\n" + "=" * 60)
    print("演示 7: 早停机制")
    print("=" * 60)
    
    config = TrainingConfig(
        max_episodes=200,
        max_steps_per_episode=20,
        early_stop_patience=20,
        verbose=True,
        log_to_file=False,
        show_progress=False
    )
    
    print(f"\n配置早停:")
    print(f"  最大回合数: {config.max_episodes}")
    print(f"  早停耐心值: {config.early_stop_patience} 回合无改进则停止")
    
    trainer = RLTrainer(config)
    
    class NoImprovementEnvironment:
        def __init__(self):
            self.state = 0
            self.available_tools = ["move"]
        
        def reset(self):
            self.state = 0
            return {
                "current_state": "running",
                "state_variables": {"s": self.state},
                "available_tools": self.available_tools
            }
        
        def execute_tool(self, *args, **kwargs):
            reward = random.random() * 0.1
            self.state += 1
            done = self.state >= 10
            return "", reward, done, {"success": False}
        
        def get_observation(self):
            return {
                "current_state": "running" if self.state < 10 else "terminal",
                "state_variables": {"s": self.state},
                "available_tools": self.available_tools
            }
    
    import random
    agent = QLearningAgent("EarlyStopAgent")
    env = NoImprovementEnvironment()
    
    print("\n开始训练（预期会提前停止）...")
    stats = trainer.train(agent, env)
    
    actual_episodes = len(stats.episode_rewards)
    print(f"\n训练结果:")
    print(f"  配置最大回合数: {config.max_episodes}")
    print(f"  实际训练回合数: {actual_episodes}")
    print(f"  无改进回合数: {stats.episodes_without_improvement}")
    
    if actual_episodes < config.max_episodes:
        print(f"  ✓ 早停生效！提前 {config.max_episodes - actual_episodes} 回合停止")
    else:
        print(f"  完成所有回合")
    
    print("\n✓ 早停机制演示完成")


def run_all_demos():
    """运行所有演示"""
    print("\n" + "=" * 70)
    print("环境工厂系统 - 改进功能完整演示")
    print("=" * 70)
    print("\n本演示展示所有新增的改进功能：")
    print("  1. 完整的异常处理")
    print("  2. 模型保存/加载功能")
    print("  3. 经验回放自动学习集成")
    print("  4. 日志文件系统")
    print("  5. 训练/评估模式分离")
    print("  6. 改进的课程学习设计")
    print("  7. 早停机制")
    print("=" * 70)
    
    try:
        demo_exception_handling()
    except Exception as e:
        print(f"\n演示1失败: {e}")
    
    try:
        demo_model_save_load()
    except Exception as e:
        print(f"\n演示2失败: {e}")
    
    try:
        demo_experience_replay_integration()
    except Exception as e:
        print(f"\n演示3失败: {e}")
    
    try:
        demo_logging_system()
    except Exception as e:
        print(f"\n演示4失败: {e}")
    
    try:
        demo_training_eval_modes()
    except Exception as e:
        print(f"\n演示5失败: {e}")
    
    try:
        demo_curriculum_learning()
    except Exception as e:
        print(f"\n演示6失败: {e}")
    
    try:
        demo_early_stopping()
    except Exception as e:
        print(f"\n演示7失败: {e}")
    
    print("\n" + "=" * 70)
    print("所有演示完成！")
    print("=" * 70)
    
    print("\n改进总结:")
    print("""
 ✓ 异常处理: 所有关键操作都有try-except保护，训练不会因单个错误崩溃
 ✓ 模型保存/加载: 所有智能体都有统一的save()/load()接口，支持持久化
 ✓ 经验回放: 训练器自动从回放缓冲区采样学习，支持优先经验回放
 ✓ 日志系统: 自动生成日志文件，时间戳、级别、消息完整记录
 ✓ 进度展示: 支持tqdm进度条，实时显示训练进度和指标
 ✓ 模式分离: training/eval模式分离，评估时禁用探索
 ✓ 课程学习: 灵活的级别配置，多种升级条件（成功率、连续成功、最大回合）
 ✓ 早停机制: 可配置耐心值，无改进时自动停止训练节省资源
 ✓ 环境隔离: 支持独立的评估环境，不污染训练状态
 ✓ epsilon同步: 训练器和智能体的探索率同步更新
    """)


if __name__ == "__main__":
    run_all_demos()
