"""
改进版训练演示 - 展示所有新功能

演示内容：
1. 断点续训：中断后可加载模型接续训练
2. 自动保存最优模型：留存训练过程中效果最好的权重文件
3. 简易控制台进度条：实时展示单回合与整体训练进度
4. 运行异常捕获模块：报错自动记录日志且程序不会直接终止
5. 早停判定机制：连续多轮收益无提升时自动终止无效训练
6. 经验回放自动学习集成
7. 训练/评估模式分离
8. 改进的课程学习设计
"""

import os
import sys
import random
import tempfile
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.rl.agent import (
    QLearningAgent, REINFORCEAgent, HeuristicAgent,
    RandomAgent, Action, load_agent_from_file
)
from src.rl.trainer import (
    TrainingConfig, RLTrainer, CurriculumTrainer,
    TrainingStage, TrainingError, EarlyStopMonitor,
    ExceptionGuard, ConsoleProgressBar
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


def demo_checkpoint_resume():
    """演示断点续训功能"""
    print("\n" + "=" * 60)
    print("演示 1: 断点续训 - 中断后可加载模型接续训练")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        # 第一阶段：训练一段后保存
        print("\n[第一阶段] 训练 30 回合后保存检查点...")
        config1 = TrainingConfig(
            max_episodes=30,
            max_steps_per_episode=50,
            save_freq=30,
            verbose=True,
            log_dir=tmpdir,
            log_to_file=True,
            log_to_console=False,
            show_progress=False,
            early_stop_patience=0,
            separate_eval_env=False
        )

        trainer1 = RLTrainer(config1)
        agent1 = QLearningAgent(
            name="ResumeTestAgent",
            learning_rate=0.2,
            epsilon=0.8,
            epsilon_decay=0.95
        )
        env = SimpleTestEnvironment(difficulty=1)

        stats1 = trainer1.train(agent1, env)

        print(f"  第一阶段完成: {len(stats1.episode_rewards)} 回合")
        print(f"  最终 epsilon: {agent1.epsilon:.4f}")
        print(f"  Q表大小: {agent1.get_q_table_size()}")
        print(f"  最佳训练奖励: {stats1.best_training_reward:.2f}")

        # 找到保存的检查点
        checkpoint_path = os.path.join(tmpdir, "final_checkpoint.json")
        assert os.path.exists(checkpoint_path), "检查点文件应当存在"

        # 第二阶段：从检查点恢复，继续训练
        print("\n[第二阶段] 从检查点恢复，继续训练 30 回合...")
        config2 = TrainingConfig(
            max_episodes=60,
            max_steps_per_episode=50,
            save_freq=100,
            verbose=True,
            log_dir=tmpdir,
            log_to_file=True,
            log_to_console=False,
            show_progress=False,
            early_stop_patience=0,
            resume_from_checkpoint=checkpoint_path,
            separate_eval_env=False
        )

        trainer2 = RLTrainer(config2)
        agent2 = QLearningAgent(name="ResumeTestAgent")
        env2 = SimpleTestEnvironment(difficulty=1)

        stats2 = trainer2.train(agent2, env2)

        print(f"  第二阶段完成: 总计 {len(stats2.episode_rewards)} 回合")
        print(f"  恢复后 epsilon: {agent2.epsilon:.4f}")
        print(f"  恢复后 Q表大小: {agent2.get_q_table_size()}")
        print(f"  最佳训练奖励: {stats2.best_training_reward:.2f}")

        # 验证续训有效
        assert len(stats2.episode_rewards) > len(stats1.episode_rewards), \
            "续训后总回合数应当增加"
        assert agent2.get_q_table_size() >= agent1.get_q_table_size(), \
            "续训后 Q表不应缩小"

        print("\n  [验证通过] 断点续训成功，训练状态和模型权重均已恢复")
    print("\n  ✓ 断点续训演示完成")


def demo_auto_save_best_model():
    """演示自动保存最优模型"""
    print("\n" + "=" * 60)
    print("演示 2: 自动保存最优模型 - 留存最佳权重文件")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        best_dir = os.path.join(tmpdir, "best_models")

        config = TrainingConfig(
            max_episodes=50,
            max_steps_per_episode=50,
            eval_freq=10,
            eval_episodes=5,
            verbose=True,
            log_dir=tmpdir,
            log_to_file=True,
            log_to_console=False,
            show_progress=False,
            early_stop_patience=0,
            auto_save_best=True,
            best_model_dir=best_dir,
            separate_eval_env=False
        )

        trainer = RLTrainer(config)
        agent = QLearningAgent(
            name="BestModelAgent",
            learning_rate=0.2,
            epsilon=0.7,
            epsilon_decay=0.95
        )
        env = SimpleTestEnvironment(difficulty=1)
        eval_env = SimpleTestEnvironment(difficulty=1)

        print("\n  开始训练（每10回合评估一次，自动保存最优模型）...")
        stats = trainer.train(agent, env, eval_environment=eval_env)

        # 检查最优模型是否保存
        best_agent_path = os.path.join(best_dir, "best_model_agent.json")
        best_meta_path = os.path.join(best_dir, "best_model_meta.json")

        if os.path.exists(best_agent_path):
            print(f"\n  最优模型已保存:")
            print(f"    智能体权重: {best_agent_path}")

            with open(best_meta_path, 'r') as f:
                meta = json.load(f)
            print(f"    最优指标值: {meta['metric_value']:.2f}")
            print(f"    保存时回合: {meta['episode']}")
            print(f"    保存时间: {meta['timestamp']}")

            # 验证可以加载最优模型
            loaded_agent = QLearningAgent("LoadedBest")
            loaded_agent.load(best_agent_path)
            print(f"    加载验证: Q表大小={loaded_agent.get_q_table_size()}")
        else:
            print("\n  (训练中未触发评估改进，无最优模型保存)")

        print(f"\n  训练器记录的最优模型路径: {trainer.get_best_model_path()}")
    print("\n  ✓ 自动保存最优模型演示完成")


def demo_console_progress_bar():
    """演示简易控制台进度条"""
    print("\n" + "=" * 60)
    print("演示 3: 简易控制台进度条 - 实时展示训练进度")
    print("=" * 60)

    print("\n  [内置进度条展示] 不依赖 tqdm 的控制台进度条:")
    print()

    # 直接展示 ConsoleProgressBar
    bar = ConsoleProgressBar(total=20, desc="  训练")
    import time
    for i in range(20):
        time.sleep(0.05)
        bar.update(1, postfix={'R': f'{random.uniform(-5, 10):.1f}',
                               'eps': f'{1.0 - i*0.04:.2f}'},
                   step_info=f"步={random.randint(5, 30)}")
    bar.close()

    print("\n  [集成训练进度条展示]:")
    print()

    config = TrainingConfig(
        max_episodes=15,
        max_steps_per_episode=30,
        verbose=False,
        log_to_file=False,
        log_to_console=False,
        show_progress=True,
        early_stop_patience=0,
        separate_eval_env=False
    )

    trainer = RLTrainer(config)
    agent = QLearningAgent("ProgressAgent", epsilon=0.5)
    env = SimpleTestEnvironment(difficulty=1)
    trainer.train(agent, env)

    print("\n  ✓ 控制台进度条演示完成")


def demo_exception_guard():
    """演示运行异常捕获模块"""
    print("\n" + "=" * 60)
    print("演示 4: 运行异常捕获模块 - 报错自动记录不终止程序")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        config = TrainingConfig(
            max_episodes=20,
            max_steps_per_episode=10,
            verbose=True,
            log_dir=tmpdir,
            log_to_file=True,
            log_to_console=False,
            show_progress=False,
            early_stop_patience=0,
            separate_eval_env=False
        )

        trainer = RLTrainer(config)

        # 创建一个间歇性失败的环境
        class FlakyEnvironment:
            def __init__(self):
                self.state = 0
                self.call_count = 0

            def reset(self):
                self.call_count += 1
                if self.call_count % 5 == 0:
                    raise RuntimeError("模拟环境重置异常！")
                self.state = 0
                return {
                    "current_state": "running",
                    "state_variables": {"s": self.state},
                    "available_tools": ["action1", "action2"]
                }

            def execute_tool(self, tool, **kwargs):
                self.state += 1
                if random.random() < 0.1:
                    raise ValueError("模拟执行异常！")
                done = self.state >= 8
                return "", 1.0, done, {"success": done}

            def get_observation(self):
                return {
                    "current_state": "running",
                    "state_variables": {"s": self.state},
                    "available_tools": ["action1", "action2"]
                }

        agent = QLearningAgent("FlakyAgent")
        flaky_env = FlakyEnvironment()

        print("\n  使用间歇性异常环境进行训练（部分回合会报错）...")
        stats = trainer.train(agent, flaky_env)

        print(f"\n  训练结果（程序未终止）:")
        print(f"    完成回合数: {len(stats.episode_rewards)}")
        print(f"    平均奖励: {stats.get_summary().get('avg_reward', 0):.2f}")

        # 查看异常统计
        err_summary = trainer.get_exception_summary()
        print(f"\n  异常统计:")
        print(f"    总异常次数: {err_summary['total_errors']}")
        print(f"    异常类型: {err_summary['error_types']}")
        print(f"    异常日志文件: {err_summary['crash_log_path']}")

        if err_summary['crash_log_path'] and os.path.exists(err_summary['crash_log_path']):
            with open(err_summary['crash_log_path'], 'r') as f:
                lines = f.readlines()
            print(f"    日志条目数: {len(lines)}")
            if lines:
                first_error = json.loads(lines[0])
                print(f"    首条异常: [{first_error['error_type']}] {first_error['message']}")

    print("\n  ✓ 异常捕获模块演示完成")


def demo_early_stop_mechanism():
    """演示早停判定机制"""
    print("\n" + "=" * 60)
    print("演示 5: 早停判定机制 - 连续多轮无提升自动终止")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        # 使用 EarlyStopMonitor 独立演示
        print("\n  [独立早停监控器演示]:")
        monitor = EarlyStopMonitor(patience=5, min_improvement=0.5, metric="train_reward")

        rewards = [1.0, 2.0, 3.0, 3.5, 3.6, 3.65, 3.66, 3.67, 3.68, 3.69, 3.70]
        for i, r in enumerate(rewards):
            should_continue = monitor.step(r)
            status = "继续" if should_continue else "停止"
            print(f"    回合{i+1}: 奖励={r:.2f}, 最佳={monitor.best_value:.2f}, "
                  f"无改进轮数={monitor.rounds_without_improvement}, {status}")
            if not should_continue:
                print(f"    -> 早停原因: {monitor.reason}")
                break

        # 集成到训练器中演示
        print("\n  [集成训练早停演示]:")
        config = TrainingConfig(
            max_episodes=200,
            max_steps_per_episode=20,
            early_stop_patience=15,
            early_stop_min_improvement=0.01,
            early_stop_metric="train_reward",
            verbose=True,
            log_dir=tmpdir,
            log_to_file=False,
            log_to_console=False,
            show_progress=False,
            separate_eval_env=False
        )

        trainer = RLTrainer(config)

        class PlateauEnvironment:
            """奖励会在一定回合后趋于平坦的环境"""
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
                self.state += 1
                reward = random.uniform(0.4, 0.6)
                done = self.state >= 10
                return "", reward, done, {"success": False}

            def get_observation(self):
                return {
                    "current_state": "running" if self.state < 10 else "terminal",
                    "state_variables": {"s": self.state},
                    "available_tools": self.available_tools
                }

        agent = QLearningAgent("EarlyStopAgent")
        env = PlateauEnvironment()

        stats = trainer.train(agent, env)

        actual_episodes = len(stats.episode_rewards)
        print(f"\n  训练结果:")
        print(f"    配置最大回合: {config.max_episodes}")
        print(f"    实际训练回合: {actual_episodes}")
        print(f"    早停状态: {trainer.get_early_stop_state()}")

        if actual_episodes < config.max_episodes:
            print(f"    ✓ 早停生效！提前 {config.max_episodes - actual_episodes} 回合终止")
        else:
            print(f"    完成所有回合（环境一直有改进）")

    print("\n  ✓ 早停判定机制演示完成")


def demo_full_integration():
    """演示所有功能的完整集成"""
    print("\n" + "=" * 60)
    print("演示 6: 完整集成 - 所有新功能协同工作")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        best_dir = os.path.join(tmpdir, "best")

        config = TrainingConfig(
            max_episodes=40,
            max_steps_per_episode=50,
            eval_freq=10,
            eval_episodes=3,
            save_freq=20,
            early_stop_patience=30,
            early_stop_min_improvement=0.1,
            early_stop_metric="eval_reward",
            verbose=True,
            log_dir=tmpdir,
            log_to_file=True,
            log_to_console=False,
            show_progress=True,
            auto_save_best=True,
            best_model_dir=best_dir,
            checkpoint_on_interrupt=True,
            separate_eval_env=False
        )

        print(f"\n  配置:")
        print(f"    最大回合: {config.max_episodes}")
        print(f"    评估频率: 每{config.eval_freq}回合")
        print(f"    早停: 连续{config.early_stop_patience}轮无改进则停止")
        print(f"    自动保存最优: {config.auto_save_best}")
        print(f"    中断保存: {config.checkpoint_on_interrupt}")
        print()

        trainer = RLTrainer(config)
        agent = QLearningAgent(
            name="IntegrationAgent",
            learning_rate=0.2,
            epsilon=0.8,
            epsilon_decay=0.95
        )
        env = SimpleTestEnvironment(difficulty=1)
        eval_env = SimpleTestEnvironment(difficulty=1)

        stats = trainer.train(agent, env, eval_environment=eval_env)

        print(f"\n  训练总结:")
        print(f"    完成回合: {len(stats.episode_rewards)}")
        print(f"    最佳训练奖励: {stats.best_training_reward:.2f}")
        print(f"    最佳评估奖励: {stats.best_eval_reward:.2f}")
        print(f"    最优模型: {trainer.get_best_model_path() or '(无)'}")
        print(f"    早停状态: {trainer.get_early_stop_state()}")
        print(f"    异常统计: {trainer.get_exception_summary()}")

        # 验证检查点文件
        checkpoint_files = [f for f in os.listdir(tmpdir) if f.endswith('.json')]
        print(f"    检查点文件: {checkpoint_files}")

    print("\n  ✓ 完整集成演示完成")


def demo_curriculum_learning():
    """演示改进的课程学习"""
    print("\n" + "=" * 60)
    print("演示 7: 改进的课程学习设计")
    print("=" * 60)

    config = TrainingConfig(
        max_episodes=100,
        max_steps_per_episode=30,
        verbose=True,
        log_to_file=False,
        log_to_console=False,
        show_progress=False,
        eval_freq=20,
        early_stop_patience=0
    )

    print("\n  创建课程学习训练器...")
    trainer = CurriculumTrainer(config)

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

    agent = QLearningAgent(
        name="CurriculumAgent",
        learning_rate=0.2,
        epsilon=0.9,
        epsilon_decay=0.98
    )

    print("  开始课程学习训练...")
    stats = trainer.train(agent)

    progress = trainer.get_level_progress()
    print(f"\n  课程学习结果:")
    for p in progress:
        status = "完成" if p['completed'] else "未完成"
        print(f"    级别 {p['level_id']}: {status}, 训练{p['episodes_trained']}回合")

    print("\n  ✓ 课程学习演示完成")


def run_all_demos():
    """运行所有演示"""
    print("\n" + "=" * 70)
    print("强化学习训练器 - 增强功能完整演示")
    print("=" * 70)
    print("\n本演示展示所有新增功能：")
    print("  1. 断点续训：中断后可加载模型接续训练")
    print("  2. 自动保存最优模型：留存训练过程中效果最好的权重文件")
    print("  3. 简易控制台进度条：实时展示单回合与整体训练进度")
    print("  4. 运行异常捕获模块：报错自动记录日志且程序不会直接终止")
    print("  5. 早停判定机制：连续多轮收益无提升时自动终止无效训练")
    print("  6. 完整集成演示")
    print("  7. 课程学习")
    print("=" * 70)

    demos = [
        ("断点续训", demo_checkpoint_resume),
        ("自动保存最优模型", demo_auto_save_best_model),
        ("控制台进度条", demo_console_progress_bar),
        ("异常捕获模块", demo_exception_guard),
        ("早停判定机制", demo_early_stop_mechanism),
        ("完整集成", demo_full_integration),
        ("课程学习", demo_curriculum_learning),
    ]

    for i, (name, demo_func) in enumerate(demos, 1):
        try:
            demo_func()
        except Exception as e:
            print(f"\n  演示{i}({name})失败: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 70)
    print("所有演示完成！")
    print("=" * 70)

    print("\n功能总结:")
    print("""
 ✓ 断点续训: 训练中断后自动保存检查点（含智能体权重），
   重启时通过 resume_from_checkpoint 配置即可从断点继续训练
 ✓ 自动保存最优模型: 每次评估后若性能提升则自动保存最优权重，
   通过 auto_save_best + best_model_dir 配置
 ✓ 控制台进度条: 内置 ConsoleProgressBar 无需第三方依赖，
   实时显示整体进度、ETA、当前指标
 ✓ 异常捕获模块: ExceptionGuard 自动分类记录异常到日志文件，
   连续错误过多时建议终止，单次异常不中断训练
 ✓ 早停机制: EarlyStopMonitor 支持多种监控指标，
   配置 patience 和 min_improvement 自动终止无效训练
    """)


if __name__ == "__main__":
    run_all_demos()
