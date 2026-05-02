"""
强化学习模块测试 - 验证智能体、经验回放和训练器功能
"""

import unittest
import json
import random
import tempfile
import os
from typing import Dict, List, Any, Optional
from unittest.mock import patch, MagicMock

from src.rl.agent import (
    Agent, RandomAgent, QLearningAgent, REINFORCEAgent, HeuristicAgent, 
    Action, load_agent_from_file
)
from src.rl.experience_buffer import (
    Experience, ExperienceBuffer, PrioritizedExperienceBuffer, EpisodeBuffer
)
from src.rl.trainer import (
    TrainingConfig, TrainingStats, RLTrainer, CurriculumTrainer,
    TrainingStage, CurriculumLevel, TrainingError
)


class TestAction(unittest.TestCase):
    """测试动作类"""
    
    def test_action_creation(self):
        """测试动作创建"""
        action = Action(
            tool="test_action",
            parameters={"param1": "value1"},
            confidence=0.8,
            reasoning="测试推理"
        )
        
        self.assertEqual(action.tool, "test_action")
        self.assertEqual(action.parameters, {"param1": "value1"})
        self.assertEqual(action.confidence, 0.8)
        self.assertEqual(action.reasoning, "测试推理")
    
    def test_action_to_dict(self):
        """测试动作转字典"""
        action = Action(
            tool="test",
            parameters={"p": 1},
            confidence=0.9,
            reasoning="test"
        )
        
        d = action.to_dict()
        
        self.assertEqual(d["tool"], "test")
        self.assertEqual(d["parameters"], {"p": 1})


class TestRandomAgent(unittest.TestCase):
    """测试随机智能体"""
    
    def setUp(self):
        self.agent = RandomAgent("TestRandomAgent")
    
    def test_agent_name(self):
        """测试智能体名称"""
        self.assertEqual(self.agent.name, "TestRandomAgent")
    
    def test_act_returns_action(self):
        """测试act方法返回动作"""
        observation = {"available_tools": ["tool1", "tool2"]}
        
        action = self.agent.act(observation)
        
        self.assertIsNotNone(action)
        self.assertIsInstance(action, Action)
        self.assertIn(action.tool, ["tool1", "tool2"])
    
    def test_act_without_tools(self):
        """测试无可用工具时act返回None"""
        observation = {}
        
        action = self.agent.act(observation)
        
        self.assertIsNone(action)
    
    def test_observe_called(self):
        """测试observe方法被调用"""
        observation = {"test": 1}
        action = Action(tool="test", parameters={})
        next_observation = {"test": 2}
        
        self.agent.observe(observation, action, 1.0, next_observation, False)
    
    def test_save_load_random_agent(self):
        """测试随机智能体保存/加载"""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "random_agent.json")
            
            success = self.agent.save(filepath)
            self.assertTrue(success)
            
            loaded_agent = RandomAgent("Loaded")
            load_success = loaded_agent.load(filepath)
            self.assertTrue(load_success)


class TestQLearningAgent(unittest.TestCase):
    """测试Q学习智能体"""
    
    def setUp(self):
        self.agent = QLearningAgent(
            name="TestQLearningAgent",
            learning_rate=0.1,
            discount_factor=0.95,
            epsilon=1.0,
            epsilon_decay=0.995,
            epsilon_min=0.01
        )
    
    def test_initialization(self):
        """测试初始化参数"""
        self.assertEqual(self.agent.learning_rate, 0.1)
        self.assertEqual(self.agent.discount_factor, 0.95)
        self.assertEqual(self.agent.epsilon, 1.0)
        self.assertEqual(self.agent.epsilon_min, 0.01)
        self.assertTrue(self.agent.training)
    
    def test_get_q_value_new_state(self):
        """测试新状态的Q值"""
        state_key = "test_state"
        action = "test_action"
        
        q_value = self.agent._get_q_value(state_key, action)
        
        self.assertEqual(q_value, 0.0)
    
    def test_update_q_value(self):
        """测试Q值更新"""
        state_key = "test_state"
        action = "test_action"
        next_state_key = "next_state"
        
        self.agent._update_q_value(state_key, action, 1.0, next_state_key, 0.5, False)
        
        q_value = self.agent._get_q_value(state_key, action)
        
        self.assertNotEqual(q_value, 0.0)
    
    def test_epsilon_greedy_exploration(self):
        """测试ε-greedy探索"""
        self.agent.epsilon = 1.0
        
        with patch('random.random', return_value=0.0):
            action = self.agent.act(
                {"available_tools": ["tool1", "tool2"]}
            )
        
        self.assertIsNotNone(action)
    
    def test_epsilon_decay(self):
        """测试ε衰减"""
        initial_epsilon = self.agent.epsilon
        
        for _ in range(10):
            self.agent._decay_epsilon()
        
        self.assertLess(self.agent.epsilon, initial_epsilon)
        self.assertGreaterEqual(self.agent.epsilon, self.agent.epsilon_min)
    
    def test_get_q_table_size(self):
        """测试Q表大小"""
        self.agent._update_q_value("s1", "a1", 1.0, "s2", 0.5, False)
        self.agent._update_q_value("s2", "a2", 1.0, "s3", 0.5, False)
        
        size = self.agent.get_q_table_size()
        
        self.assertEqual(size, 2)
    
    def test_train_eval_modes(self):
        """测试训练/评估模式切换"""
        self.assertTrue(self.agent.training)
        
        self.agent.eval_mode()
        self.assertFalse(self.agent.training)
        
        self.agent.train_mode()
        self.assertTrue(self.agent.training)
    
    def test_eval_mode_no_exploration(self):
        """测试评估模式下不进行探索"""
        observation = {"available_tools": ["tool1", "tool2"]}
        
        self.agent._update_q_value("state", "tool1", 10.0, "next", 0, True)
        self.agent._update_q_value("state", "tool2", 1.0, "next", 0, True)
        
        self.agent.eval_mode()
        
        with patch('src.rl.agent.QLearningAgent._extract_state_key', return_value="state"):
            actions = [self.agent.act(observation) for _ in range(10)]
        
        tools = [a.tool for a in actions if a]
        self.assertTrue(all(t == "tool1" for t in tools))
    
    def test_save_load_qlearning(self):
        """测试Q学习智能体保存/加载"""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "q_agent.json")
            
            self.agent._update_q_value("s1", "a1", 1.0, "s2", 0.5, False)
            
            q_size_before = self.agent.get_q_table_size()
            epsilon_before = self.agent.epsilon
            
            success = self.agent.save(filepath)
            self.assertTrue(success)
            
            loaded_agent = QLearningAgent("Loaded")
            load_success = loaded_agent.load(filepath)
            self.assertTrue(load_success)
            
            self.assertEqual(loaded_agent.get_q_table_size(), q_size_before)
            self.assertAlmostEqual(loaded_agent.epsilon, epsilon_before)
    
    def test_load_agent_from_file(self):
        """测试自动识别智能体类型加载"""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "agent.json")
            
            success = self.agent.save(filepath)
            self.assertTrue(success)
            
            loaded_agent = load_agent_from_file(filepath)
            
            self.assertIsNotNone(loaded_agent)
            self.assertIsInstance(loaded_agent, QLearningAgent)


class TestREINFORCEAgent(unittest.TestCase):
    """测试REINFORCE智能体"""
    
    def setUp(self):
        self.agent = REINFORCEAgent(
            name="TestREINFORCEAgent",
            learning_rate=0.01
        )
    
    def test_initialization(self):
        """测试初始化"""
        self.assertEqual(self.agent.name, "TestREINFORCEAgent")
        self.assertEqual(self.agent.learning_rate, 0.01)
    
    def test_act_returns_action(self):
        """测试act方法"""
        observation = {"available_tools": ["tool1", "tool2"]}
        
        action = self.agent.act(observation)
        
        self.assertIsNotNone(action)
        self.assertIsInstance(action, Action)
    
    def test_observe_stores_experience(self):
        """测试observe存储经验"""
        observation = {"test": 1}
        action = Action(tool="test", parameters={})
        next_observation = {"test": 2}
        
        self.agent.observe(observation, action, 1.0, next_observation, False)
        
        self.assertEqual(len(self.agent.episode_buffer), 1)
    
    def test_save_load_reinforce(self):
        """测试REINFORCE智能体保存/加载"""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "reinforce_agent.json")
            
            observation = {"available_tools": ["a1", "a2"]}
            self.agent.act(observation)
            
            success = self.agent.save(filepath)
            self.assertTrue(success)
            
            loaded_agent = REINFORCEAgent("Loaded")
            load_success = loaded_agent.load(filepath)
            self.assertTrue(load_success)


class TestHeuristicAgent(unittest.TestCase):
    """测试启发式智能体"""
    
    def setUp(self):
        self.agent = HeuristicAgent("TestHeuristicAgent")
    
    def test_add_heuristic(self):
        """测试添加启发式规则"""
        self.agent.add_heuristic(
            condition=lambda obs: obs.get("state") == "initial",
            action_func=lambda obs: Action(
                tool="search",
                parameters={"query": "test"},
                reasoning="初始状态搜索"
            ),
            priority=1
        )
        
        self.assertEqual(len(self.agent.heuristics), 1)
    
    def test_act_with_matching_heuristic(self):
        """测试匹配启发式规则的act"""
        self.agent.add_heuristic(
            condition=lambda obs: obs.get("current_state") == "initial",
            action_func=lambda obs: Action(
                tool="search_products",
                parameters={"query": "Laptop"},
                reasoning="初始状态搜索产品"
            ),
            priority=1
        )
        
        observation = {"current_state": "initial", "available_tools": ["search_products"]}
        
        action = self.agent.act(observation)
        
        self.assertIsNotNone(action)
        self.assertEqual(action.tool, "search_products")


class TestExperience(unittest.TestCase):
    """测试经验类"""
    
    def test_experience_creation(self):
        """测试经验创建"""
        experience = Experience(
            state={"test": 1},
            action=Action(tool="test", parameters={}),
            reward=1.0,
            next_state={"test": 2},
            done=False,
            info={"extra": "data"}
        )
        
        self.assertEqual(experience.reward, 1.0)
        self.assertEqual(experience.done, False)
        self.assertEqual(experience.info, {"extra": "data"})


class TestExperienceBuffer(unittest.TestCase):
    """测试经验回放缓冲区"""
    
    def setUp(self):
        self.buffer = ExperienceBuffer(capacity=100)
    
    def test_add_experience(self):
        """测试添加经验"""
        exp = Experience(
            state={"s": 1},
            action=Action(tool="a", parameters={}),
            reward=1.0,
            next_state={"s": 2},
            done=False
        )
        
        self.buffer.add(exp)
        
        self.assertEqual(self.buffer.size(), 1)
    
    def test_sample_batch(self):
        """测试批量采样"""
        for i in range(10):
            exp = Experience(
                state={"s": i},
                action=Action(tool=f"a{i}", parameters={}),
                reward=float(i),
                next_state={"s": i + 1},
                done=i == 9
            )
            self.buffer.add(exp)
        
        batch = self.buffer.sample(batch_size=5)
        
        self.assertEqual(len(batch), 5)
    
    def test_capacity_limit(self):
        """测试容量限制"""
        buffer = ExperienceBuffer(capacity=5)
        
        for i in range(10):
            exp = Experience(
                state={"s": i},
                action=Action(tool="a", parameters={}),
                reward=1.0,
                next_state={"s": i + 1},
                done=False
            )
            buffer.add(exp)
        
        self.assertEqual(buffer.size(), 5)
    
    def test_clear_buffer(self):
        """测试清空缓冲区"""
        exp = Experience(
            state={"s": 1},
            action=Action(tool="a", parameters={}),
            reward=1.0,
            next_state={"s": 2},
            done=False
        )
        
        self.buffer.add(exp)
        self.buffer.clear()
        
        self.assertEqual(self.buffer.size(), 0)
    
    def test_len_method(self):
        """测试__len__方法"""
        exp = Experience(
            state={"s": 1},
            action=Action(tool="a", parameters={}),
            reward=1.0,
            next_state={"s": 2},
            done=False
        )
        
        self.buffer.add(exp)
        self.buffer.add(exp)
        
        self.assertEqual(len(self.buffer), 2)


class TestPrioritizedExperienceBuffer(unittest.TestCase):
    """测试优先经验回放缓冲区"""
    
    def setUp(self):
        self.buffer = PrioritizedExperienceBuffer(capacity=100, alpha=0.7)
    
    def test_add_with_priority(self):
        """测试带优先级添加"""
        exp = Experience(
            state={"s": 1},
            action=Action(tool="a", parameters={}),
            reward=10.0,
            next_state={"s": 2},
            done=False
        )
        
        self.buffer.add(exp, priority=2.0)
        
        self.assertEqual(self.buffer.size(), 1)


class TestEpisodeBuffer(unittest.TestCase):
    """测试回合缓冲区"""
    
    def setUp(self):
        self.buffer = EpisodeBuffer(capacity=10)
    
    def test_add_episode(self):
        """测试添加回合"""
        episode = [
            Experience(
                state={"s": 1},
                action=Action(tool="a", parameters={}),
                reward=1.0,
                next_state={"s": 2},
                done=False
            ),
            Experience(
                state={"s": 2},
                action=Action(tool="b", parameters={}),
                reward=10.0,
                next_state={"s": 3},
                done=True
            )
        ]
        
        self.buffer.add_episode(episode)
        
        self.assertEqual(self.buffer.num_episodes(), 1)
    
    def test_sample_episode(self):
        """测试采样回合"""
        for i in range(3):
            episode = [
                Experience(
                    state={"s": i},
                    action=Action(tool="a", parameters={}),
                    reward=float(i),
                    next_state={"s": i + 1},
                    done=True
                )
            ]
            self.buffer.add_episode(episode)
        
        sampled = self.buffer.sample_episodes(num_episodes=2)
        
        self.assertEqual(len(sampled), 2)


class TestTrainingConfig(unittest.TestCase):
    """测试训练配置"""
    
    def test_default_config(self):
        """测试默认配置"""
        config = TrainingConfig()
        
        self.assertEqual(config.max_episodes, 1000)
        self.assertEqual(config.max_steps_per_episode, 100)
        self.assertEqual(config.learning_rate, 0.1)
        self.assertEqual(config.discount_factor, 0.99)
        self.assertEqual(config.epsilon, 1.0)
        self.assertTrue(config.use_prioritized_replay)
        self.assertTrue(config.log_to_file)
        self.assertTrue(config.show_progress)
    
    def test_custom_config(self):
        """测试自定义配置"""
        config = TrainingConfig(
            max_episodes=50,
            early_stop_patience=20,
            separate_eval_env=False
        )
        
        self.assertEqual(config.max_episodes, 50)
        self.assertEqual(config.early_stop_patience, 20)
        self.assertFalse(config.separate_eval_env)


class TestTrainingStats(unittest.TestCase):
    """测试训练统计"""
    
    def test_stats_creation(self):
        """测试统计创建"""
        stats = TrainingStats(
            episodes_trained=100,
            total_steps=500,
            total_reward=1000.0,
            avg_reward_per_episode=10.0,
            best_episode_reward=25.0
        )
        
        self.assertEqual(stats.episodes_trained, 100)
        self.assertEqual(stats.avg_reward_per_episode, 10.0)
    
    def test_get_summary(self):
        """测试获取统计摘要"""
        stats = TrainingStats(
            episodes_trained=50,
            total_steps=200,
            total_reward=500.0
        )
        
        summary = stats.get_summary()
        
        self.assertEqual(summary["episodes_trained"], 50)
        self.assertIn("avg_reward", summary)


class SimpleTestEnvironment:
    """简单测试环境"""
    
    def __init__(self, difficulty=1):
        self.difficulty = difficulty
        self.state = 0
        self.max_state = 5 * difficulty
        self.available_tools = ["move_right", "move_left"]
    
    def reset(self):
        self.state = 0
        return {
            "current_state": "running",
            "state_variables": {"position": self.state},
            "available_tools": self.available_tools
        }
    
    def execute_tool(self, tool, **kwargs):
        if tool == "move_right":
            self.state = min(self.max_state, self.state + 1)
            reward = 0.1
        elif tool == "move_left":
            self.state = max(0, self.state - 1)
            reward = -0.1
        else:
            reward = -0.5
        
        done = False
        info = {}
        
        if self.state >= self.max_state:
            done = True
            reward = 10.0 * self.difficulty
            info["success"] = True
        
        return "", reward, done, info
    
    def get_observation(self):
        return {
            "current_state": "running" if self.state < self.max_state else "terminal",
            "state_variables": {"position": self.state},
            "available_tools": self.available_tools
        }


class TestRLTrainer(unittest.TestCase):
    """测试强化学习训练器"""
    
    def setUp(self):
        self.config = TrainingConfig(
            max_episodes=5,
            max_steps_per_episode=10,
            verbose=False,
            log_to_file=False,
            show_progress=False
        )
        self.trainer = RLTrainer(self.config)
    
    def test_trainer_initialization(self):
        """测试训练器初始化"""
        self.assertEqual(self.trainer.config.max_episodes, 5)
        self.assertEqual(self.trainer.config.max_steps_per_episode, 10)
    
    def test_training_stats_initial(self):
        """测试初始训练统计"""
        stats = self.trainer.get_stats()
        
        self.assertEqual(stats.episodes_trained, 0)
        self.assertEqual(stats.total_steps, 0)
    
    def test_simple_training(self):
        """测试简单训练"""
        agent = QLearningAgent("TestAgent")
        env = SimpleTestEnvironment(difficulty=1)
        
        stats = self.trainer.train(agent, env)
        
        self.assertGreater(len(stats.episode_rewards), 0)
    
    def test_save_load_checkpoint(self):
        """测试Checkpoint保存/加载"""
        with tempfile.TemporaryDirectory() as tmpdir:
            agent = QLearningAgent("TestAgent")
            env = SimpleTestEnvironment(difficulty=1)
            
            self.trainer.train(agent, env)
            
            checkpoint_path = os.path.join(tmpdir, "checkpoint.json")
            success = self.trainer.save_checkpoint(checkpoint_path, agent)
            self.assertTrue(success)
            
            new_trainer = RLTrainer(self.config)
            load_success = new_trainer.load_checkpoint(checkpoint_path)
            self.assertTrue(load_success)


class TestCurriculumLevel(unittest.TestCase):
    """测试课程级别"""
    
    def test_level_creation(self):
        """测试级别创建"""
        def env_creator():
            return SimpleTestEnvironment(difficulty=1)
        
        level = CurriculumLevel(
            level_id="easy",
            difficulty=1,
            environment_creator=env_creator,
            min_success_rate=0.7,
            min_consecutive_success=3,
            max_episodes_per_level=50,
            description="简单级别"
        )
        
        self.assertEqual(level.level_id, "easy")
        self.assertEqual(level.difficulty, 1)
        self.assertEqual(level.min_success_rate, 0.7)
    
    def test_level_check_progression(self):
        """测试级别升级检查"""
        def env_creator():
            return SimpleTestEnvironment(difficulty=1)
        
        level = CurriculumLevel(
            level_id="test",
            difficulty=1,
            environment_creator=env_creator,
            min_success_rate=0.7,
            min_consecutive_success=2,
            max_episodes_per_level=10
        )
        
        level.level_stats["episodes_trained"] = 5
        level.level_stats["success_rate"] = 0.8
        level.level_stats["consecutive_successes"] = 3
        level.level_stats["avg_reward"] = 10.0
        
        can_progress = level.can_progress()
        self.assertTrue(can_progress)


class TestCurriculumTrainer(unittest.TestCase):
    """测试课程学习训练器"""
    
    def setUp(self):
        self.config = TrainingConfig(
            max_episodes=50,
            max_steps_per_episode=10,
            verbose=False,
            log_to_file=False,
            show_progress=False
        )
        self.trainer = CurriculumTrainer(self.config)
    
    def test_add_level(self):
        """测试添加级别"""
        def create_env_diff1():
            return SimpleTestEnvironment(difficulty=1)
        
        self.trainer.add_level(
            level_id="easy",
            difficulty=1,
            environment_creator=create_env_diff1,
            min_success_rate=0.7,
            min_consecutive_success=2,
            max_episodes_per_level=50,
            description="简单"
        )
        
        self.assertEqual(len(self.trainer.levels), 1)
    
    def test_get_level_progress(self):
        """测试获取级别进度"""
        def create_env1():
            return SimpleTestEnvironment(difficulty=1)
        
        self.trainer.add_level("easy", 1, create_env1)
        
        progress = self.trainer.get_level_progress()
        
        self.assertEqual(len(progress), 1)
        self.assertEqual(progress[0]["level_id"], "easy")
        self.assertFalse(progress[0]["completed"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
