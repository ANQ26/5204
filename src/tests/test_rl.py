"""
强化学习模块测试 - 验证智能体、经验回放和训练器功能
"""

import unittest
import json
import random
from typing import Dict, List, Any, Optional
from unittest.mock import patch, MagicMock

from src.rl.agent import (
    Agent, RandomAgent, QLearningAgent, REINFORCEAgent, HeuristicAgent, Action
)
from src.rl.experience_buffer import (
    Experience, ExperienceBuffer, PrioritizedExperienceBuffer, EpisodeBuffer
)
from src.rl.trainer import (
    TrainingConfig, TrainingStats, RLTrainer, CurriculumTrainer
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
        
        self.assertEqual(config.max_episodes, 100)
        self.assertEqual(config.max_steps_per_episode, 50)
        self.assertEqual(config.learning_rate, 0.1)
        self.assertEqual(config.discount_factor, 0.99)
        self.assertEqual(config.epsilon, 1.0)


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


class TestRLTrainer(unittest.TestCase):
    """测试强化学习训练器"""
    
    def setUp(self):
        self.config = TrainingConfig(
            max_episodes=10,
            max_steps_per_episode=5,
            verbose=False
        )
        self.trainer = RLTrainer(self.config)
    
    def test_trainer_initialization(self):
        """测试训练器初始化"""
        self.assertEqual(self.trainer.config.max_episodes, 10)
        self.assertEqual(self.trainer.config.max_steps_per_episode, 5)
    
    def test_training_stats_initial(self):
        """测试初始训练统计"""
        stats = self.trainer.get_stats()
        
        self.assertEqual(stats.episodes_trained, 0)
        self.assertEqual(stats.total_steps, 0)


class TestCurriculumTrainer(unittest.TestCase):
    """测试课程学习训练器"""
    
    def setUp(self):
        self.config = TrainingConfig(
            max_episodes=10,
            max_steps_per_episode=5,
            verbose=False
        )
        self.trainer = CurriculumTrainer(self.config)
    
    def test_add_curriculum_stage(self):
        """测试添加课程阶段"""
        self.trainer.add_stage(
            stage_id="easy",
            description="简单阶段",
            difficulty=1,
            config_override={"max_steps_per_episode": 10}
        )
        
        self.assertEqual(len(self.trainer.curriculum_stages), 1)
    
    def test_get_current_stage(self):
        """测试获取当前阶段"""
        self.trainer.add_stage("stage1", "阶段1", 1)
        self.trainer.add_stage("stage2", "阶段2", 2)
        
        stage = self.trainer.get_current_stage()
        
        self.assertIsNotNone(stage)
        self.assertEqual(stage["stage_id"], "stage1")


if __name__ == "__main__":
    unittest.main(verbosity=2)
