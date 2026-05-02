"""
符号化反思模块测试 - 验证知识提取、规划和策略优化功能
"""

import unittest
import json
from typing import Dict, List, Any
from unittest.mock import patch, MagicMock

from src.reflection.symbolic_reflection import (
    RuleType, Rule, KnowledgeBase, SymbolicReflector
)
from src.reflection.planning import (
    PlanStep, Plan, Planner, ForwardSearchPlanner,
    HierarchicalPlanner, ReactivePlanner
)
from src.reflection.strategy_optimizer import (
    OptimizationMethod, Policy, StrategyOptimizer
)


class TestRuleType(unittest.TestCase):
    """测试规则类型枚举"""
    
    def test_rule_types_exist(self):
        """测试规则类型存在"""
        self.assertTrue(hasattr(RuleType, 'PRECONDITION'))
        self.assertTrue(hasattr(RuleType, 'EFFECT'))
        self.assertTrue(hasattr(RuleType, 'TRANSITION'))
        self.assertTrue(hasattr(RuleType, 'REWARD'))
        self.assertTrue(hasattr(RuleType, 'CONSTRAINT'))


class TestRule(unittest.TestCase):
    """测试规则类"""
    
    def test_rule_creation(self):
        """测试规则创建"""
        rule = Rule(
            rule_type=RuleType.PRECONDITION,
            conditions={'state': 'initial'},
            conclusions={'can_search': True},
            confidence=0.9,
            description="初始状态可以搜索"
        )
        
        self.assertEqual(rule.rule_type, RuleType.PRECONDITION)
        self.assertEqual(rule.conditions, {'state': 'initial'})
        self.assertEqual(rule.confidence, 0.9)
    
    def test_rule_to_dict(self):
        """测试规则转字典"""
        rule = Rule(
            rule_type=RuleType.REWARD,
            conditions={'action': 'checkout'},
            conclusions={'reward': 'positive'},
            confidence=0.8,
            description="结账获得正奖励"
        )
        
        d = rule.to_dict()
        
        self.assertEqual(d['rule_type'], 'REWARD')
        self.assertEqual(d['conditions'], {'action': 'checkout'})
        self.assertEqual(d['confidence'], 0.8)


class TestKnowledgeBase(unittest.TestCase):
    """测试知识库"""
    
    def setUp(self):
        self.kb = KnowledgeBase()
    
    def test_add_rule(self):
        """测试添加规则"""
        rule = Rule(
            rule_type=RuleType.PRECONDITION,
            conditions={'state': 'initial'},
            conclusions={'can_act': True},
            confidence=1.0
        )
        
        rule_id = self.kb.add_rule(rule)
        
        self.assertIsNotNone(rule_id)
        self.assertEqual(self.kb.get_rule_count(), 1)
    
    def test_get_rule(self):
        """测试获取规则"""
        rule = Rule(
            rule_type=RuleType.EFFECT,
            conditions={'action': 'add_to_cart'},
            conclusions={'cart_updated': True},
            confidence=0.9
        )
        
        rule_id = self.kb.add_rule(rule)
        retrieved = self.kb.get_rule(rule_id)
        
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.rule_type, RuleType.EFFECT)
    
    def test_get_rules_by_type(self):
        """测试按类型获取规则"""
        rule1 = Rule(rule_type=RuleType.PRECONDITION, conditions={}, conclusions={}, confidence=1.0)
        rule2 = Rule(rule_type=RuleType.REWARD, conditions={}, conclusions={}, confidence=1.0)
        rule3 = Rule(rule_type=RuleType.PRECONDITION, conditions={}, conclusions={}, confidence=1.0)
        
        self.kb.add_rule(rule1)
        self.kb.add_rule(rule2)
        self.kb.add_rule(rule3)
        
        precond_rules = self.kb.get_rules_by_type(RuleType.PRECONDITION)
        
        self.assertEqual(len(precond_rules), 2)
    
    def test_match_rules(self):
        """测试匹配规则"""
        rule = Rule(
            rule_type=RuleType.TRANSITION,
            conditions={'state': 'initial', 'action': 'search'},
            conclusions={'new_state': 'browsing'},
            confidence=0.95
        )
        
        self.kb.add_rule(rule)
        
        state = {'state': 'initial', 'action': 'search'}
        matched = self.kb.match_rules(state)
        
        self.assertEqual(len(matched), 1)
        self.assertEqual(matched[0].conclusions, {'new_state': 'browsing'})
    
    def test_infer(self):
        """测试推理"""
        rule = Rule(
            rule_type=RuleType.EFFECT,
            conditions={'action': 'checkout'},
            conclusions={'reward': 10.0, 'done': True},
            confidence=1.0
        )
        
        self.kb.add_rule(rule)
        
        inferences = self.kb.infer({'action': 'checkout'})
        
        self.assertIn('reward', inferences)
        self.assertIn('done', inferences)
    
    def test_update_confidence(self):
        """测试更新置信度"""
        rule = Rule(
            rule_type=RuleType.REWARD,
            conditions={},
            conclusions={},
            confidence=0.5
        )
        
        rule_id = self.kb.add_rule(rule)
        self.kb.update_confidence(rule_id, 0.8)
        
        updated = self.kb.get_rule(rule_id)
        self.assertEqual(updated.confidence, 0.8)
    
    def test_get_stats(self):
        """测试获取统计"""
        rule = Rule(
            rule_type=RuleType.PRECONDITION,
            conditions={},
            conclusions={},
            confidence=0.9
        )
        
        self.kb.add_rule(rule)
        
        stats = self.kb.get_stats()
        
        self.assertEqual(stats['total_rules'], 1)
        self.assertEqual(stats['rules_by_type']['PRECONDITION'], 1)


class TestSymbolicReflector(unittest.TestCase):
    """测试符号化反思器"""
    
    def setUp(self):
        self.reflector = SymbolicReflector()
    
    def test_initialization(self):
        """测试初始化"""
        self.assertIsNotNone(self.reflector.knowledge_base)
    
    def test_extract_transition_rules(self):
        """测试提取转换规则"""
        episode_data = {
            'states': [
                {'state': 'initial'},
                {'state': 'browsing'},
                {'state': 'cart_active'}
            ],
            'actions': [
                {'tool': 'search_products', 'parameters': {}},
                {'tool': 'add_to_cart', 'parameters': {}}
            ],
            'rewards': [1.0, 5.0],
            'dones': [False, False, True]
        }
        
        self.reflector._extract_transition_rules(episode_data)
        
        stats = self.reflector.knowledge_base.get_stats()
        self.assertGreater(stats['total_rules'], 0)
    
    def test_extract_reward_rules(self):
        """测试提取奖励规则"""
        episode_data = {
            'states': [{'state': 'browsing'}, {'state': 'cart_active'}],
            'actions': [
                {'tool': 'add_to_cart', 'parameters': {}},
                {'tool': 'checkout', 'parameters': {}}
            ],
            'rewards': [5.0, 20.0],
            'dones': [False, True]
        }
        
        self.reflector._extract_reward_rules(episode_data)
        
        reward_rules = self.reflector.knowledge_base.get_rules_by_type(RuleType.REWARD)
        self.assertGreater(len(reward_rules), 0)
    
    def test_reflect_on_episode(self):
        """测试回合反思"""
        episode_data = {
            'states': [
                {'state': 'initial'},
                {'state': 'browsing'},
                {'state': 'terminal'}
            ],
            'actions': [
                {'tool': 'search_products', 'parameters': {'query': 'test'}},
                {'tool': 'checkout', 'parameters': {}}
            ],
            'rewards': [1.0, 15.0],
            'dones': [False, False, True]
        }
        
        extracted_rules = self.reflector.reflect_on_episode(episode_data)
        
        self.assertIsInstance(extracted_rules, list)
        self.assertGreater(self.reflector.knowledge_base.get_rule_count(), 0)
    
    def test_get_reflection_summary(self):
        """测试获取反思摘要"""
        episode_data = {
            'states': [{'state': 'initial'}, {'state': 'terminal'}],
            'actions': [{'tool': 'search', 'parameters': {}}],
            'rewards': [10.0],
            'dones': [False, True]
        }
        
        self.reflector.reflect_on_episode(episode_data)
        
        summary = self.reflector.get_reflection_summary()
        
        self.assertIn('knowledge_base', summary)
        self.assertIn('episodes_reflected', summary)
        self.assertIn('high_confidence_rules', summary)
    
    def test_generate_improvement_suggestions(self):
        """测试生成改进建议"""
        for i in range(3):
            episode_data = {
                'states': [{'step': i}, {'step': i + 1}],
                'actions': [{'tool': f'action_{i}', 'parameters': {}}],
                'rewards': [float(i * 2)],
                'dones': [False, True]
            }
            self.reflector.reflect_on_episode(episode_data)
        
        suggestions = self.reflector.generate_improvement_suggestions()
        
        self.assertIsInstance(suggestions, list)


class TestPlanStep(unittest.TestCase):
    """测试规划步骤"""
    
    def test_plan_step_creation(self):
        """测试规划步骤创建"""
        step = PlanStep(
            tool="search_products",
            parameters={"query": "Laptop"},
            estimated_reward=1.0,
            description="搜索产品"
        )
        
        self.assertEqual(step.tool, "search_products")
        self.assertEqual(step.estimated_reward, 1.0)
        self.assertEqual(step.description, "搜索产品")


class TestPlan(unittest.TestCase):
    """测试规划"""
    
    def test_plan_creation(self):
        """测试规划创建"""
        steps = [
            PlanStep(tool="search", parameters={}, estimated_reward=1.0),
            PlanStep(tool="add", parameters={}, estimated_reward=5.0),
            PlanStep(tool="checkout", parameters={}, estimated_reward=20.0)
        ]
        
        plan = Plan(
            steps=steps,
            estimated_total_reward=26.0,
            planning_time=0.05
        )
        
        self.assertEqual(len(plan), 3)
        self.assertEqual(plan.estimated_total_reward, 26.0)
        self.assertEqual(len(plan.steps), 3)


class TestForwardSearchPlanner(unittest.TestCase):
    """测试前向搜索规划器"""
    
    def setUp(self):
        self.planner = ForwardSearchPlanner(
            search_strategy="bfs",
            max_depth=10
        )
    
    def test_initialization(self):
        """测试初始化"""
        self.assertEqual(self.planner.search_strategy, "bfs")
        self.assertEqual(self.planner.max_depth, 10)
    
    def test_register_action_model(self):
        """测试注册动作模型"""
        self.planner.register_action_model(
            action_name="search",
            preconditions={'state': 'initial'},
            effects={'state': 'browsing'},
            reward=1.0
        )
        
        self.assertEqual(len(self.planner.action_models), 1)
    
    def test_bfs_plan(self):
        """测试BFS规划"""
        self.planner.register_action_model(
            action_name="step1",
            preconditions={'level': 0},
            effects={'level': 1},
            reward=1.0
        )
        self.planner.register_action_model(
            action_name="step2",
            preconditions={'level': 1},
            effects={'level': 2},
            reward=2.0
        )
        
        plan = self.planner.plan(
            initial_state={'level': 0},
            goal_state={'level': 2},
            available_actions=["step1", "step2"]
        )
        
        self.assertIsNotNone(plan)
    
    def test_astar_plan(self):
        """测试A*规划"""
        planner = ForwardSearchPlanner(search_strategy="astar", max_depth=10)
        
        planner.register_action_model("a1", {}, {'x': 1}, 1.0)
        planner.register_action_model("a2", {'x': 1}, {'x': 2}, 1.0)
        
        plan = planner.plan(
            initial_state={'x': 0},
            goal_state={'x': 2},
            available_actions=["a1", "a2"]
        )
        
        self.assertIsNotNone(plan)
    
    def test_greedy_plan(self):
        """测试贪心规划"""
        planner = ForwardSearchPlanner(search_strategy="greedy", max_depth=10)
        
        planner.register_action_model("g1", {}, {'v': 1}, 5.0)
        planner.register_action_model("g2", {'v': 1}, {'v': 2}, 10.0)
        
        plan = planner.plan(
            initial_state={'v': 0},
            goal_state={'v': 2},
            available_actions=["g1", "g2"]
        )
        
        self.assertIsNotNone(plan)


class TestHierarchicalPlanner(unittest.TestCase):
    """测试层次规划器"""
    
    def setUp(self):
        self.planner = HierarchicalPlanner()
    
    def test_add_high_level_goal(self):
        """测试添加高层目标"""
        self.planner.add_high_level_goal(
            goal_name="complete_shopping",
            sub_goals=["search", "add_to_cart", "checkout"]
        )
        
        self.assertEqual(len(self.planner.high_level_goals), 1)


class TestReactivePlanner(unittest.TestCase):
    """测试反应式规划器"""
    
    def setUp(self):
        self.planner = ReactivePlanner()
    
    def test_add_rule(self):
        """测试添加规则"""
        self.planner.add_rule(
            condition='state.get("s") == 0',
            action="step1",
            priority=1,
            parameters={"param": "value"}
        )
        
        self.assertEqual(len(self.planner.rules), 1)
    
    def test_plan_with_matching_rule(self):
        """测试匹配规则的规划"""
        self.planner.add_rule(
            condition='state.get("status") == "initial"',
            action="search",
            priority=2,
            parameters={"query": "test"}
        )
        
        plan = self.planner.plan(
            initial_state={"status": "initial"},
            goal_state={},
            available_actions=["search"]
        )
        
        self.assertIsNotNone(plan)
        self.assertGreater(len(plan), 0)


class TestOptimizationMethod(unittest.TestCase):
    """测试优化方法枚举"""
    
    def test_methods_exist(self):
        """测试方法存在"""
        self.assertTrue(hasattr(OptimizationMethod, 'Q_ITERATION'))
        self.assertTrue(hasattr(OptimizationMethod, 'VALUE_ITERATION'))
        self.assertTrue(hasattr(OptimizationMethod, 'POLICY_ITERATION'))
        self.assertTrue(hasattr(OptimizationMethod, 'CROSS_ENTROPY'))
        self.assertTrue(hasattr(OptimizationMethod, 'EVOLUTIONARY'))


class TestPolicy(unittest.TestCase):
    """测试策略"""
    
    def test_policy_creation(self):
        """测试策略创建"""
        policy = Policy(
            name="TestPolicy",
            state_action_probs={
                "state1": {"action1": 0.8, "action2": 0.2},
                "state2": {"action1": 0.1, "action2": 0.9}
            },
            value_function={"state1": 10.0, "state2": 15.0}
        )
        
        self.assertEqual(policy.name, "TestPolicy")
        self.assertEqual(len(policy.state_action_probs), 2)
        self.assertEqual(len(policy.value_function), 2)


class TestStrategyOptimizer(unittest.TestCase):
    """测试策略优化器"""
    
    def setUp(self):
        self.optimizer = StrategyOptimizer(
            method=OptimizationMethod.Q_ITERATION,
            discount_factor=0.95,
            learning_rate=0.1
        )
    
    def test_initialization(self):
        """测试初始化"""
        self.assertEqual(self.optimizer.method, OptimizationMethod.Q_ITERATION)
        self.assertEqual(self.optimizer.discount_factor, 0.95)
        self.assertEqual(self.optimizer.learning_rate, 0.1)
    
    def test_add_experience(self):
        """测试添加经验"""
        self.optimizer.add_experience(
            state_key="s1",
            action="a1",
            reward=10.0,
            next_state_key="s2",
            done=False
        )
        
        self.assertEqual(len(self.optimizer.experiences), 1)
    
    def test_optimize_q_iteration(self):
        """测试Q迭代优化"""
        for i in range(5):
            self.optimizer.add_experience(
                state_key=f"s{i}",
                action=f"a{i}",
                reward=float(i),
                next_state_key=f"s{i + 1}",
                done=(i == 4)
            )
        
        policy = self.optimizer.optimize(iterations=10)
        
        self.assertIsInstance(policy, Policy)
    
    def test_optimize_value_iteration(self):
        """测试值迭代优化"""
        optimizer = StrategyOptimizer(
            method=OptimizationMethod.VALUE_ITERATION,
            discount_factor=0.9
        )
        
        optimizer.add_experience("s1", "a1", 1.0, "s2", False)
        optimizer.add_experience("s2", "a2", 10.0, "s3", True)
        
        policy = optimizer.optimize(iterations=10)
        
        self.assertIsInstance(policy, Policy)
    
    def test_optimize_policy_iteration(self):
        """测试策略迭代优化"""
        optimizer = StrategyOptimizer(
            method=OptimizationMethod.POLICY_ITERATION,
            discount_factor=0.9
        )
        
        optimizer.add_experience("s1", "a1", 5.0, "s2", False)
        optimizer.add_experience("s2", "a2", 15.0, "s3", True)
        
        policy = optimizer.optimize(iterations=5)
        
        self.assertIsInstance(policy, Policy)
    
    def test_optimize_cross_entropy(self):
        """测试交叉熵优化"""
        optimizer = StrategyOptimizer(
            method=OptimizationMethod.CROSS_ENTROPY
        )
        
        for i in range(10):
            optimizer.add_experience(
                f"s{i}",
                f"a{i % 2}",
                float(i * 2),
                f"s{i + 1}",
                (i == 9)
            )
        
        policy = optimizer.optimize(iterations=5, population_size=10)
        
        self.assertIsInstance(policy, Policy)
    
    def test_optimize_evolutionary(self):
        """测试进化优化"""
        optimizer = StrategyOptimizer(
            method=OptimizationMethod.EVOLUTIONARY
        )
        
        for i in range(8):
            optimizer.add_experience(
                f"s{i}",
                f"a{i % 3}",
                float(i),
                f"s{i + 1}",
                (i == 7)
            )
        
        policy = optimizer.optimize(iterations=5, population_size=10)
        
        self.assertIsInstance(policy, Policy)
    
    def test_get_best_action(self):
        """测试获取最佳动作"""
        optimizer = StrategyOptimizer(method=OptimizationMethod.Q_ITERATION)
        
        optimizer.add_experience("s1", "a1", 5.0, "s2", False)
        optimizer.add_experience("s1", "a2", 15.0, "s2", False)
        
        optimizer.optimize(iterations=10)
        
        best_action = optimizer.get_best_action("s1")
        
        self.assertIsNotNone(best_action)


if __name__ == "__main__":
    unittest.main(verbosity=2)
