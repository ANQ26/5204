"""
训练流程示例 - 演示完整的智能体训练流程
"""

import json
import time
from typing import Dict, List, Optional, Any, Tuple

from src.sandbox.environment_manager import EnvironmentManager
from src.sandbox.environment_executor import EnvironmentExecutor
from src.rl.agent import QLearningAgent, REINFORCEAgent, Action
from src.rl.trainer import RLTrainer, TrainingConfig
from src.reflection.symbolic_reflection import SymbolicReflector, KnowledgeBase, Rule, RuleType
from src.reflection.strategy_optimizer import StrategyOptimizer, OptimizationMethod, Policy
from src.reflection.planning import ForwardSearchPlanner, HierarchicalPlanner, PlanStep, Plan


def run_training_pipeline():
    """
    运行完整的训练流程示例
    
    展示从"被动执行"到"自主规划"的进化过程
    """
    print("=" * 70)
    print("环境工厂系统 - 完整训练流程演示")
    print("=" * 70)
    print("\n本演示展示智能体如何从被动执行进化到自主规划:")
    print("  阶段1: 被动执行 - 随机或规则驱动的动作")
    print("  阶段2: 强化学习 - 从经验中学习Q值")
    print("  阶段3: 符号化反思 - 提取规则和模式")
    print("  阶段4: 自主规划 - 基于知识库进行前瞻性规划")
    print("=" * 70)
    
    print("\n" + "-" * 70)
    print("阶段1: 环境初始化和被动执行")
    print("-" * 70)
    
    env_manager = EnvironmentManager()
    
    print("\n1.1 创建电商环境...")
    env_instance = env_manager.create_ecommerce_environment(
        name="智能购物训练环境",
        user_balance=2000.0
    )
    
    print(f"   环境ID: {env_instance.id}")
    print(f"   环境名称: {env_instance.name}")
    
    print("\n1.2 加载环境实例...")
    environment = env_manager.load_environment(env_instance)
    
    print("\n1.3 被动执行演示 (随机智能体)...")
    from src.rl.agent import RandomAgent
    
    random_agent = RandomAgent("RandomShopper")
    executor = EnvironmentExecutor(timeout=5.0)
    
    print("   运行随机智能体10个回合:")
    random_rewards = []
    for episode in range(10):
        observation = environment.reset()
        episode_reward = 0.0
        steps = 0
        
        for step in range(15):
            action = random_agent.act(observation)
            if action is None:
                break
            
            result, reward, done, info = executor.execute_tool(
                environment, action.tool, action.parameters
            )
            
            episode_reward += reward
            steps += 1
            
            next_observation = environment.get_observation()
            random_agent.observe(observation, action, reward, next_observation, done)
            observation = next_observation
            
            if done:
                break
        
        random_rewards.append(episode_reward)
        if (episode + 1) % 5 == 0:
            print(f"   回合 {episode + 1}: 奖励={episode_reward:.2f}, 步数={steps}")
    
    random_avg = sum(random_rewards) / len(random_rewards)
    print(f"\n   随机智能体平均奖励: {random_avg:.2f}")
    
    print("\n" + "-" * 70)
    print("阶段2: 强化学习训练")
    print("-" * 70)
    
    print("\n2.1 初始化Q学习智能体...")
    q_agent = QLearningAgent(
        name="QLearningShopper",
        learning_rate=0.15,
        discount_factor=0.95,
        epsilon=0.8,
        epsilon_decay=0.98,
        epsilon_min=0.05
    )
    
    print("\n2.2 配置训练参数...")
    config = TrainingConfig(
        max_episodes=50,
        max_steps_per_episode=20,
        learning_rate=0.15,
        discount_factor=0.95,
        epsilon=0.8,
        epsilon_decay=0.98,
        eval_freq=10,
        eval_episodes=5,
        verbose=True
    )
    
    print("\n2.3 开始强化学习训练...")
    trainer = RLTrainer(config)
    
    q_learning_rewards = []
    for episode in range(config.max_episodes):
        observation = environment.reset()
        episode_reward = 0.0
        steps = 0
        
        for step in range(config.max_steps_per_episode):
            action = q_agent.act(observation)
            if action is None:
                break
            
            result, reward, done, info = environment.execute_tool(
                action.tool, **action.parameters
            )
            
            episode_reward += reward
            steps += 1
            
            next_observation = environment.get_observation()
            q_agent.observe(observation, action, reward, next_observation, done)
            observation = next_observation
            
            if done:
                break
        
        q_learning_rewards.append(episode_reward)
        
        if (episode + 1) % 10 == 0:
            avg_reward = sum(q_learning_rewards[-10:]) / 10
            print(f"   回合 {episode + 1}: 近10回合平均奖励={avg_reward:.2f}, "
                  f"epsilon={q_agent.epsilon:.4f}, Q表大小={q_agent.get_q_table_size()}")
    
    q_avg = sum(q_learning_rewards[-10:]) / 10
    print(f"\n2.4 强化学习训练完成!")
    print(f"   最后10回合平均奖励: {q_avg:.2f}")
    print(f"   Q表大小: {q_agent.get_q_table_size()}")
    print(f"   相比随机智能体提升: {(q_avg - random_avg) / max(abs(random_avg), 1e-6) * 100:.1f}%")
    
    print("\n" + "-" * 70)
    print("阶段3: 符号化反思和知识提取")
    print("-" * 70)
    
    print("\n3.1 初始化符号化反思器...")
    reflector = SymbolicReflector()
    
    print("\n3.2 从训练经验中提取知识...")
    
    states_list = []
    actions_list = []
    rewards_list = []
    dones_list = []
    
    for episode in range(20):
        observation = environment.reset()
        
        for step in range(15):
            action = q_agent.act(observation)
            if action is None:
                break
            
            states_list.append(observation)
            actions_list.append({"tool": action.tool, "parameters": action.parameters})
            
            result, reward, done, info = environment.execute_tool(
                action.tool, **action.parameters
            )
            
            rewards_list.append(reward)
            dones_list.append(done)
            
            next_observation = environment.get_observation()
            q_agent.observe(observation, action, reward, next_observation, done)
            observation = next_observation
            
            if done:
                break
        
        if dones_list:
            episode_data = {
                "states": states_list[-len(rewards_list) + len(dones_list):],
                "actions": actions_list[-len(rewards_list) + len(dones_list):],
                "rewards": rewards_list[-len(rewards_list) + len(dones_list):],
                "dones": [True]
            }
            reflector.reflect_on_episode(episode_data)
    
    print("\n3.3 知识库统计...")
    kb_stats = reflector.get_reflection_summary()
    print(f"   知识库规则总数: {kb_stats['knowledge_base']['total_rules']}")
    print(f"   高置信度规则数: {kb_stats['high_confidence_rules']}")
    print(f"   已分析回合数: {kb_stats['episodes_reflected']}")
    
    print("\n3.4 改进建议...")
    suggestions = reflector.generate_improvement_suggestions()
    for i, suggestion in enumerate(suggestions[:5]):
        print(f"   {i + 1}. {suggestion['suggestion']}")
    
    print("\n" + "-" * 70)
    print("阶段4: 策略优化和自主规划")
    print("-" * 70)
    
    print("\n4.1 初始化策略优化器...")
    optimizer = StrategyOptimizer(
        method=OptimizationMethod.Q_ITERATION,
        discount_factor=0.95,
        learning_rate=0.1
    )
    
    print("\n4.2 添加训练经验到优化器...")
    for i in range(min(100, len(states_list) - 1)):
        state = states_list[i]
        action = actions_list[i]
        reward = rewards_list[i]
        next_state = states_list[min(i + 1, len(states_list) - 1)]
        done = dones_list[i] if i < len(dones_list) else False
        
        state_key = f"state_{i % 20}"
        next_state_key = f"state_{(i + 1) % 20}"
        
        optimizer.add_experience(
            state_key=state_key,
            action=action["tool"],
            reward=reward,
            next_state_key=next_state_key,
            done=done
        )
    
    print("\n4.3 执行策略优化...")
    optimized_policy = optimizer.optimize(iterations=50)
    
    print(f"   优化后状态数: {len(optimized_policy.state_action_probs)}")
    print(f"   优化方法: {optimizer.method.value}")
    
    print("\n4.4 初始化规划器...")
    planner = ForwardSearchPlanner(search_strategy="astar", max_depth=15)
    
    print("\n4.5 注册动作模型到规划器...")
    
    reward_rules = reflector.knowledge_base.get_rules_by_type(RuleType.REWARD)
    for rule in reward_rules:
        action_name = rule.metadata.get("action", "unknown")
        avg_reward = rule.metadata.get("avg_reward", 0.0)
        
        if action_name != "unknown":
            planner.register_action_model(
                action_name,
                preconditions=rule.conditions,
                effects=rule.conclusions,
                reward=avg_reward
            )
    
    print("\n4.6 执行自主规划任务...")
    print("   目标: 完成购物流程 (从浏览到结账)")
    
    initial_observation = environment.reset()
    goal_state = {"current_state": "terminal"}
    
    available_actions = ["search_products", "add_to_cart", "view_cart", "checkout", "get_product_info"]
    
    plan = planner.plan(
        initial_state=initial_observation,
        goal_state=goal_state,
        available_actions=available_actions
    )
    
    if plan and len(plan) > 0:
        print(f"\n   ✓ 找到自主规划!")
        print(f"   规划步数: {len(plan)}")
        print(f"   预计总奖励: {plan.estimated_total_reward:.2f}")
        print("\n   规划步骤:")
        for i, step in enumerate(plan.steps):
            print(f"   {i + 1}. {step.action} (预计奖励: {step.estimated_reward:.2f})")
    else:
        print("\n   ✗ 未找到完整规划，使用启发式策略...")
    
    print("\n" + "-" * 70)
    print("阶段5: 效果对比评估")
    print("-" * 70)
    
    print("\n5.1 评估各阶段智能体性能...")
    
    print("\n   评估配置: 10个回合, 每回合最大20步")
    
    def evaluate_agent(agent, env, episodes=10):
        rewards = []
        steps_list = []
        
        for episode in range(episodes):
            observation = env.reset()
            episode_reward = 0.0
            steps = 0
            
            for step in range(20):
                action = agent.act(observation)
                if action is None:
                    break
                
                result, reward, done, info = env.execute_tool(
                    action.tool, **action.parameters
                )
                
                episode_reward += reward
                steps += 1
                
                next_observation = env.get_observation()
                agent.observe(observation, action, reward, next_observation, done)
                observation = next_observation
                
                if done:
                    break
            
            rewards.append(episode_reward)
            steps_list.append(steps)
        
        return {
            "avg_reward": sum(rewards) / len(rewards) if rewards else 0,
            "avg_steps": sum(steps_list) / len(steps_list) if steps_list else 0,
            "max_reward": max(rewards) if rewards else 0,
            "min_reward": min(rewards) if rewards else 0
        }
    
    print("\n   评估随机智能体...")
    random_stats = evaluate_agent(random_agent, environment, episodes=10)
    print(f"      平均奖励: {random_stats['avg_reward']:.2f}")
    print(f"      平均步数: {random_stats['avg_steps']:.1f}")
    
    print("\n   评估Q学习智能体...")
    q_stats = evaluate_agent(q_agent, environment, episodes=10)
    print(f"      平均奖励: {q_stats['avg_reward']:.2f}")
    print(f"      平均步数: {q_stats['avg_steps']:.1f}")
    
    print("\n5.2 性能提升分析...")
    
    reward_improvement = q_stats['avg_reward'] - random_stats['avg_reward']
    improvement_percent = (reward_improvement / max(abs(random_stats['avg_reward']), 1e-6)) * 100
    
    print(f"\n   奖励提升: {reward_improvement:.2f} ({improvement_percent:.1f}%)")
    print(f"   步数减少: {random_stats['avg_steps'] - q_stats['avg_steps']:.1f} 步")
    
    print("\n" + "=" * 70)
    print("训练流程演示完成")
    print("=" * 70)
    
    print("\n总结:")
    print("  1. 被动执行阶段: 随机选择动作，无学习能力")
    print("  2. 强化学习阶段: 从经验中学习Q值，改进决策")
    print("  3. 符号化反思阶段: 提取可解释的规则和模式")
    print("  4. 自主规划阶段: 基于知识库进行前瞻性规划")
    print(f"\n  性能提升: 奖励提升 {improvement_percent:.1f}%")
    
    return {
        "environment_id": env_instance.id,
        "random_agent_stats": random_stats,
        "q_agent_stats": q_stats,
        "knowledge_base_size": kb_stats['knowledge_base']['total_rules'],
        "improvement_percent": improvement_percent,
        "plan_found": plan is not None and len(plan) > 0
    }


def demonstrate_full_pipeline():
    """
    演示完整的端到端训练流程
    """
    print("\n" + "=" * 70)
    print("端到端训练流程演示")
    print("=" * 70)
    
    from src.reflection.planning import ReactivePlanner
    from src.reflection.symbolic_reflection import Rule, RuleType
    
    env_manager = EnvironmentManager()
    env_instance = env_manager.create_ecommerce_environment()
    environment = env_manager.load_environment(env_instance)
    
    print("\n1. 创建反应式规划器...")
    reactive_planner = ReactivePlanner()
    
    print("\n2. 添加业务规则...")
    
    reactive_planner.add_rule(
        condition='state.get("current_state") == "initial"',
        action="search_products",
        priority=3,
        parameters={"query": "Laptop"}
    )
    
    reactive_planner.add_rule(
        condition='state.get("current_state") == "browsing"',
        action="add_to_cart",
        priority=2,
        parameters={"product_id": "p1", "quantity": 1}
    )
    
    reactive_planner.add_rule(
        condition='state.get("current_state") == "cart_active"',
        action="checkout",
        priority=1,
        parameters={"payment_method": "credit_card"}
    )
    
    print("\n3. 执行反应式规划...")
    print("   使用规则触发的即时决策")
    
    observation = environment.reset()
    goal_state = {"current_state": "terminal"}
    
    for step in range(10):
        plan = reactive_planner.plan(
            initial_state=observation,
            goal_state=goal_state,
            available_actions=["search_products", "add_to_cart", "view_cart", "checkout", "get_product_info"]
        )
        
        if plan and len(plan) > 0:
            step_action = plan.steps[0]
            result, reward, done, info = environment.execute_tool(
                step_action.tool, **step_action.parameters
            )
            
            print(f"   步骤 {step + 1}: {step_action.tool} -> 奖励={reward:.2f}")
            print(f"      推理: {step_action.description}")
            
            if done:
                print("\n   ✓ 任务完成!")
                break
            
            observation = environment.get_observation()
        else:
            print(f"   步骤 {step + 1}: 无匹配规则")
            break
    
    print("\n" + "=" * 70)
    print("端到端演示完成")
    print("=" * 70)
    
    return {
        "environment_id": env_instance.id,
        "demo_completed": True
    }


if __name__ == "__main__":
    run_training_pipeline()
    demonstrate_full_pipeline()
