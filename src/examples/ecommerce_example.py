"""
电商环境示例 - 演示电商模拟环境的使用
"""

import json
from typing import Dict, List, Optional, Any

from src.sandbox.environment_manager import EnvironmentManager
from src.sandbox.environment_executor import EnvironmentExecutor
from src.rl.agent import QLearningAgent, Action, RandomAgent


def run_ecommerce_demo():
    """
    运行电商环境演示
    """
    print("=" * 60)
    print("电商环境演示")
    print("=" * 60)
    
    env_manager = EnvironmentManager()
    
    print("\n1. 创建电商环境...")
    env_instance = env_manager.create_ecommerce_environment(
        name="智能购物环境",
        user_balance=1000.0
    )
    
    print(f"   环境ID: {env_instance.id}")
    print(f"   环境名称: {env_instance.name}")
    print(f"   环境类型: {env_instance.type}")
    
    print("\n2. 加载环境实例...")
    environment = env_manager.load_environment(env_instance)
    
    print("\n3. 获取环境初始状态...")
    observation = environment.get_observation()
    print(f"   当前状态: {observation['current_state']}")
    print(f"   可用工具: {observation['available_tools']}")
    print(f"   用户余额: {observation['state_variables'].get('user_balance')}")
    
    print("\n4. 演示智能体交互...")
    print("-" * 40)
    
    print("\n   步骤1: 搜索产品")
    result, reward, done, info = environment.execute_tool(
        "search_products",
        query="Laptop"
    )
    print(f"   结果: {json.dumps(result, indent=2, ensure_ascii=False)}")
    print(f"   奖励: {reward}")
    
    print("\n   步骤2: 查看产品信息")
    result, reward, done, info = environment.execute_tool(
        "get_product_info",
        product_id="p1"
    )
    print(f"   结果: {json.dumps(result, indent=2, ensure_ascii=False)}")
    print(f"   奖励: {reward}")
    
    print("\n   步骤3: 添加到购物车")
    result, reward, done, info = environment.execute_tool(
        "add_to_cart",
        product_id="p1",
        quantity=1
    )
    print(f"   结果: {json.dumps(result, indent=2, ensure_ascii=False)}")
    print(f"   奖励: {reward}")
    
    print("\n   步骤4: 查看购物车")
    result, reward, done, info = environment.execute_tool("view_cart")
    print(f"   结果: {json.dumps(result, indent=2, ensure_ascii=False)}")
    print(f"   奖励: {reward}")
    
    print("\n   步骤5: 结账")
    result, reward, done, info = environment.execute_tool(
        "checkout",
        payment_method="credit_card"
    )
    print(f"   结果: {json.dumps(result, indent=2, ensure_ascii=False)}")
    print(f"   奖励: {reward}")
    print(f"   完成状态: {done}")
    
    print("\n5. 演示随机智能体...")
    print("-" * 40)
    
    environment.reset()
    random_agent = RandomAgent("RandomShopper")
    executor = EnvironmentExecutor(timeout=5.0)
    
    print("   运行随机智能体5个回合:")
    for episode in range(5):
        observation = environment.reset()
        episode_reward = 0.0
        steps = 0
        
        for step in range(10):
            action = random_agent.act(observation)
            if action is None:
                break
            
            result, reward, done, info = executor.execute_tool(
                environment, action.tool, action.parameters
            )
            
            episode_reward += reward
            steps += 1
            
            observation = environment.get_observation()
            random_agent.observe(observation, action, reward, observation, done)
            
            if done:
                break
        
        print(f"   回合 {episode + 1}: 奖励={episode_reward:.2f}, 步数={steps}")
    
    print("\n6. 演示Q学习智能体...")
    print("-" * 40)
    
    environment.reset()
    q_agent = QLearningAgent(
        name="QLearningShopper",
        learning_rate=0.1,
        discount_factor=0.99,
        epsilon=0.5
    )
    
    print("   训练Q学习智能体20个回合:")
    for episode in range(20):
        observation = environment.reset()
        episode_reward = 0.0
        steps = 0
        
        for step in range(10):
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
        
        if (episode + 1) % 5 == 0:
            print(f"   回合 {episode + 1}: 奖励={episode_reward:.2f}, epsilon={q_agent.epsilon:.4f}")
    
    print("\n7. 智能体统计:")
    print("-" * 40)
    stats = q_agent.get_stats()
    print(f"   智能体名称: {stats['name']}")
    print(f"   总步数: {stats['total_steps']}")
    print(f"   回合数: {stats['episode_count']}")
    print(f"   平均奖励: {stats['average_reward']:.2f}")
    print(f"   Q表大小: {q_agent.get_q_table_size()}")
    
    print("\n" + "=" * 60)
    print("电商环境演示完成")
    print("=" * 60)
    
    return {
        "environment_id": env_instance.id,
        "agent_stats": stats,
        "q_table_size": q_agent.get_q_table_size()
    }


def demonstrate_ecommerce_planning():
    """
    演示电商环境中的规划能力
    """
    print("\n" + "=" * 60)
    print("电商环境规划演示")
    print("=" * 60)
    
    from src.reflection.planning import ForwardSearchPlanner, PlanStep, Plan
    
    env_manager = EnvironmentManager()
    env_instance = env_manager.create_ecommerce_environment()
    environment = env_manager.load_environment(env_instance)
    
    print("\n1. 初始化规划器...")
    planner = ForwardSearchPlanner(search_strategy="astar", max_depth=10)
    
    print("\n2. 注册动作模型...")
    planner.register_action_model(
        "search_products",
        preconditions=['state["current_state"] == "initial"'],
        effects=['state["current_state"] = "browsing"'],
        reward=0.1
    )
    planner.register_action_model(
        "add_to_cart",
        preconditions=['state["current_state"] == "browsing"'],
        effects=['state["current_state"] = "cart_active"'],
        reward=1.0
    )
    planner.register_action_model(
        "checkout",
        preconditions=['state["current_state"] == "cart_active"'],
        effects=['state["current_state"] = "terminal"'],
        reward=10.0
    )
    
    print("\n3. 定义目标状态...")
    initial_observation = environment.get_observation()
    goal_state = {"current_state": "terminal"}
    
    print(f"   初始状态: {initial_observation['current_state']}")
    print(f"   目标状态: {goal_state}")
    
    print("\n4. 执行规划搜索...")
    available_actions = ["search_products", "add_to_cart", "checkout", "view_cart", "get_product_info"]
    
    plan = planner.plan(
        initial_state=initial_observation,
        goal_state=goal_state,
        available_actions=available_actions
    )
    
    if plan:
        print(f"   找到规划! 步数: {len(plan)}")
        print(f"   预计总奖励: {plan.estimated_total_reward:.2f}")
        print("\n   规划步骤:")
        for i, step in enumerate(plan.steps):
            print(f"   {i + 1}. {step.action} (预计奖励: {step.estimated_reward:.2f})")
    else:
        print("   未找到可行规划")
    
    print("\n" + "=" * 60)
    print("规划演示完成")
    print("=" * 60)
    
    return plan


if __name__ == "__main__":
    run_ecommerce_demo()
    demonstrate_ecommerce_planning()
