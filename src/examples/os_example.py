"""
操作系统环境示例 - 演示操作系统模拟环境的使用
"""

import json
from typing import Dict, List, Optional, Any

from src.sandbox.environment_manager import EnvironmentManager
from src.sandbox.environment_executor import EnvironmentExecutor
from src.rl.agent import QLearningAgent, Action, RandomAgent, HeuristicAgent


def run_os_demo():
    """
    运行操作系统环境演示
    """
    print("=" * 60)
    print("操作系统环境演示")
    print("=" * 60)
    
    env_manager = EnvironmentManager()
    
    print("\n1. 创建操作系统环境...")
    env_instance = env_manager.create_os_environment(
        name="模拟Linux环境",
        user="developer"
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
    print(f"   当前用户: {observation['state_variables'].get('user')}")
    
    print("\n4. 演示文件系统操作...")
    print("-" * 40)
    
    print("\n   步骤1: 列出当前目录")
    result, reward, done, info = environment.execute_tool(
        "list_directory",
        path="home/user"
    )
    print(f"   结果: {json.dumps(result, indent=2, ensure_ascii=False)}")
    print(f"   奖励: {reward}")
    
    print("\n   步骤2: 读取文件")
    result, reward, done, info = environment.execute_tool(
        "read_file",
        path="home/user/documents/readme.txt"
    )
    print(f"   结果: {json.dumps(result, indent=2, ensure_ascii=False)}")
    print(f"   奖励: {reward}")
    
    print("\n   步骤3: 创建新目录")
    result, reward, done, info = environment.execute_tool(
        "create_directory",
        path="home/user/projects/new_folder"
    )
    print(f"   结果: {json.dumps(result, indent=2, ensure_ascii=False)}")
    print(f"   奖励: {reward}")
    
    print("\n   步骤4: 写入文件")
    result, reward, done, info = environment.execute_tool(
        "write_file",
        path="home/user/projects/new_folder/hello.py",
        content="print('Hello, World!')\n"
    )
    print(f"   结果: {json.dumps(result, indent=2, ensure_ascii=False)}")
    print(f"   奖励: {reward}")
    
    print("\n   步骤5: 验证文件内容")
    result, reward, done, info = environment.execute_tool(
        "read_file",
        path="home/user/projects/new_folder/hello.py"
    )
    print(f"   结果: {json.dumps(result, indent=2, ensure_ascii=False)}")
    print(f"   奖励: {reward}")
    
    print("\n   步骤6: 执行命令")
    result, reward, done, info = environment.execute_tool(
        "execute_command",
        command="echo 'Testing command execution'"
    )
    print(f"   结果: {json.dumps(result, indent=2, ensure_ascii=False)}")
    print(f"   奖励: {reward}")
    
    print("\n5. 演示启发式智能体...")
    print("-" * 40)
    
    environment.reset()
    heuristic_agent = HeuristicAgent("HeuristicOSAgent")
    
    heuristic_agent.add_rule(
        condition='len(state.get("command_history", [])) < 3',
        action="list_directory",
        priority=1,
        parameters={"path": "home/user"}
    )
    
    heuristic_agent.add_rule(
        condition='state.get("user") == "developer"',
        action="read_file",
        priority=2,
        parameters={"path": "home/user/documents/readme.txt"}
    )
    
    print("   运行启发式智能体:")
    observation = environment.reset()
    for step in range(5):
        action = heuristic_agent.act(observation)
        if action is None:
            print("   无可用动作")
            break
        
        result, reward, done, info = environment.execute_tool(
            action.tool, **action.parameters
        )
        
        print(f"   步骤 {step + 1}: {action.tool} -> 奖励={reward:.2f}")
        print(f"      推理: {action.reasoning}")
        
        next_observation = environment.get_observation()
        heuristic_agent.observe(observation, action, reward, next_observation, done)
        observation = next_observation
        
        if done:
            break
    
    print("\n6. 演示符号化反思...")
    print("-" * 40)
    
    from src.reflection.symbolic_reflection import SymbolicReflector, KnowledgeBase, Rule, RuleType
    
    reflector = SymbolicReflector()
    
    print("   模拟经验数据进行反思...")
    
    episode_data = {
        "states": [
            {"state_variables": {"user": "developer", "filesystem": {}}, "current_state": "initial"},
            {"state_variables": {"user": "developer", "filesystem": {}}, "current_state": "active"},
            {"state_variables": {"user": "developer", "filesystem": {}}, "current_state": "active"}
        ],
        "actions": [
            {"tool": "list_directory", "parameters": {"path": "home/user"}},
            {"tool": "read_file", "parameters": {"path": "home/user/documents/readme.txt"}},
            {"tool": "write_file", "parameters": {"path": "home/user/test.txt", "content": "test"}}
        ],
        "rewards": [0.5, 0.5, -0.5],
        "dones": [False, False, True]
    }
    
    reflector.reflect_on_episode(episode_data)
    
    print("   反思结果:")
    summary = reflector.get_reflection_summary()
    print(f"   - 知识库规则数: {summary['knowledge_base']['total_rules']}")
    print(f"   - 高置信度规则数: {summary['high_confidence_rules']}")
    print(f"   - 反思回合数: {summary['episodes_reflected']}")
    
    print("\n   改进建议:")
    suggestions = reflector.generate_improvement_suggestions()
    for i, suggestion in enumerate(suggestions):
        print(f"   {i + 1}. {suggestion['suggestion']}")
    
    print("\n7. 智能体统计:")
    print("-" * 40)
    stats = heuristic_agent.get_stats()
    print(f"   智能体名称: {stats['name']}")
    print(f"   总步数: {stats['total_steps']}")
    print(f"   回合数: {stats['episode_count']}")
    print(f"   平均奖励: {stats['average_reward']:.2f}")
    
    print("\n" + "=" * 60)
    print("操作系统环境演示完成")
    print("=" * 60)
    
    return {
        "environment_id": env_instance.id,
        "agent_stats": stats,
        "reflection_summary": summary
    }


def demonstrate_os_navigation():
    """
    演示操作系统环境中的导航任务
    """
    print("\n" + "=" * 60)
    print("操作系统导航任务演示")
    print("=" * 60)
    
    from src.reflection.planning import HierarchicalPlanner, ForwardSearchPlanner, PlanStep, Plan
    
    env_manager = EnvironmentManager()
    env_instance = env_manager.create_os_environment()
    environment = env_manager.load_environment(env_instance)
    
    print("\n1. 初始化层次规划器...")
    hierarchical_planner = HierarchicalPlanner()
    
    print("\n2. 注册抽象动作...")
    
    def navigate_to_docs_decomposer(state):
        return {"current_state": "documents_visited"}
    
    def read_file_decomposer(state):
        return {"current_state": "file_read"}
    
    hierarchical_planner.register_abstract_action(
        "navigate_to_documents",
        ["list_directory", "read_file"],
        navigate_to_docs_decomposer
    )
    
    hierarchical_planner.register_abstract_action(
        "create_and_write_file",
        ["create_directory", "write_file"],
        read_file_decomposer
    )
    
    print("\n3. 定义导航任务...")
    initial_observation = environment.get_observation()
    goal_state = {"current_state": "file_read"}
    
    print(f"   初始状态: {initial_observation['current_state']}")
    print(f"   目标状态: {goal_state}")
    
    print("\n4. 执行层次规划...")
    available_actions = ["list_directory", "read_file", "write_file", "create_directory", "delete_file"]
    
    plan = hierarchical_planner.plan(
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
    print("导航任务演示完成")
    print("=" * 60)
    
    return plan


if __name__ == "__main__":
    run_os_demo()
    demonstrate_os_navigation()
