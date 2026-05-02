"""
规划模块 - 实现智能体的自主规划能力
"""

import heapq
from typing import Dict, List, Optional, Any, Tuple, Callable, Set
from dataclasses import dataclass, field
from collections import deque, defaultdict
from abc import ABC, abstractmethod


@dataclass
class PlanStep:
    action: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    preconditions: List[str] = field(default_factory=list)
    effects: List[str] = field(default_factory=list)
    estimated_reward: float = 0.0
    confidence: float = 1.0
    description: str = ""


@dataclass
class Plan:
    steps: List[PlanStep] = field(default_factory=list)
    goal_state: Dict[str, Any] = field(default_factory=dict)
    estimated_total_reward: float = 0.0
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __len__(self) -> int:
        return len(self.steps)
    
    def get_step(self, index: int) -> Optional[PlanStep]:
        if 0 <= index < len(self.steps):
            return self.steps[index]
        return None
    
    def add_step(self, step: PlanStep):
        self.steps.append(step)
        self.estimated_total_reward += step.estimated_reward
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "steps": [
                {
                    "action": s.action,
                    "parameters": s.parameters,
                    "preconditions": s.preconditions,
                    "effects": s.effects,
                    "estimated_reward": s.estimated_reward,
                    "confidence": s.confidence,
                    "description": s.description
                }
                for s in self.steps
            ],
            "goal_state": self.goal_state,
            "estimated_total_reward": self.estimated_total_reward,
            "confidence": self.confidence
        }


class Planner(ABC):
    """
    规划器基类
    """
    
    @abstractmethod
    def plan(
        self,
        initial_state: Dict[str, Any],
        goal_state: Dict[str, Any],
        available_actions: List[str],
        knowledge_base=None
    ) -> Optional[Plan]:
        pass


class ForwardSearchPlanner(Planner):
    """
    前向搜索规划器
    
    核心特点：
    1. 从初始状态向前搜索到目标状态
    2. 使用启发式函数引导搜索
    3. 支持多种搜索策略（BFS、DFS、A*）
    """
    
    def __init__(self, search_strategy: str = "astar", max_depth: int = 20):
        self.search_strategy = search_strategy
        self.max_depth = max_depth
        self.action_models: Dict[str, Dict] = {}
    
    def register_action_model(
        self,
        action_name: str,
        preconditions: List[str],
        effects: List[str],
        reward: float = 0.0
    ):
        self.action_models[action_name] = {
            "preconditions": preconditions,
            "effects": effects,
            "reward": reward
        }
    
    def plan(
        self,
        initial_state: Dict[str, Any],
        goal_state: Dict[str, Any],
        available_actions: List[str],
        knowledge_base=None
    ) -> Optional[Plan]:
        if self._is_goal_state(initial_state, goal_state):
            return Plan(goal_state=goal_state, confidence=1.0)
        
        if self.search_strategy == "bfs":
            return self._bfs_search(initial_state, goal_state, available_actions, knowledge_base)
        elif self.search_strategy == "astar":
            return self._astar_search(initial_state, goal_state, available_actions, knowledge_base)
        else:
            return self._greedy_search(initial_state, goal_state, available_actions, knowledge_base)
    
    def _is_goal_state(self, state: Dict[str, Any], goal: Dict[str, Any]) -> bool:
        state_vars = state.get("state_variables", {}) if "state_variables" in state else state
        
        for key, value in goal.items():
            if key not in state_vars or state_vars[key] != value:
                return False
        
        return True
    
    def _apply_action(
        self,
        state: Dict[str, Any],
        action_name: str,
        knowledge_base=None
    ) -> Tuple[Dict[str, Any], float]:
        new_state = {
            "state_variables": state.get("state_variables", {}).copy(),
            "current_state": state.get("current_state")
        }
        
        reward = 0.0
        
        if action_name in self.action_models:
            model = self.action_models[action_name]
            reward = model.get("reward", 0.0)
            
            for effect in model.get("effects", []):
                try:
                    exec(effect, {"state": new_state["state_variables"]})
                except Exception:
                    pass
        
        return new_state, reward
    
    def _bfs_search(
        self,
        initial_state: Dict[str, Any],
        goal_state: Dict[str, Any],
        available_actions: List[str],
        knowledge_base=None
    ) -> Optional[Plan]:
        queue = deque()
        queue.append((initial_state, []))
        
        visited = set()
        state_hash = self._hash_state(initial_state)
        visited.add(state_hash)
        
        while queue:
            current_state, path = queue.popleft()
            
            if len(path) >= self.max_depth:
                continue
            
            for action in available_actions:
                next_state, reward = self._apply_action(current_state, action, knowledge_base)
                next_state_hash = self._hash_state(next_state)
                
                if next_state_hash not in visited:
                    visited.add(next_state_hash)
                    new_path = path + [(action, reward)]
                    
                    if self._is_goal_state(next_state, goal_state):
                        return self._path_to_plan(new_path, goal_state)
                    
                    queue.append((next_state, new_path))
        
        return None
    
    def _astar_search(
        self,
        initial_state: Dict[str, Any],
        goal_state: Dict[str, Any],
        available_actions: List[str],
        knowledge_base=None
    ) -> Optional[Plan]:
        priority_queue = []
        initial_heuristic = self._heuristic(initial_state, goal_state)
        heapq.heappush(priority_queue, (initial_heuristic, 0, initial_state, []))
        
        visited = {}
        state_hash = self._hash_state(initial_state)
        visited[state_hash] = 0
        
        while priority_queue:
            f_value, g_value, current_state, path = heapq.heappop(priority_queue)
            
            if len(path) >= self.max_depth:
                continue
            
            current_hash = self._hash_state(current_state)
            if visited.get(current_hash, float('inf')) < g_value:
                continue
            
            if self._is_goal_state(current_state, goal_state):
                return self._path_to_plan(path, goal_state)
            
            for action in available_actions:
                next_state, reward = self._apply_action(current_state, action, knowledge_base)
                next_g = g_value + 1 - reward
                next_h = self._heuristic(next_state, goal_state)
                next_f = next_g + next_h
                
                next_hash = self._hash_state(next_state)
                if next_hash not in visited or next_g < visited[next_hash]:
                    visited[next_hash] = next_g
                    new_path = path + [(action, reward)]
                    heapq.heappush(priority_queue, (next_f, next_g, next_state, new_path))
        
        return None
    
    def _greedy_search(
        self,
        initial_state: Dict[str, Any],
        goal_state: Dict[str, Any],
        available_actions: List[str],
        knowledge_base=None
    ) -> Optional[Plan]:
        current_state = initial_state
        path = []
        
        for _ in range(self.max_depth):
            if self._is_goal_state(current_state, goal_state):
                return self._path_to_plan(path, goal_state)
            
            best_action = None
            best_heuristic = float('inf')
            best_next_state = None
            best_reward = 0.0
            
            for action in available_actions:
                next_state, reward = self._apply_action(current_state, action, knowledge_base)
                heuristic = self._heuristic(next_state, goal_state)
                
                adjusted_heuristic = heuristic - reward
                
                if adjusted_heuristic < best_heuristic:
                    best_heuristic = adjusted_heuristic
                    best_action = action
                    best_next_state = next_state
                    best_reward = reward
            
            if best_action is None:
                break
            
            path.append((best_action, best_reward))
            current_state = best_next_state
        
        if self._is_goal_state(current_state, goal_state):
            return self._path_to_plan(path, goal_state)
        
        return None
    
    def _heuristic(self, state: Dict[str, Any], goal: Dict[str, Any]) -> float:
        state_vars = state.get("state_variables", {}) if "state_variables" in state else state
        
        distance = 0
        for key, target_value in goal.items():
            current_value = state_vars.get(key)
            if current_value != target_value:
                if isinstance(target_value, (int, float)) and isinstance(current_value, (int, float)):
                    distance += abs(target_value - current_value)
                else:
                    distance += 1
        
        return distance
    
    def _hash_state(self, state: Dict[str, Any]) -> str:
        import hashlib
        import json
        
        state_vars = state.get("state_variables", {}) if "state_variables" in state else state
        current_state_val = state.get("current_state", "")
        
        hash_input = json.dumps({
            "state_variables": state_vars,
            "current_state": current_state_val
        }, sort_keys=True)
        
        return hashlib.md5(hash_input.encode()).hexdigest()
    
    def _path_to_plan(self, path: List[Tuple[str, float]], goal_state: Dict[str, Any]) -> Plan:
        plan = Plan(goal_state=goal_state)
        
        for action, reward in path:
            step = PlanStep(
                action=action,
                estimated_reward=reward,
                confidence=1.0
            )
            plan.add_step(step)
        
        return plan


class HierarchicalPlanner(Planner):
    """
    层次规划器
    
    核心特点：
    1. 将复杂任务分解为子任务
    2. 使用高层策略进行抽象规划
    3. 在执行层进行详细规划
    """
    
    def __init__(self):
        self.subtask_decomposers: Dict[str, Callable] = {}
        self.abstract_actions: Dict[str, List[str]] = {}
    
    def register_abstract_action(
        self,
        abstract_name: str,
        primitive_actions: List[str],
        decomposer: Optional[Callable] = None
    ):
        self.abstract_actions[abstract_name] = primitive_actions
        if decomposer:
            self.subtask_decomposers[abstract_name] = decomposer
    
    def plan(
        self,
        initial_state: Dict[str, Any],
        goal_state: Dict[str, Any],
        available_actions: List[str],
        knowledge_base=None
    ) -> Optional[Plan]:
        abstract_plan = self._abstract_plan(initial_state, goal_state, available_actions, knowledge_base)
        
        if abstract_plan is None:
            return None
        
        concrete_plan = self._refine_plan(abstract_plan, initial_state, goal_state, knowledge_base)
        
        return concrete_plan
    
    def _abstract_plan(
        self,
        initial_state: Dict[str, Any],
        goal_state: Dict[str, Any],
        available_actions: List[str],
        knowledge_base=None
    ) -> Optional[Plan]:
        all_actions = available_actions + list(self.abstract_actions.keys())
        
        planner = ForwardSearchPlanner(search_strategy="astar", max_depth=10)
        
        for abstract_name, primitives in self.abstract_actions.items():
            planner.register_action_model(
                abstract_name,
                preconditions=[],
                effects=[f'state["_{abstract_name}_executed"] = True'],
                reward=0.5
            )
        
        return planner.plan(initial_state, goal_state, all_actions, knowledge_base)
    
    def _refine_plan(
        self,
        abstract_plan: Plan,
        initial_state: Dict[str, Any],
        goal_state: Dict[str, Any],
        knowledge_base=None
    ) -> Plan:
        concrete_plan = Plan(goal_state=goal_state)
        current_state = initial_state
        
        for abstract_step in abstract_plan.steps:
            action_name = abstract_step.action
            
            if action_name in self.abstract_actions:
                if action_name in self.subtask_decomposers:
                    subtask_goal = self.subtask_decomposers[action_name](current_state)
                    
                    planner = ForwardSearchPlanner(search_strategy="astar", max_depth=5)
                    subtask_plan = planner.plan(
                        current_state,
                        subtask_goal,
                        self.abstract_actions[action_name],
                        knowledge_base
                    )
                    
                    if subtask_plan:
                        for step in subtask_plan.steps:
                            concrete_plan.add_step(step)
                else:
                    for primitive in self.abstract_actions[action_name]:
                        step = PlanStep(
                            action=primitive,
                            estimated_reward=0.1,
                            confidence=0.8
                        )
                        concrete_plan.add_step(step)
            else:
                concrete_plan.add_step(abstract_step)
        
        return concrete_plan


class ReactivePlanner(Planner):
    """
    反应式规划器
    
    核心特点：
    1. 基于规则的即时决策
    2. 不需要完整的状态空间搜索
    3. 适合动态环境
    """
    
    def __init__(self):
        self.rules: List[Dict] = []
    
    def add_rule(
        self,
        condition: str,
        action: str,
        priority: int = 0,
        parameters: Dict = None
    ):
        self.rules.append({
            "condition": condition,
            "action": action,
            "priority": priority,
            "parameters": parameters or {}
        })
        self.rules.sort(key=lambda x: x["priority"], reverse=True)
    
    def plan(
        self,
        initial_state: Dict[str, Any],
        goal_state: Dict[str, Any],
        available_actions: List[str],
        knowledge_base=None
    ) -> Optional[Plan]:
        for rule in self.rules:
            try:
                namespace = {
                    "state": initial_state.get("state_variables", {}),
                    "current_state": initial_state.get("current_state"),
                    "goal": goal_state
                }
                
                if eval(rule["condition"], namespace):
                    action = rule["action"]
                    
                    if action in available_actions:
                        plan = Plan(goal_state=goal_state)
                        plan.add_step(PlanStep(
                            action=action,
                            parameters=rule["parameters"],
                            confidence=1.0,
                            description=f"规则触发: {rule['condition']}"
                        ))
                        return plan
            except Exception:
                continue
        
        return None
