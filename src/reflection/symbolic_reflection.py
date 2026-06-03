"""
符号化反思算法 - 实现智能体的自我反思和知识提取
"""

import re
import json
from typing import Dict, List, Optional, Any, Tuple, Set, Callable
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum
from abc import ABC, abstractmethod


class RuleType(Enum):
    PRECONDITION = "precondition"
    EFFECT = "effect"
    TRANSITION = "transition"
    REWARD = "reward"
    CONSTRAINT = "constraint"


@dataclass
class Rule:
    name: str
    rule_type: RuleType
    conditions: List[str]
    conclusions: List[str]
    confidence: float = 1.0
    support: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_string(self) -> str:
        conditions_str = " ∧ ".join(self.conditions)
        conclusions_str = " ∧ ".join(self.conclusions)
        return f"({conditions_str}) → ({conclusions_str})"
    
    def matches(self, state: Dict[str, Any]) -> bool:
        for condition in self.conditions:
            if not self._evaluate_condition(condition, state):
                return False
        return True
    
    def _evaluate_condition(self, condition: str, state: Dict[str, Any]) -> bool:
        try:
            namespace = {"state": state}
            return eval(condition, namespace)
        except Exception:
            return False


class KnowledgeBase:
    """
    知识库 - 存储智能体学习到的符号化知识
    
    核心功能：
    1. 存储规则和事实
    2. 支持规则的增删改查
    3. 提供推理接口
    """
    
    def __init__(self):
        self.rules: Dict[str, Rule] = {}
        self.facts: Dict[str, Any] = {}
        self.rule_index: Dict[str, List[str]] = defaultdict(list)
    
    def add_rule(self, rule: Rule):
        self.rules[rule.name] = rule
        
        for condition in rule.conditions:
            self.rule_index[condition].append(rule.name)
        
        for conclusion in rule.conclusions:
            self.rule_index[conclusion].append(rule.name)
    
    def remove_rule(self, rule_name: str):
        if rule_name in self.rules:
            rule = self.rules[rule_name]
            
            for condition in rule.conditions:
                if rule_name in self.rule_index.get(condition, []):
                    self.rule_index[condition].remove(rule_name)
            
            for conclusion in rule.conclusions:
                if rule_name in self.rule_index.get(conclusion, []):
                    self.rule_index[conclusion].remove(rule_name)
            
            del self.rules[rule_name]
    
    def get_rules_by_type(self, rule_type: RuleType) -> List[Rule]:
        return [r for r in self.rules.values() if r.rule_type == rule_type]
    
    def get_rules_for_state(self, state: Dict[str, Any]) -> List[Rule]:
        matching_rules = []
        for rule in self.rules.values():
            if rule.matches(state):
                matching_rules.append(rule)
        return matching_rules
    
    def add_fact(self, key: str, value: Any):
        self.facts[key] = value
    
    def get_fact(self, key: str) -> Optional[Any]:
        return self.facts.get(key)
    
    def infer(self, initial_state: Dict[str, Any]) -> Dict[str, Any]:
        inferred = initial_state.copy()
        changed = True
        
        while changed:
            changed = False
            for rule in self.rules.values():
                if rule.matches(inferred):
                    for conclusion in rule.conclusions:
                        try:
                            result = eval(conclusion, {"state": inferred})
                            if isinstance(result, dict):
                                for key, value in result.items():
                                    if key not in inferred or inferred[key] != value:
                                        inferred[key] = value
                                        changed = True
                        except Exception:
                            pass
        
        return inferred
    
    def get_stats(self) -> Dict[str, Any]:
        type_counts = defaultdict(int)
        for rule in self.rules.values():
            type_counts[rule.rule_type.value] += 1
        
        return {
            "total_rules": len(self.rules),
            "rules_by_type": dict(type_counts),
            "total_facts": len(self.facts)
        }
    
    def save(self, filepath: str):
        data = {
            "rules": [
                {
                    "name": r.name,
                    "rule_type": r.rule_type.value,
                    "conditions": r.conditions,
                    "conclusions": r.conclusions,
                    "confidence": r.confidence,
                    "support": r.support,
                    "metadata": r.metadata
                }
                for r in self.rules.values()
            ],
            "facts": self.facts
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def load(self, filepath: str):
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.rules.clear()
        self.rule_index.clear()
        
        for rule_data in data.get("rules", []):
            rule = Rule(
                name=rule_data["name"],
                rule_type=RuleType(rule_data["rule_type"]),
                conditions=rule_data["conditions"],
                conclusions=rule_data["conclusions"],
                confidence=rule_data.get("confidence", 1.0),
                support=rule_data.get("support", 0),
                metadata=rule_data.get("metadata", {})
            )
            self.add_rule(rule)
        
        self.facts = data.get("facts", {})


class SymbolicReflector:
    """
    符号化反思器
    
    核心功能：
    1. 从经验中提取符号化规则
    2. 分析行为模式和规律
    3. 生成改进建议
    4. 支持从"被动执行"到"自主规划"的进化
    """
    
    def __init__(self, knowledge_base: KnowledgeBase = None):
        self.knowledge_base = knowledge_base or KnowledgeBase()
        self.experience_history: List[Dict] = []
        self.learning_rate = 0.1
        self.min_support = 3
        self.min_confidence = 0.6
    
    def reflect_on_episode(self, episode_data: Dict[str, Any]):
        """
        对一个完整的回合进行反思
        
        Args:
            episode_data: 包含以下字段的字典
                - states: 状态序列
                - actions: 动作序列
                - rewards: 奖励序列
                - dones: 终止标志序列
        """
        self.experience_history.append(episode_data)
        
        self._extract_precondition_rules(episode_data)
        self._extract_effect_rules(episode_data)
        self._extract_transition_rules(episode_data)
        self._extract_reward_patterns(episode_data)
    
    def _extract_precondition_rules(self, episode_data: Dict):
        states = episode_data.get("states", [])
        actions = episode_data.get("actions", [])
        rewards = episode_data.get("rewards", [])
        
        for i, (state, action, reward) in enumerate(zip(states, actions, rewards)):
            if reward > 0:
                action_name = action.get("tool") if isinstance(action, dict) else action
                
                conditions = self._state_to_conditions(state)
                
                rule_name = f"precondition_{action_name}_{len(self.knowledge_base.rules)}"
                rule = Rule(
                    name=rule_name,
                    rule_type=RuleType.PRECONDITION,
                    conditions=conditions,
                    conclusions=[f'action("{action_name}") is successful'],
                    confidence=0.8,
                    support=1,
                    metadata={
                        "action": action_name,
                        "reward": reward,
                        "episode_index": len(self.experience_history) - 1
                    }
                )
                
                existing_rules = self.knowledge_base.get_rules_by_type(RuleType.PRECONDITION)
                matched = False
                for existing in existing_rules:
                    if self._rules_similar(existing, rule):
                        existing.support += 1
                        existing.confidence = min(1.0, existing.confidence + self.learning_rate)
                        matched = True
                        break
                
                if not matched:
                    self.knowledge_base.add_rule(rule)
    
    def _extract_effect_rules(self, episode_data: Dict):
        states = episode_data.get("states", [])
        actions = episode_data.get("actions", [])
        rewards = episode_data.get("rewards", [])
        
        for i in range(len(states) - 1):
            state = states[i]
            action = actions[i]
            next_state = states[i + 1]
            reward = rewards[i]
            
            action_name = action.get("tool") if isinstance(action, dict) else action
            
            state_changes = self._compare_states(state, next_state)
            
            if state_changes:
                conditions = self._state_to_conditions(state)
                conclusions = [
                    f'state["{key}"] = {json.dumps(value)}' 
                    for key, value in state_changes.items()
                ]
                
                rule_name = f"effect_{action_name}_{len(self.knowledge_base.rules)}"
                rule = Rule(
                    name=rule_name,
                    rule_type=RuleType.EFFECT,
                    conditions=conditions,
                    conclusions=conclusions,
                    confidence=0.7,
                    support=1,
                    metadata={
                        "action": action_name,
                        "reward": reward,
                        "state_changes": state_changes
                    }
                )
                
                existing_rules = self.knowledge_base.get_rules_by_type(RuleType.EFFECT)
                matched = False
                for existing in existing_rules:
                    if self._rules_similar(existing, rule):
                        existing.support += 1
                        existing.confidence = min(1.0, existing.confidence + self.learning_rate * 0.5)
                        matched = True
                        break
                
                if not matched:
                    self.knowledge_base.add_rule(rule)
    
    def _extract_transition_rules(self, episode_data: Dict):
        states = episode_data.get("states", [])
        actions = episode_data.get("actions", [])
        
        for i in range(len(states) - 1):
            state = states[i]
            action = actions[i]
            next_state = states[i + 1]
            
            current_state_value = state.get("current_state", "unknown")
            next_state_value = next_state.get("current_state", "unknown")
            action_name = action.get("tool") if isinstance(action, dict) else action
            
            if current_state_value != next_state_value:
                conditions = [f'state["current_state"] == "{current_state_value}"']
                conclusions = [f'state["current_state"] = "{next_state_value}"']
                
                rule_name = f"transition_{current_state_value}_to_{next_state_value}_{action_name}"
                rule = Rule(
                    name=rule_name,
                    rule_type=RuleType.TRANSITION,
                    conditions=conditions,
                    conclusions=conclusions,
                    confidence=0.9,
                    support=1,
                    metadata={
                        "from_state": current_state_value,
                        "to_state": next_state_value,
                        "action": action_name
                    }
                )
                
                existing = self.knowledge_base.rules.get(rule_name)
                if existing:
                    existing.support += 1
                else:
                    self.knowledge_base.add_rule(rule)
    
    def _extract_reward_patterns(self, episode_data: Dict):
        states = episode_data.get("states", [])
        actions = episode_data.get("actions", [])
        rewards = episode_data.get("rewards", [])
        
        positive_actions = {}
        negative_actions = {}
        
        for state, action, reward in zip(states, actions, rewards):
            action_name = action.get("tool") if isinstance(action, dict) else action
            
            if reward > 0:
                if action_name not in positive_actions:
                    positive_actions[action_name] = {"count": 0, "total_reward": 0}
                positive_actions[action_name]["count"] += 1
                positive_actions[action_name]["total_reward"] += reward
            elif reward < 0:
                if action_name not in negative_actions:
                    negative_actions[action_name] = {"count": 0, "total_reward": 0}
                negative_actions[action_name]["count"] += 1
                negative_actions[action_name]["total_reward"] += reward
        
        for action_name, data in positive_actions.items():
            if data["count"] >= self.min_support:
                avg_reward = data["total_reward"] / data["count"]
                confidence = min(1.0, data["count"] / (self.min_support * 2))
                
                rule_name = f"reward_positive_{action_name}"
                rule = Rule(
                    name=rule_name,
                    rule_type=RuleType.REWARD,
                    conditions=[f'action == "{action_name}"'],
                    conclusions=[f'reward = {avg_reward:.2f}'],
                    confidence=confidence,
                    support=data["count"],
                    metadata={
                        "action": action_name,
                        "avg_reward": avg_reward,
                        "type": "positive"
                    }
                )
                
                existing = self.knowledge_base.rules.get(rule_name)
                if existing:
                    existing.support += data["count"]
                    existing.confidence = min(1.0, existing.confidence + 0.1)
                else:
                    self.knowledge_base.add_rule(rule)
        
        for action_name, data in negative_actions.items():
            if data["count"] >= self.min_support:
                avg_reward = data["total_reward"] / data["count"]
                confidence = min(1.0, data["count"] / (self.min_support * 2))
                
                rule_name = f"reward_negative_{action_name}"
                rule = Rule(
                    name=rule_name,
                    rule_type=RuleType.REWARD,
                    conditions=[f'action == "{action_name}"'],
                    conclusions=[f'reward = {avg_reward:.2f}'],
                    confidence=confidence,
                    support=data["count"],
                    metadata={
                        "action": action_name,
                        "avg_reward": avg_reward,
                        "type": "negative"
                    }
                )
                
                existing = self.knowledge_base.rules.get(rule_name)
                if existing:
                    existing.support += data["count"]
                    existing.confidence = min(1.0, existing.confidence + 0.1)
                else:
                    self.knowledge_base.add_rule(rule)
    
    def _state_to_conditions(self, state: Dict) -> List[str]:
        conditions = []
        
        state_vars = state.get("state_variables", {}) if "state_variables" in state else state
        
        for key, value in state_vars.items():
            if isinstance(value, str):
                conditions.append(f'state["{key}"] == "{value}"')
            elif isinstance(value, (int, float, bool)):
                conditions.append(f'state["{key}"] == {value}')
        
        if "current_state" in state:
            conditions.append(f'state["current_state"] == "{state["current_state"]}"')
        
        return conditions
    
    def _compare_states(self, state1: Dict, state2: Dict) -> Dict[str, Any]:
        changes = {}
        
        vars1 = state1.get("state_variables", {}) if "state_variables" in state1 else state1
        vars2 = state2.get("state_variables", {}) if "state_variables" in state2 else state2
        
        all_keys = set(vars1.keys()).union(set(vars2.keys()))
        
        for key in all_keys:
            if vars1.get(key) != vars2.get(key):
                changes[key] = vars2.get(key)
        
        if state1.get("current_state") != state2.get("current_state"):
            changes["current_state"] = state2.get("current_state")
        
        return changes
    
    def _rules_similar(self, rule1: Rule, rule2: Rule) -> bool:
        if rule1.rule_type != rule2.rule_type:
            return False
        
        if set(rule1.conditions) == set(rule2.conditions):
            return True
        
        common_conditions = set(rule1.conditions).intersection(set(rule2.conditions))
        total_conditions = set(rule1.conditions).union(set(rule2.conditions))
        
        if total_conditions and len(common_conditions) / len(total_conditions) > 0.7:
            return True
        
        return False
    
    def generate_improvement_suggestions(self) -> List[Dict]:
        suggestions = []
        
        reward_rules = self.knowledge_base.get_rules_by_type(RuleType.REWARD)
        
        positive_rules = [r for r in reward_rules if r.metadata.get("type") == "positive"]
        negative_rules = [r for r in reward_rules if r.metadata.get("type") == "negative"]
        
        for rule in positive_rules:
            if rule.confidence >= self.min_confidence and rule.support >= self.min_support:
                suggestions.append({
                    "type": "positive_action",
                    "action": rule.metadata.get("action"),
                    "confidence": rule.confidence,
                    "support": rule.support,
                    "suggestion": f"建议增加使用动作 '{rule.metadata.get('action')}'，平均奖励为 {rule.metadata.get('avg_reward'):.2f}"
                })
        
        for rule in negative_rules:
            if rule.confidence >= self.min_confidence and rule.support >= self.min_support:
                suggestions.append({
                    "type": "negative_action",
                    "action": rule.metadata.get("action"),
                    "confidence": rule.confidence,
                    "support": rule.support,
                    "suggestion": f"建议减少使用动作 '{rule.metadata.get('action')}'，平均奖励为 {rule.metadata.get('avg_reward'):.2f}"
                })
        
        transition_rules = self.knowledge_base.get_rules_by_type(RuleType.TRANSITION)
        if transition_rules:
            suggestions.append({
                "type": "state_transition",
                "count": len(transition_rules),
                "suggestion": f"已学习 {len(transition_rules)} 个状态转换规则，可以用于规划"
            })
        
        return suggestions
    
    def get_reflection_summary(self) -> Dict[str, Any]:
        kb_stats = self.knowledge_base.get_stats()
        
        return {
            "knowledge_base": kb_stats,
            "episodes_reflected": len(self.experience_history),
            "improvement_suggestions": len(self.generate_improvement_suggestions()),
            "high_confidence_rules": len([
                r for r in self.knowledge_base.rules.values() 
                if r.confidence >= self.min_confidence
            ])
        }
