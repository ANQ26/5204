"""
策略优化器 - 优化智能体的决策策略
"""

import math
import random
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum


class OptimizationMethod(Enum):
    POLICY_ITERATION = "policy_iteration"
    VALUE_ITERATION = "value_iteration"
    Q_ITERATION = "q_iteration"
    CROSS_ENTROPY = "cross_entropy"
    EVOLUTIONARY = "evolutionary"


@dataclass
class Policy:
    name: str = "default_policy"
    state_action_probs: Dict[str, Dict[str, float]] = field(default_factory=lambda: defaultdict(lambda: defaultdict(float)))
    value_function: Dict[str, float] = field(default_factory=dict)
    q_function: Dict[str, Dict[str, float]] = field(default_factory=lambda: defaultdict(dict))
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_action_probability(self, state_key: str, action: str) -> float:
        return self.state_action_probs[state_key].get(action, 0.0)
    
    def get_best_action(self, state_key: str, available_actions: List[str]) -> Optional[str]:
        if state_key not in self.state_action_probs:
            return random.choice(available_actions) if available_actions else None
        
        probs = self.state_action_probs[state_key]
        best_action = None
        best_prob = -1.0
        
        for action in available_actions:
            prob = probs.get(action, 0.0)
            if prob > best_prob:
                best_prob = prob
                best_action = action
        
        return best_action
    
    def sample_action(self, state_key: str, available_actions: List[str]) -> Optional[str]:
        if state_key not in self.state_action_probs:
            return random.choice(available_actions) if available_actions else None
        
        probs = self.state_action_probs[state_key]
        
        actions = []
        probabilities = []
        for action in available_actions:
            prob = probs.get(action, 0.001)
            actions.append(action)
            probabilities.append(prob)
        
        total = sum(probabilities)
        probabilities = [p / total for p in probabilities]
        
        return random.choices(actions, weights=probabilities, k=1)[0] if actions else None


class StrategyOptimizer:
    """
    策略优化器
    
    核心功能：
    1. 从经验中优化策略
    2. 支持多种优化方法
    3. 实现从"被动执行"到"自主规划"的进化
    """
    
    def __init__(
        self,
        method: OptimizationMethod = OptimizationMethod.Q_ITERATION,
        discount_factor: float = 0.99,
        learning_rate: float = 0.1
    ):
        self.method = method
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        self.policy = Policy()
        self.experience_buffer: List[Dict] = []
    
    def add_experience(
        self,
        state_key: str,
        action: str,
        reward: float,
        next_state_key: str,
        done: bool,
        available_actions: List[str] = None
    ):
        self.experience_buffer.append({
            "state": state_key,
            "action": action,
            "reward": reward,
            "next_state": next_state_key,
            "done": done,
            "available_actions": available_actions or []
        })
    
    def optimize(self, iterations: int = 100) -> Policy:
        if self.method == OptimizationMethod.Q_ITERATION:
            return self._q_iteration(iterations)
        elif self.method == OptimizationMethod.VALUE_ITERATION:
            return self._value_iteration(iterations)
        elif self.method == OptimizationMethod.POLICY_ITERATION:
            return self._policy_iteration(iterations)
        elif self.method == OptimizationMethod.CROSS_ENTROPY:
            return self._cross_entropy_method(iterations)
        elif self.method == OptimizationMethod.EVOLUTIONARY:
            return self._evolutionary_optimization(iterations)
        
        return self.policy
    
    def _q_iteration(self, iterations: int) -> Policy:
        if not self.experience_buffer:
            return self.policy
        
        for exp in self.experience_buffer:
            state = exp["state"]
            action = exp["action"]
            reward = exp["reward"]
            next_state = exp["next_state"]
            done = exp["done"]
            
            current_q = self.policy.q_function[state].get(action, 0.0)
            
            if next_state in self.policy.q_function and self.policy.q_function[next_state]:
                max_next_q = max(self.policy.q_function[next_state].values())
            else:
                max_next_q = 0.0
            
            target = reward + self.discount_factor * max_next_q * (not done)
            new_q = current_q + self.learning_rate * (target - current_q)
            
            self.policy.q_function[state][action] = new_q
        
        for state in self.policy.q_function:
            if not self.policy.q_function[state]:
                continue
            
            max_q = max(self.policy.q_function[state].values())
            
            for action in self.policy.q_function[state]:
                if self.policy.q_function[state][action] == max_q:
                    self.policy.state_action_probs[state][action] = 1.0
                else:
                    self.policy.state_action_probs[state][action] = 0.0
        
        return self.policy
    
    def _value_iteration(self, iterations: int) -> Policy:
        if not self.experience_buffer:
            return self.policy
        
        transition_counts: Dict[str, Dict[str, Dict[str, int]]] = defaultdict(lambda: defaultdict(dict))
        reward_sums: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        
        for exp in self.experience_buffer:
            state = exp["state"]
            action = exp["action"]
            reward = exp["reward"]
            next_state = exp["next_state"]
            
            transition_counts[state][action][next_state] = transition_counts[state][action].get(next_state, 0) + 1
            reward_sums[state][action] += reward
        
        states = set()
        for exp in self.experience_buffer:
            states.add(exp["state"])
            states.add(exp["next_state"])
        
        for _ in range(iterations):
            delta = 0.0
            
            for state in states:
                if state not in self.policy.value_function:
                    self.policy.value_function[state] = 0.0
                
                old_value = self.policy.value_function[state]
                
                max_value = 0.0
                for action in transition_counts[state]:
                    action_value = 0.0
                    
                    total_transitions = sum(transition_counts[state][action].values())
                    avg_reward = reward_sums[state][action] / total_transitions if total_transitions > 0 else 0.0
                    
                    for next_state, count in transition_counts[state][action].items():
                        prob = count / total_transitions
                        next_value = self.policy.value_function.get(next_state, 0.0)
                        action_value += prob * (avg_reward + self.discount_factor * next_value)
                    
                    max_value = max(max_value, action_value)
                
                self.policy.value_function[state] = max_value
                delta = max(delta, abs(old_value - max_value))
            
            if delta < 1e-6:
                break
        
        for state in states:
            best_actions = []
            best_value = -float('inf')
            
            for action in transition_counts[state]:
                action_value = 0.0
                total_transitions = sum(transition_counts[state][action].values())
                avg_reward = reward_sums[state][action] / total_transitions if total_transitions > 0 else 0.0
                
                for next_state, count in transition_counts[state][action].items():
                    prob = count / total_transitions
                    next_value = self.policy.value_function.get(next_state, 0.0)
                    action_value += prob * (avg_reward + self.discount_factor * next_value)
                
                if action_value > best_value:
                    best_value = action_value
                    best_actions = [action]
                elif action_value == best_value:
                    best_actions.append(action)
            
            for action in best_actions:
                self.policy.state_action_probs[state][action] = 1.0 / len(best_actions)
        
        return self.policy
    
    def _policy_iteration(self, iterations: int) -> Policy:
        if not self.experience_buffer:
            return self.policy
        
        states = set()
        for exp in self.experience_buffer:
            states.add(exp["state"])
            states.add(exp["next_state"])
        
        for state in states:
            for exp in self.experience_buffer:
                if exp["state"] == state:
                    action = exp["action"]
                    self.policy.state_action_probs[state][action] = 1.0
                    break
        
        for _ in range(iterations):
            policy_stable = True
            
            self._policy_evaluation()
            
            for state in states:
                old_action = self.policy.get_best_action(state, list(self.policy.state_action_probs[state].keys()))
                
                best_action = self._policy_improvement(state)
                
                if old_action != best_action and best_action:
                    policy_stable = False
                    self.policy.state_action_probs[state].clear()
                    self.policy.state_action_probs[state][best_action] = 1.0
            
            if policy_stable:
                break
        
        return self.policy
    
    def _policy_evaluation(self):
        for exp in self.experience_buffer:
            state = exp["state"]
            if state not in self.policy.value_function:
                self.policy.value_function[state] = 0.0
        
        for _ in range(100):
            delta = 0.0
            
            for exp in self.experience_buffer:
                state = exp["state"]
                action = exp["action"]
                reward = exp["reward"]
                next_state = exp["next_state"]
                done = exp["done"]
                
                old_value = self.policy.value_function.get(state, 0.0)
                
                action_prob = self.policy.get_action_probability(state, action)
                
                if action_prob > 0:
                    next_value = self.policy.value_function.get(next_state, 0.0)
                    new_value = reward + self.discount_factor * next_value * (not done)
                    
                    self.policy.value_function[state] = (
                        (1 - self.learning_rate) * old_value + self.learning_rate * new_value
                    )
                    
                    delta = max(delta, abs(old_value - self.policy.value_function[state]))
            
            if delta < 1e-6:
                break
    
    def _policy_improvement(self, state: str) -> Optional[str]:
        q_values = {}
        
        for exp in self.experience_buffer:
            if exp["state"] == state:
                action = exp["action"]
                if action not in q_values:
                    q_values[action] = 0.0
                
                next_value = self.policy.value_function.get(exp["next_state"], 0.0)
                q_values[action] += exp["reward"] + self.discount_factor * next_value
        
        if not q_values:
            return None
        
        best_action = max(q_values.keys(), key=lambda a: q_values[a])
        return best_action
    
    def _cross_entropy_method(self, iterations: int, elite_fraction: float = 0.2) -> Policy:
        if not self.experience_buffer:
            return self.policy
        
        states = set()
        all_actions = set()
        for exp in self.experience_buffer:
            states.add(exp["state"])
            all_actions.add(exp["action"])
        
        for state in states:
            for action in all_actions:
                self.policy.state_action_probs[state][action] = 1.0 / len(all_actions)
        
        for _ in range(iterations):
            episode_rewards: Dict[int, float] = defaultdict(float)
            current_episode = 0
            
            for exp in self.experience_buffer:
                episode_rewards[current_episode] += exp["reward"]
                if exp["done"]:
                    current_episode += 1
            
            if not episode_rewards:
                break
            
            sorted_episodes = sorted(episode_rewards.items(), key=lambda x: x[1], reverse=True)
            elite_count = max(1, int(len(sorted_episodes) * elite_fraction))
            elite_episodes = set(ep for ep, _ in sorted_episodes[:elite_count])
            
            state_action_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
            
            current_episode = 0
            for exp in self.experience_buffer:
                if current_episode in elite_episodes:
                    state = exp["state"]
                    action = exp["action"]
                    state_action_counts[state][action] += 1
                
                if exp["done"]:
                    current_episode += 1
            
            for state in state_action_counts:
                total = sum(state_action_counts[state].values())
                for action in state_action_counts[state]:
                    self.policy.state_action_probs[state][action] = state_action_counts[state][action] / total
        
        return self.policy
    
    def _evolutionary_optimization(
        self,
        iterations: int,
        population_size: int = 20,
        mutation_rate: float = 0.1
    ) -> Policy:
        if not self.experience_buffer:
            return self.policy
        
        states = set()
        all_actions = set()
        for exp in self.experience_buffer:
            states.add(exp["state"])
            all_actions.add(exp["action"])
        
        all_actions = list(all_actions)
        states = list(states)
        
        def generate_random_policy() -> Dict[str, Dict[str, float]]:
            policy = defaultdict(lambda: defaultdict(float))
            for state in states:
                remaining = 1.0
                for i, action in enumerate(all_actions):
                    if i == len(all_actions) - 1:
                        policy[state][action] = remaining
                    else:
                        prob = random.uniform(0, remaining)
                        policy[state][action] = prob
                        remaining -= prob
            return policy
        
        def evaluate_policy(policy_dict: Dict[str, Dict[str, float]]) -> float:
            total_reward = 0.0
            episode_count = 0
            
            for exp in self.experience_buffer:
                state = exp["state"]
                action = exp["action"]
                
                prob = policy_dict[state].get(action, 0.01)
                total_reward += exp["reward"] * prob
                
                if exp["done"]:
                    episode_count += 1
            
            return total_reward / max(episode_count, 1)
        
        population = [generate_random_policy() for _ in range(population_size)]
        
        for _ in range(iterations):
            fitness_scores = [evaluate_policy(p) for p in population]
            
            sorted_pop = sorted(zip(population, fitness_scores), key=lambda x: x[1], reverse=True)
            
            elite_count = population_size // 2
            elites = [p for p, _ in sorted_pop[:elite_count]]
            
            new_population = elites[:]
            
            while len(new_population) < population_size:
                parent1 = random.choice(elites)
                parent2 = random.choice(elites)
                
                child = defaultdict(lambda: defaultdict(float))
                for state in states:
                    if random.random() < 0.5:
                        for action in all_actions:
                            child[state][action] = parent1[state].get(action, 0.0)
                    else:
                        for action in all_actions:
                            child[state][action] = parent2[state].get(action, 0.0)
                    
                    if random.random() < mutation_rate:
                        action_to_mutate = random.choice(all_actions)
                        child[state][action_to_mutate] += random.uniform(-0.2, 0.2)
                        child[state][action_to_mutate] = max(0.0, min(1.0, child[state][action_to_mutate]))
                        
                        total = sum(child[state].values())
                        for action in child[state]:
                            child[state][action] /= total
                
                new_population.append(child)
            
            population = new_population
        
        best_policy = max(population, key=evaluate_policy)
        self.policy.state_action_probs = best_policy
        
        return self.policy
    
    def get_optimized_action(
        self,
        state_key: str,
        available_actions: List[str],
        exploration_rate: float = 0.0
    ) -> Optional[str]:
        if random.random() < exploration_rate:
            return random.choice(available_actions) if available_actions else None
        
        return self.policy.get_best_action(state_key, available_actions)
    
    def clear_experience(self):
        self.experience_buffer.clear()
    
    def get_policy_stats(self) -> Dict[str, Any]:
        return {
            "method": self.method.value,
            "states_count": len(self.policy.state_action_probs),
            "experience_count": len(self.experience_buffer),
            "discount_factor": self.discount_factor,
            "learning_rate": self.learning_rate
        }
