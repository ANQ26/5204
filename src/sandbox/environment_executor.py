"""
环境执行器 - 负责在沙盒中安全执行环境代码和工具调用
"""

import time
import threading
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import traceback


class ExecutionStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


@dataclass
class ExecutionResult:
    status: ExecutionStatus
    result: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0
    step_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class EnvironmentExecutor:
    """
    环境执行器
    
    核心功能：
    1. 安全执行环境中的工具调用
    2. 支持超时控制和资源限制
    3. 记录执行轨迹和日志
    4. 支持多轮对话式执行
    """
    
    def __init__(self, timeout: float = 30.0, max_steps: int = 100):
        self.timeout = timeout
        self.max_steps = max_steps
        self.execution_history: List[Dict] = []
    
    def execute_with_timeout(
        self,
        func: Callable,
        *args,
        timeout: float = None,
        **kwargs
    ) -> ExecutionResult:
        timeout = timeout or self.timeout
        result_container = {"result": None, "error": None}
        
        def target():
            try:
                result_container["result"] = func(*args, **kwargs)
            except Exception as e:
                result_container["error"] = str(e)
                result_container["traceback"] = traceback.format_exc()
        
        thread = threading.Thread(target=target)
        start_time = time.time()
        
        thread.start()
        thread.join(timeout=timeout)
        
        execution_time = time.time() - start_time
        
        if thread.is_alive():
            return ExecutionResult(
                status=ExecutionStatus.TIMEOUT,
                error=f"Execution timed out after {timeout} seconds",
                execution_time=execution_time
            )
        
        if result_container["error"] is not None:
            return ExecutionResult(
                status=ExecutionStatus.FAILED,
                error=result_container["error"],
                execution_time=execution_time,
                metadata={"traceback": result_container.get("traceback")}
            )
        
        return ExecutionResult(
            status=ExecutionStatus.COMPLETED,
            result=result_container["result"],
            execution_time=execution_time
        )
    
    def execute_tool(
        self,
        environment,
        tool_name: str,
        parameters: Dict[str, Any] = None
    ) -> Tuple[Dict[str, Any], float, bool, Dict]:
        parameters = parameters or {}
        
        start_time = time.time()
        
        try:
            result, reward, done, info = environment.execute_tool(tool_name, **parameters)
            
            execution_time = time.time() - start_time
            
            self.execution_history.append({
                "tool": tool_name,
                "parameters": parameters,
                "result": result,
                "reward": reward,
                "done": done,
                "info": info,
                "execution_time": execution_time,
                "timestamp": time.time()
            })
            
            return result, reward, done, info
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_result = {
                "error": str(e),
                "status": "failed",
                "traceback": traceback.format_exc()
            }
            
            self.execution_history.append({
                "tool": tool_name,
                "parameters": parameters,
                "error": str(e),
                "execution_time": execution_time,
                "timestamp": time.time(),
                "status": "failed"
            })
            
            return error_result, -1.0, False, {"success": False, "error": str(e)}
    
    def execute_multi_turn(
        self,
        environment,
        agent,
        max_turns: int = None,
        callback: Callable = None
    ) -> ExecutionResult:
        max_turns = max_turns or self.max_steps
        total_reward = 0.0
        step_count = 0
        start_time = time.time()
        
        observation = environment.get_observation()
        
        for step in range(max_turns):
            step_count += 1
            
            action = agent.act(observation)
            
            if action is None:
                break
            
            tool_name = action.get("tool")
            parameters = action.get("parameters", {})
            
            result, reward, done, info = self.execute_tool(environment, tool_name, parameters)
            
            total_reward += reward
            
            next_observation = environment.get_observation()
            
            agent.observe(observation, action, reward, next_observation, done)
            
            observation = next_observation
            
            if callback:
                callback({
                    "step": step,
                    "action": action,
                    "result": result,
                    "reward": reward,
                    "done": done,
                    "observation": observation
                })
            
            if done:
                break
        
        execution_time = time.time() - start_time
        
        return ExecutionResult(
            status=ExecutionStatus.COMPLETED,
            result={
                "total_reward": total_reward,
                "step_count": step_count,
                "execution_history": self.execution_history[-step_count:]
            },
            execution_time=execution_time,
            step_count=step_count
        )
    
    def reset_history(self):
        self.execution_history = []
    
    def get_history_summary(self) -> Dict[str, Any]:
        if not self.execution_history:
            return {
                "total_steps": 0,
                "total_reward": 0.0,
                "success_rate": 0.0,
                "average_execution_time": 0.0
            }
        
        total_reward = sum(h.get("reward", 0) for h in self.execution_history)
        successful = sum(1 for h in self.execution_history if h.get("status") != "failed" and not h.get("error"))
        avg_time = sum(h.get("execution_time", 0) for h in self.execution_history) / len(self.execution_history)
        
        return {
            "total_steps": len(self.execution_history),
            "total_reward": total_reward,
            "success_rate": successful / len(self.execution_history) if self.execution_history else 0.0,
            "average_execution_time": avg_time,
            "tool_usage": self._get_tool_usage_stats()
        }
    
    def _get_tool_usage_stats(self) -> Dict[str, Dict]:
        stats = {}
        for entry in self.execution_history:
            tool = entry.get("tool", "unknown")
            if tool not in stats:
                stats[tool] = {
                    "count": 0,
                    "successes": 0,
                    "failures": 0,
                    "total_reward": 0.0,
                    "average_time": 0.0
                }
            
            stats[tool]["count"] += 1
            if entry.get("error"):
                stats[tool]["failures"] += 1
            else:
                stats[tool]["successes"] += 1
            
            stats[tool]["total_reward"] += entry.get("reward", 0)
        
        for tool in stats:
            stats[tool]["average_time"] = (
                sum(h.get("execution_time", 0) for h in self.execution_history if h.get("tool") == tool) /
                stats[tool]["count"] if stats[tool]["count"] > 0 else 0.0
            )
        
        return stats
