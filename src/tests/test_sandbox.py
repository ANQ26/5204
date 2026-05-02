"""
沙盒环境测试 - 验证环境管理、执行和验证功能
"""

import unittest
import json
from typing import Dict, List, Any
from unittest.mock import patch, MagicMock

from src.sandbox.environment_manager import (
    EnvironmentManager, EnvironmentInstance
)
from src.sandbox.environment_executor import (
    EnvironmentExecutor, ExecutionStatus, ExecutionResult
)
from src.sandbox.environment_validator import (
    EnvironmentValidator, ValidationSeverity, ValidationIssue
)


class TestEnvironmentManager(unittest.TestCase):
    """测试环境管理器"""
    
    def setUp(self):
        self.manager = EnvironmentManager()
    
    def test_create_environment(self):
        """测试创建环境"""
        instance = self.manager.create_environment(
            name="TestEnv",
            type="base",
            config={"test": "value"}
        )
        
        self.assertIsNotNone(instance.id)
        self.assertEqual(instance.name, "TestEnv")
        self.assertEqual(instance.type, "base")
        self.assertEqual(instance.status, "ready")
    
    def test_create_ecommerce_environment(self):
        """测试创建电商环境"""
        instance = self.manager.create_ecommerce_environment(
            name="EcommerceTest",
            user_balance=1500.0
        )
        
        self.assertEqual(instance.type, "e-commerce")
        self.assertEqual(instance.config.get("user_balance"), 1500.0)
    
    def test_create_os_environment(self):
        """测试创建操作系统环境"""
        instance = self.manager.create_os_environment(
            name="OSTest",
            initial_directory="/home/test"
        )
        
        self.assertEqual(instance.type, "operating-system")
        self.assertEqual(instance.config.get("initial_directory"), "/home/test")
    
    def test_list_environments(self):
        """测试列出环境"""
        self.manager.create_environment(name="Env1", type="base")
        self.manager.create_ecommerce_environment(name="Env2")
        
        envs = self.manager.list_environments()
        
        self.assertEqual(len(envs), 2)
    
    def test_get_environment(self):
        """测试获取环境"""
        instance = self.manager.create_environment(name="GetTest", type="base")
        
        retrieved = self.manager.get_environment(instance.id)
        
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.id, instance.id)
    
    def test_remove_environment(self):
        """测试移除环境"""
        instance = self.manager.create_environment(name="RemoveTest", type="base")
        
        result = self.manager.remove_environment(instance.id)
        
        self.assertTrue(result)
        self.assertIsNone(self.manager.get_environment(instance.id))
    
    def test_environment_count(self):
        """测试环境计数"""
        self.manager.create_environment(name="Count1", type="base")
        self.manager.create_environment(name="Count2", type="base")
        
        count = self.manager.get_environment_count()
        
        self.assertEqual(count, 2)


class TestEnvironmentExecution(unittest.TestCase):
    """测试环境执行器"""
    
    def setUp(self):
        self.executor = EnvironmentExecutor(timeout=5.0)
        self.manager = EnvironmentManager()
    
    def test_execution_result_creation(self):
        """测试执行结果创建"""
        result = ExecutionResult(
            status=ExecutionStatus.SUCCESS,
            message="Test executed",
            data={"value": 42},
            reward=1.0
        )
        
        self.assertEqual(result.status, ExecutionStatus.SUCCESS)
        self.assertEqual(result.message, "Test executed")
        self.assertEqual(result.data, {"value": 42})
        self.assertEqual(result.reward, 1.0)
    
    def test_execution_with_timeout(self):
        """测试带超时的执行"""
        import time
        
        def quick_func():
            time.sleep(0.01)
            return ExecutionResult(
                status=ExecutionStatus.SUCCESS,
                message="Quick",
                reward=1.0
            )
        
        result = self.executor.execute_with_timeout(quick_func, timeout=1.0)
        
        self.assertEqual(result.status, ExecutionStatus.SUCCESS)
    
    def test_execute_multi_turn(self):
        """测试多轮执行"""
        instance = self.manager.create_ecommerce_environment(name="MultiTurnTest")
        env = self.manager.load_environment(instance)
        
        result = self.executor.execute_multi_turn(
            env,
            max_turns=3,
            stop_condition=lambda r: False
        )
        
        self.assertIsNotNone(result)
        self.assertLessEqual(len(result['turns']), 3)


class TestEnvironmentValidator(unittest.TestCase):
    """测试环境验证器"""
    
    def setUp(self):
        self.validator = EnvironmentValidator()
    
    def test_validation_issue_creation(self):
        """测试验证问题创建"""
        issue = ValidationIssue(
            severity=ValidationSeverity.ERROR,
            message="Test error",
            code="E001",
            location="test.py:10"
        )
        
        self.assertEqual(issue.severity, ValidationSeverity.ERROR)
        self.assertEqual(issue.message, "Test error")
        self.assertEqual(issue.code, "E001")
    
    def test_validate_syntax_valid(self):
        """测试验证有效语法"""
        valid_code = '''
def hello():
    print("Hello World")
'''
        
        result = self.validator.validate_syntax(valid_code)
        
        self.assertEqual(result.severity, ValidationSeverity.INFO)
        self.assertTrue(result.message.startswith("语法验证通过"))
    
    def test_validate_syntax_invalid(self):
        """测试验证无效语法"""
        invalid_code = '''
def hello(
    print("Hello World")
'''
        
        result = self.validator.validate_syntax(invalid_code)
        
        self.assertEqual(result.severity, ValidationSeverity.ERROR)
        self.assertIn("语法错误", result.message)
    
    def test_validate_imports_safe(self):
        """测试验证安全导入"""
        safe_code = '''
import json
import math
from typing import List
'''
        
        issues = self.validator.validate_imports(safe_code)
        
        self.assertEqual(len([i for i in issues if i.severity == ValidationSeverity.ERROR]), 0)
    
    def test_validate_imports_unsafe(self):
        """测试验证不安全导入"""
        unsafe_code = '''
import os
import sys
from subprocess import call
'''
        
        issues = self.validator.validate_imports(unsafe_code)
        
        self.assertGreater(len([i for i in issues if i.severity == ValidationSeverity.WARNING]), 0)
    
    def test_validate_native_functions(self):
        """测试验证危险内置函数"""
        dangerous_code = '''
eval("1 + 1")
exec("print('hello')")
'''
        
        issues = self.validator.validate_native_functions(dangerous_code)
        
        self.assertGreater(len(issues), 0)
    
    def test_full_validate(self):
        """测试完整验证流程"""
        code = '''
class TestEnv:
    def __init__(self):
        pass
    
    def reset(self):
        return {}
    
    def execute_tool(self, tool, **kwargs):
        return "success", 1.0, False, {}
    
    def get_observation(self):
        return {}
'''
        
        result = self.validator.validate(code)
        
        self.assertTrue(result.valid)
        self.assertEqual(len(result.errors), 0)
    
    def test_validate_logic_errors(self):
        """测试验证逻辑错误"""
        code_with_errors = '''
class TestEnv:
    pass
'''
        
        result = self.validator.validate_logic(code_with_errors)
        
        self.assertGreater(len(result), 0)


class TestEcommerceEnvironmentIntegration(unittest.TestCase):
    """电商环境集成测试"""
    
    def setUp(self):
        self.manager = EnvironmentManager()
        self.instance = self.manager.create_ecommerce_environment(
            name="IntegrationTest",
            user_balance=2000.0
        )
        self.env = self.manager.load_environment(self.instance)
    
    def test_ecommerce_basic_flow(self):
        """测试电商基础流程"""
        obs = self.env.reset()
        
        self.assertEqual(obs.get("current_state"), "initial")
        
        result, reward, done, info = self.env.execute_tool(
            "search_products", query="Laptop"
        )
        
        self.assertIn("search_results", info)
        self.assertEqual(obs.get("current_state"), "browsing" if obs.get("current_state") else "initial")
    
    def test_ecommerce_observation(self):
        """测试电商环境观察"""
        self.env.reset()
        obs = self.env.get_observation()
        
        self.assertIn("current_state", obs)
        self.assertIn("user_balance", obs)
        self.assertIn("cart", obs)
    
    def test_ecommerce_add_to_cart(self):
        """测试添加到购物车"""
        self.env.reset()
        
        result, reward, done, info = self.env.execute_tool(
            "add_to_cart", product_id="p1", quantity=1
        )
        
        self.assertIsInstance(result, str)


class TestOSEnvironmentIntegration(unittest.TestCase):
    """操作系统环境集成测试"""
    
    def setUp(self):
        self.manager = EnvironmentManager()
        self.instance = self.manager.create_os_environment(
            name="OSIntegrationTest"
        )
        self.env = self.manager.load_environment(self.instance)
    
    def test_os_basic_flow(self):
        """测试操作系统基础流程"""
        obs = self.env.reset()
        
        self.assertEqual(obs.get("current_state"), "running")
        
        result, reward, done, info = self.env.execute_tool(
            "list_directory", path="/home/user"
        )
        
        self.assertIsInstance(result, str)
    
    def test_os_observation(self):
        """测试OS环境观察"""
        self.env.reset()
        obs = self.env.get_observation()
        
        self.assertIn("current_state", obs)
        self.assertIn("current_directory", obs)
        self.assertIn("user", obs)
    
    def test_os_write_read_file(self):
        """测试写读文件"""
        self.env.reset()
        
        write_result, write_reward, write_done, write_info = self.env.execute_tool(
            "write_file", path="/home/user/test.txt", content="Hello World"
        )
        
        read_result, read_reward, read_done, read_info = self.env.execute_tool(
            "read_file", path="/home/user/test.txt"
        )
        
        self.assertIsInstance(read_result, str)


if __name__ == "__main__":
    unittest.main(verbosity=2)
