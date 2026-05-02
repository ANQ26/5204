"""
代码生成引擎测试 - 验证EnvScaler和模板引擎功能
"""

import unittest
import json
from typing import Dict, List, Any

from src.code_generator.env_scaler import (
    EnvScaler, TemplateEngine, EnvironmentSpec,
    BaseCodeGenerator
)


class TestEnvironmentSpec(unittest.TestCase):
    """测试环境规格类"""
    
    def test_spec_creation(self):
        """测试环境规格创建"""
        spec = EnvironmentSpec(
            name="TestEnv",
            type="base",
            description="测试环境",
            tools=[{"name": "test_tool", "description": "测试工具"}],
            state_variables={"var1": {"initial": 0, "type": "int"}}
        )
        
        self.assertEqual(spec.name, "TestEnv")
        self.assertEqual(spec.type, "base")
        self.assertEqual(len(spec.tools), 1)
        self.assertIn("var1", spec.state_variables)


class TestTemplateEngine(unittest.TestCase):
    """测试模板引擎"""
    
    def setUp(self):
        self.engine = TemplateEngine()
    
    def test_load_template(self):
        """测试加载模板"""
        self.engine.load_template("test_template", "Hello {{name}}!")
        templates = self.engine.list_templates()
        self.assertIn("test_template", templates)
    
    def test_render_template(self):
        """测试渲染模板"""
        self.engine.load_template("greeting", "Hello {{name}}, you are {{age}} years old.")
        result = self.engine.render("greeting", {"name": "Alice", "age": 25})
        self.assertEqual(result, "Hello Alice, you are 25 years old.")
    
    def test_add_variable(self):
        """测试添加变量"""
        self.engine.add_variable("default_name", "World")
        self.engine.load_template("hello", "Hello {{default_name}}!")
        result = self.engine.render("hello")
        self.assertEqual(result, "Hello World!")


class TestEnvScaler(unittest.TestCase):
    """测试环境缩放代码生成引擎"""
    
    def setUp(self):
        self.scaler = EnvScaler()
    
    def test_generate_base_environment(self):
        """测试生成基础环境"""
        spec = EnvironmentSpec(
            name="BaseTestEnv",
            type="base",
            description="基础测试环境",
            tools=[
                {
                    "name": "test_action",
                    "description": "测试动作",
                    "parameters": {"param1": "str"}
                }
            ],
            state_variables={
                "counter": {"initial": 0, "type": "int"},
                "status": {"initial": "idle", "type": "str"}
            }
        )
        
        code = self.scaler.generate(spec)
        
        self.assertIsInstance(code, str)
        self.assertIn("GeneratedEnvironment", code)
        self.assertIn("BaseTestEnv", code)
        self.assertIn("test_action", code)
    
    def test_generate_ecommerce_environment(self):
        """测试生成电商环境"""
        spec = EnvironmentSpec(
            name="EcommerceTestEnv",
            type="e-commerce",
            description="电商测试环境",
            tools=[
                {"name": "search_products", "description": "搜索产品", "parameters": {"query": "str"}},
                {"name": "add_to_cart", "description": "添加到购物车", "parameters": {"product_id": "str", "quantity": "int"}},
                {"name": "checkout", "description": "结账", "parameters": {"payment_method": "str"}}
            ],
            state_variables={
                "user_balance": {"initial": 1000.0, "type": "float"},
                "cart": {"initial": {}, "type": "dict"}
            }
        )
        
        code = self.scaler.generate(spec)
        
        self.assertIsInstance(code, str)
        self.assertIn("search_products", code)
        self.assertIn("add_to_cart", code)
        self.assertIn("checkout", code)
        self.assertIn("user_balance", code)
    
    def test_generate_os_environment(self):
        """测试生成操作系统环境"""
        spec = EnvironmentSpec(
            name="OSTestEnv",
            type="operating-system",
            description="操作系统测试环境",
            tools=[
                {"name": "list_directory", "description": "列出目录", "parameters": {"path": "str"}},
                {"name": "read_file", "description": "读取文件", "parameters": {"path": "str"}},
                {"name": "write_file", "description": "写入文件", "parameters": {"path": "str", "content": "str"}}
            ],
            state_variables={
                "user": {"initial": "test_user", "type": "str"},
                "filesystem": {"initial": {}, "type": "dict"}
            }
        )
        
        code = self.scaler.generate(spec)
        
        self.assertIsInstance(code, str)
        self.assertIn("list_directory", code)
        self.assertIn("read_file", code)
        self.assertIn("write_file", code)
    
    def test_safety_sanitization(self):
        """测试安全清理"""
        spec = EnvironmentSpec(
            name="SafetyTestEnv",
            type="base",
            description="安全测试环境",
            tools=[]
        )
        
        spec.__dict__["_unsafe_code"] = '''
import os
import sys
eval("dangerous_code")
exec("dangerous_code")
'''
        
        code = self.scaler.generate(spec)
        
        self.assertIn("# Safety: os module usage restricted", code)
        self.assertIn("# Safety: sys module usage restricted", code)


class TestEnvironmentCodeValidation(unittest.TestCase):
    """测试生成的环境代码验证"""
    
    def setUp(self):
        self.scaler = EnvScaler()
    
    def test_syntax_validity(self):
        """测试生成代码的语法有效性"""
        spec = EnvironmentSpec(
            name="SyntaxTestEnv",
            type="base",
            description="语法测试环境",
            tools=[{"name": "test", "description": "测试", "parameters": {}}]
        )
        
        code = self.scaler.generate(spec)
        
        import ast
        try:
            ast.parse(code)
            valid = True
        except SyntaxError:
            valid = False
        
        self.assertTrue(valid, "生成的代码应该是语法有效的")
    
    def test_class_structure(self):
        """测试生成代码的类结构"""
        spec = EnvironmentSpec(
            name="ClassTestEnv",
            type="base",
            description="类结构测试环境",
            tools=[]
        )
        
        code = self.scaler.generate(spec)
        
        self.assertIn("class GeneratedEnvironment", code)
        self.assertIn("def __init__(self)", code)
        self.assertIn("def reset(self)", code)
        self.assertIn("def execute_tool(self", code)
        self.assertIn("def get_observation(self", code)


if __name__ == "__main__":
    unittest.main(verbosity=2)
