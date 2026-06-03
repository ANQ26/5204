"""
环境管理器 - 负责创建、加载和管理沙盒环境
"""

import os
import sys
import importlib.util
from typing import Dict, List, Optional, Any, Type
from dataclasses import dataclass, field
import uuid
import json

from src.code_generator.env_scaler import EnvScaler, EnvironmentSpec


@dataclass
class EnvironmentInstance:
    id: str
    name: str
    type: str
    spec: EnvironmentSpec
    code: str
    instance: Any = None
    created_at: float = field(default_factory=lambda: __import__('time').time())
    metadata: Dict[str, Any] = field(default_factory=dict)


class EnvironmentManager:
    """
    环境管理器
    
    核心功能：
    1. 从规格创建新环境
    2. 动态加载和编译生成的环境代码
    3. 管理环境实例的生命周期
    4. 提供环境实例的访问接口
    """
    
    def __init__(self, env_scaler: EnvScaler = None):
        self.env_scaler = env_scaler or EnvScaler()
        self.environments: Dict[str, EnvironmentInstance] = {}
        self._loaded_modules: Dict[str, Any] = {}
    
    def create_environment(self, spec: EnvironmentSpec) -> EnvironmentInstance:
        env_id = str(uuid.uuid4())[:8]
        
        code = self.env_scaler.generate(spec)
        
        instance = EnvironmentInstance(
            id=env_id,
            name=spec.name,
            type=spec.type,
            spec=spec,
            code=code,
            metadata={
                "generator_version": "1.0.0",
                "spec_hash": hash(json.dumps(spec.__dict__, default=str))
            }
        )
        
        self.environments[env_id] = instance
        return instance
    
    def load_environment(self, instance: EnvironmentInstance) -> Any:
        if instance.id in self._loaded_modules:
            return self._loaded_modules[instance.id]
        
        module_name = f"generated_env_{instance.id}"
        spec = importlib.util.spec_from_loader(module_name, loader=None)
        module = importlib.util.module_from_spec(spec)
        
        sys.modules[module_name] = module
        
        try:
            exec(instance.code, module.__dict__)
            self._loaded_modules[instance.id] = module
            
            if hasattr(module, 'GeneratedEnvironment'):
                env_class = module.GeneratedEnvironment
                instance.instance = env_class()
                return instance.instance
            else:
                raise ValueError("Generated code does not contain GeneratedEnvironment class")
                
        except Exception as e:
            del sys.modules[module_name]
            raise RuntimeError(f"Failed to load environment: {e}")
    
    def get_environment(self, env_id: str) -> Optional[EnvironmentInstance]:
        return self.environments.get(env_id)
    
    def get_environment_instance(self, env_id: str) -> Optional[Any]:
        instance = self.get_environment(env_id)
        if instance and instance.instance:
            return instance.instance
        
        if instance:
            return self.load_environment(instance)
        
        return None
    
    def list_environments(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": env.id,
                "name": env.name,
                "type": env.type,
                "created_at": env.created_at,
                "has_instance": env.instance is not None
            }
            for env in self.environments.values()
        ]
    
    def delete_environment(self, env_id: str) -> bool:
        if env_id in self.environments:
            del self.environments[env_id]
        
        if env_id in self._loaded_modules:
            module_name = f"generated_env_{env_id}"
            if module_name in sys.modules:
                del sys.modules[module_name]
            del self._loaded_modules[env_id]
        
        return True
    
    def reset_environment(self, env_id: str) -> Optional[Dict[str, Any]]:
        env_instance = self.get_environment_instance(env_id)
        if env_instance:
            return env_instance.reset()
        return None
    
    def execute_tool(self, env_id: str, tool_name: str, **parameters) -> tuple:
        env_instance = self.get_environment_instance(env_id)
        if env_instance:
            return env_instance.execute_tool(tool_name, **parameters)
        return None, -1.0, False, {"success": False, "error": "Environment not found"}
    
    def get_observation(self, env_id: str) -> Optional[Dict[str, Any]]:
        env_instance = self.get_environment_instance(env_id)
        if env_instance:
            return env_instance.get_observation()
        return None
    
    def get_available_tools(self, env_id: str) -> List[str]:
        env_instance = self.get_environment_instance(env_id)
        if env_instance:
            return env_instance.get_available_tools()
        return []
    
    def save_environment(self, env_id: str, filepath: str) -> bool:
        instance = self.get_environment(env_id)
        if not instance:
            return False
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump({
                    "id": instance.id,
                    "name": instance.name,
                    "type": instance.type,
                    "spec": instance.spec.__dict__,
                    "code": instance.code,
                    "metadata": instance.metadata
                }, f, indent=2, default=str, ensure_ascii=False)
            return True
        except Exception:
            return False
    
    def load_environment_from_file(self, filepath: str) -> Optional[EnvironmentInstance]:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            spec_dict = data["spec"]
            spec = EnvironmentSpec(
                name=spec_dict.get("name", ""),
                type=spec_dict.get("type", "base"),
                description=spec_dict.get("description", ""),
                tools=spec_dict.get("tools", []),
                state_variables=spec_dict.get("state_variables", {}),
                transitions=spec_dict.get("transitions", []),
                constraints=spec_dict.get("constraints", []),
                rewards=spec_dict.get("rewards", [])
            )
            
            instance = EnvironmentInstance(
                id=data["id"],
                name=data["name"],
                type=data["type"],
                spec=spec,
                code=data["code"],
                metadata=data.get("metadata", {})
            )
            
            self.environments[instance.id] = instance
            return instance
            
        except Exception as e:
            print(f"Failed to load environment: {e}")
            return None
    
    def create_ecommerce_environment(
        self,
        name: str = "E-Commerce Environment",
        user_balance: float = 1000.0,
        products: Dict[str, Dict] = None
    ) -> EnvironmentInstance:
        if products is None:
            products = {
                "p1": {"name": "Laptop Pro", "price": 899.99, "stock": 10, "category": "electronics"},
                "p2": {"name": "Wireless Mouse", "price": 29.99, "stock": 50, "category": "accessories"},
                "p3": {"name": "USB-C Cable", "price": 19.99, "stock": 100, "category": "accessories"},
                "p4": {"name": "Monitor 27\"", "price": 399.99, "stock": 5, "category": "electronics"},
            }
        
        inventory = {pid: p["stock"] for pid, p in products.items()}
        
        spec = EnvironmentSpec(
            name=name,
            type="e-commerce",
            description="交互式电商环境，支持产品搜索、购物车管理、结账等功能",
            tools=[
                {"name": "search_products", "description": "搜索产品", "parameters": {"query": "str", "category": "str"}},
                {"name": "add_to_cart", "description": "添加到购物车", "parameters": {"product_id": "str", "quantity": "int"}},
                {"name": "view_cart", "description": "查看购物车", "parameters": {}},
                {"name": "checkout", "description": "结账", "parameters": {"payment_method": "str"}},
                {"name": "get_product_info", "description": "获取产品信息", "parameters": {"product_id": "str"}},
            ],
            state_variables={
                "user_balance": {"initial": user_balance, "type": "float"},
                "products": {"initial": products, "type": "dict"},
                "inventory": {"initial": inventory, "type": "dict"},
                "cart": {"initial": {}, "type": "dict"},
                "orders": {"initial": [], "type": "list"},
            },
            transitions=[
                {"from": "initial", "action": "search_products", "to": "browsing"},
                {"from": "browsing", "action": "add_to_cart", "to": "cart_active"},
                {"from": "cart_active", "action": "view_cart", "to": "cart_active"},
                {"from": "cart_active", "action": "add_to_cart", "to": "cart_active"},
                {"from": "cart_active", "action": "checkout", "to": "terminal"},
            ],
            constraints=[
                {"name": "positive_quantity", "check": "lambda self, tool, params: params.get('quantity', 0) > 0 if tool == 'add_to_cart' else True", "penalty": -1},
                {"name": "valid_product", "check": "lambda self, tool, params: params.get('product_id') in self.state.get('products', {}) if tool in ['add_to_cart', 'get_product_info'] else True", "penalty": -1},
            ],
            rewards=[
                {"condition": "lambda self, tool, params, result: tool == 'checkout' and result.get('status') == 'success'", "value": 10.0, "description": "成功结账"},
                {"condition": "lambda self, tool, params, result: tool == 'add_to_cart' and result.get('status') == 'success'", "value": 1.0, "description": "成功添加商品"},
                {"condition": "lambda self, tool, params, result: 'error' in result", "value": -0.5, "description": "执行错误"},
            ]
        )
        
        return self.create_environment(spec)
    
    def create_os_environment(
        self,
        name: str = "Operating System Environment",
        user: str = "user",
        permissions: Dict = None
    ) -> EnvironmentInstance:
        if permissions is None:
            permissions = {"allowed": ["read", "write", "execute", "delete"]}
        
        filesystem = {
            "home": {
                "user": {
                    "documents": {
                        "readme.txt": "Welcome to the simulated OS environment!\n",
                        "notes.md": "# My Notes\n\nThis is a simulated file system.\n"
                    },
                    "projects": {},
                    "downloads": {}
                }
            },
            "etc": {
                "passwd": "root:x:0:0:root:/root:/bin/bash\nuser:x:1000:1000:User:/home/user:/bin/bash\n"
            },
            "tmp": {}
        }
        
        spec = EnvironmentSpec(
            name=name,
            type="operating-system",
            description="交互式操作系统环境，支持文件操作、目录管理、命令执行等功能",
            tools=[
                {"name": "list_directory", "description": "列出目录内容", "parameters": {"path": "str"}},
                {"name": "read_file", "description": "读取文件内容", "parameters": {"path": "str"}},
                {"name": "write_file", "description": "写入文件", "parameters": {"path": "str", "content": "str"}},
                {"name": "create_directory", "description": "创建目录", "parameters": {"path": "str"}},
                {"name": "delete_file", "description": "删除文件/目录", "parameters": {"path": "str"}},
                {"name": "execute_command", "description": "执行命令", "parameters": {"command": "str"}},
            ],
            state_variables={
                "user": {"initial": user, "type": "str"},
                "user_role": {"initial": "user", "type": "str"},
                "permissions": {"initial": permissions, "type": "dict"},
                "filesystem": {"initial": filesystem, "type": "dict"},
                "current_dir": {"initial": "/home/user", "type": "str"},
                "command_history": {"initial": [], "type": "list"},
            },
            transitions=[
                {"from": "initial", "action": "*", "to": "active"},
                {"from": "active", "action": "*", "to": "active"},
            ],
            constraints=[
                {"name": "valid_path", "check": "lambda self, tool, params: not ('..' in params.get('path', '') or params.get('path', '').startswith('/')) if tool in ['list_directory', 'read_file', 'write_file', 'create_directory', 'delete_file'] else True", "penalty": -1},
                {"name": "permission_check", "check": "lambda self, tool, params: self._validate_permissions(tool.split('_')[0] if '_' in tool else tool) if tool in ['write_file', 'delete_file', 'execute_command'] else True", "penalty": -2},
            ],
            rewards=[
                {"condition": "lambda self, tool, params, result: result.get('status') == 'success'", "value": 0.5, "description": "成功执行操作"},
                {"condition": "lambda self, tool, params, result: 'error' in result", "value": -0.5, "description": "执行错误"},
                {"condition": "lambda self, tool, params, result: tool == 'execute_command' and result.get('exit_code') == 0", "value": 1.0, "description": "成功执行命令"},
            ]
        )
        
        return self.create_environment(spec)
