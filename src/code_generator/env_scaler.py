"""
EnvScaler - 环境缩放代码生成引擎

基于EnvScaler思路，通过结构化的模板和参数化配置，
自动生成具备逻辑一致性的交互式沙盒环境代码。
"""

import os
import re
import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod


@dataclass
class EnvironmentSpec:
    name: str
    type: str  # e.g., "e-commerce", "operating-system", "game"
    description: str
    tools: List[Dict[str, Any]] = field(default_factory=list)
    state_variables: Dict[str, Any] = field(default_factory=dict)
    transitions: List[Dict[str, Any]] = field(default_factory=list)
    constraints: List[Dict[str, Any]] = field(default_factory=list)
    rewards: List[Dict[str, Any]] = field(default_factory=list)


class BaseCodeGenerator(ABC):
    @abstractmethod
    def generate(self, spec: EnvironmentSpec) -> str:
        pass


class EnvScaler(BaseCodeGenerator):
    """
    环境缩放代码生成引擎
    
    核心功能：
    1. 基于环境规格自动生成完整的Python环境代码
    2. 确保生成的环境具有逻辑一致性
    3. 支持多类型环境（电商、操作系统等）
    """
    
    def __init__(self, template_engine=None):
        self.template_engine = template_engine or TemplateEngine()
        self._load_builtin_templates()
    
    def _load_builtin_templates(self):
        self.builtin_templates = {
            "e-commerce": self._get_ecommerce_template(),
            "operating-system": self._get_os_template(),
            "base": self._get_base_template()
        }
    
    def generate(self, spec: EnvironmentSpec) -> str:
        template_type = spec.type if spec.type in self.builtin_templates else "base"
        template = self.builtin_templates[template_type]
        
        code = self._fill_template(spec, template)
        validated_code = self._validate_and_sanitize(code, spec)
        
        return validated_code
    
    def _fill_template(self, spec: EnvironmentSpec, template: str) -> str:
        tools_code = self._generate_tools_code(spec.tools)
        state_code = self._generate_state_code(spec.state_variables)
        transition_code = self._generate_transitions_code(spec.transitions)
        constraint_code = self._generate_constraints_code(spec.constraints)
        reward_code = self._generate_rewards_code(spec.rewards)
        
        filled = template.replace("{{ENV_NAME}}", spec.name)
        filled = filled.replace("{{ENV_DESCRIPTION}}", spec.description)
        filled = filled.replace("{{TOOLS_CODE}}", tools_code)
        filled = filled.replace("{{STATE_VARIABLES}}", state_code)
        filled = filled.replace("{{TRANSITIONS_CODE}}", transition_code)
        filled = filled.replace("{{CONSTRAINTS_CODE}}", constraint_code)
        filled = filled.replace("{{REWARDS_CODE}}", reward_code)
        
        return filled
    
    def _generate_tools_code(self, tools: List[Dict]) -> str:
        if not tools:
            return "        self.tools = {}"
        
        tools_list = []
        for tool in tools:
            name = tool.get("name", "unknown_tool")
            description = tool.get("description", "")
            parameters = tool.get("parameters", {})
            implementation = tool.get("implementation", self._default_tool_implementation(name, parameters))
            
            tool_def = f'''
        "{name}": {{
            "description": "{description}",
            "parameters": {json.dumps(parameters)},
            "implementation": {implementation}
        }}'''
            tools_list.append(tool_def)
        
        return "        self.tools = {" + ", ".join(tools_list) + "}"
    
    def _generate_state_code(self, state_vars: Dict[str, Any]) -> str:
        if not state_vars:
            return "        self.state = {}"
        
        state_list = []
        for var_name, var_spec in state_vars.items():
            initial_value = var_spec.get("initial", "None")
            data_type = var_spec.get("type", "any")
            
            if data_type == "str":
                initial_value = f'"{initial_value}"' if initial_value != "None" else "None"
            elif data_type == "int":
                initial_value = str(initial_value) if initial_value != "None" else "None"
            elif data_type == "float":
                initial_value = str(initial_value) if initial_value != "None" else "None"
            elif data_type == "list":
                initial_value = "[]" if initial_value == "None" else initial_value
            elif data_type == "dict":
                initial_value = "{}" if initial_value == "None" else initial_value
            
            state_list.append(f'"{var_name}": {initial_value}')
        
        return "        self.state = {" + ", ".join(state_list) + "}"
    
    def _generate_transitions_code(self, transitions: List[Dict]) -> str:
        if not transitions:
            return "        self.transitions = {}"
        
        transition_list = []
        for transition in transitions:
            from_state = transition.get("from", "*")
            action = transition.get("action", "unknown")
            to_state = transition.get("to", from_state)
            condition = transition.get("condition", "True")
            effect = transition.get("effect", "pass")
            
            transition_def = f'''
        ("{from_state}", "{action}"): {{
            "to_state": "{to_state}",
            "condition": {condition},
            "effect": {effect}
        }}'''
            transition_list.append(transition_def)
        
        return "        self.transitions = {" + ", ".join(transition_list) + "}"
    
    def _generate_constraints_code(self, constraints: List[Dict]) -> str:
        if not constraints:
            return "        self.constraints = []"
        
        constraint_list = []
        for constraint in constraints:
            name = constraint.get("name", "constraint")
            check = constraint.get("check", "True")
            penalty = constraint.get("penalty", 0)
            
            constraint_def = f'''
        {{
            "name": "{name}",
            "check": {check},
            "penalty": {penalty}
        }}'''
            constraint_list.append(constraint_def)
        
        return "        self.constraints = [" + ", ".join(constraint_list) + "]"
    
    def _generate_rewards_code(self, rewards: List[Dict]) -> str:
        if not rewards:
            return "        self.rewards = {}"
        
        reward_list = []
        for reward in rewards:
            condition = reward.get("condition", "False")
            value = reward.get("value", 0)
            description = reward.get("description", "")
            
            reward_def = f'''
        "{description}": {{
            "condition": {condition},
            "value": {value}
        }}'''
            reward_list.append(reward_def)
        
        return "        self.rewards = {" + ", ".join(reward_list) + "}"
    
    def _default_tool_implementation(self, name: str, parameters: Dict) -> str:
        params = ", ".join([f'{k}' for k in parameters.keys()]) if parameters else ""
        return f'''lambda self, {params}: self._default_{name}_implementation({params})'''
    
    def _validate_and_sanitize(self, code: str, spec: EnvironmentSpec) -> str:
        if spec.type == "e-commerce":
            code = self._add_ecommerce_validations(code)
        elif spec.type == "operating-system":
            code = self._add_os_validations(code)
        
        safety_checks = [
            ("import os", "# Safety: os module usage restricted"),
            ("import sys", "# Safety: sys module usage restricted"),
            ("__import__", "# Safety: dynamic import disabled"),
            ("eval(", "# Safety: eval disabled"),
            ("exec(", "# Safety: exec disabled")
        ]
        
        for dangerous, replacement in safety_checks:
            if dangerous in code:
                code = code.replace(dangerous, replacement)
        
        return code
    
    def _add_ecommerce_validations(self, code: str) -> str:
        validation_code = '''
    def _validate_transaction(self, amount: float) -> bool:
        if amount <= 0:
            return False
        if self.state.get("user_balance", 0) < amount:
            return False
        return True
    
    def _validate_inventory(self, product_id: str, quantity: int) -> bool:
        inventory = self.state.get("inventory", {})
        if product_id not in inventory:
            return False
        if inventory[product_id] < quantity:
            return False
        return True
'''
        if "class" in code and "def __init__" in code:
            class_end = code.find("def _validate_transaction")
            if class_end == -1:
                class_end = code.rfind("}")
                code = code[:class_end] + validation_code + code[class_end:]
        
        return code
    
    def _add_os_validations(self, code: str) -> str:
        validation_code = '''
    def _validate_file_path(self, path: str) -> bool:
        if not path or not isinstance(path, str):
            return False
        if ".." in path or path.startswith("/"):
            return False
        return True
    
    def _validate_permissions(self, operation: str) -> bool:
        user_permissions = self.state.get("permissions", {})
        if operation in user_permissions.get("allowed", []):
            return True
        if "admin" in self.state.get("user_role", ""):
            return True
        return False
'''
        if "class" in code and "def __init__" in code:
            class_end = code.find("def _validate_file_path")
            if class_end == -1:
                class_end = code.rfind("}")
                code = code[:class_end] + validation_code + code[class_end:]
        
        return code
    
    def _get_base_template(self) -> str:
        return '''
"""
自动生成的环境代码 - {{ENV_NAME}}
描述: {{ENV_DESCRIPTION}}
"""

import json
from typing import Dict, List, Any, Optional, Tuple


class GeneratedEnvironment:
    def __init__(self):
        self.name = "{{ENV_NAME}}"
        self.description = "{{ENV_DESCRIPTION}}"
        
{{STATE_VARIABLES}}
        
{{TOOLS_CODE}}
        
{{TRANSITIONS_CODE}}
        
{{CONSTRAINTS_CODE}}
        
{{REWARDS_CODE}}
        
        self.current_state = "initial"
        self.history = []
        self.done = False
    
    def reset(self) -> Dict[str, Any]:
        self.current_state = "initial"
        self.history = []
        self.done = False
        
{{STATE_VARIABLES}}
        
        return self.get_observation()
    
    def get_observation(self) -> Dict[str, Any]:
        return {
            "current_state": self.current_state,
            "state_variables": self.state.copy(),
            "available_tools": list(self.tools.keys()),
            "history": self.history[-10:] if len(self.history) > 10 else self.history
        }
    
    def execute_tool(self, tool_name: str, **parameters) -> Tuple[Dict[str, Any], float, bool, Dict]:
        if tool_name not in self.tools:
            return self._error_response(f"工具 '{tool_name}' 不存在")
        
        tool = self.tools[tool_name]
        
        if not self._check_permissions(tool_name):
            return self._error_response(f"无权限执行工具 '{tool_name}'")
        
        if not self._check_constraints(tool_name, parameters):
            return self._error_response(f"执行工具 '{tool_name}' 违反约束条件")
        
        try:
            result = self._invoke_tool(tool_name, **parameters)
            reward = self._calculate_reward(tool_name, parameters, result)
            
            self._update_history(tool_name, parameters, result, reward)
            self._update_state(tool_name, parameters, result)
            
            done = self._check_terminal_condition()
            
            return result, reward, done, {"success": True}
            
        except Exception as e:
            return self._error_response(f"执行工具 '{tool_name}' 出错: {str(e)}")
    
    def _invoke_tool(self, tool_name: str, **parameters) -> Dict[str, Any]:
        tool = self.tools[tool_name]
        implementation = tool.get("implementation")
        
        if callable(implementation):
            result = implementation(self, **parameters)
        else:
            result = self._default_tool_execution(tool_name, **parameters)
        
        return result if isinstance(result, dict) else {"result": result}
    
    def _default_tool_execution(self, tool_name: str, **parameters) -> Dict[str, Any]:
        return {
            "tool": tool_name,
            "parameters": parameters,
            "status": "executed",
            "message": f"工具 '{tool_name}' 已执行"
        }
    
    def _check_permissions(self, tool_name: str) -> bool:
        return True
    
    def _check_constraints(self, tool_name: str, parameters: Dict) -> bool:
        for constraint in self.constraints:
            check = constraint.get("check")
            if callable(check):
                if not check(self, tool_name, parameters):
                    return False
            elif isinstance(check, str):
                try:
                    namespace = {"self": self, "tool_name": tool_name, "parameters": parameters}
                    if not eval(check, namespace):
                        return False
                except:
                    pass
        
        return True
    
    def _calculate_reward(self, tool_name: str, parameters: Dict, result: Dict) -> float:
        total_reward = 0.0
        
        for reward_name, reward_spec in self.rewards.items():
            condition = reward_spec.get("condition")
            value = reward_spec.get("value", 0)
            
            if callable(condition):
                if condition(self, tool_name, parameters, result):
                    total_reward += value
            elif isinstance(condition, str):
                try:
                    namespace = {
                        "self": self, 
                        "tool_name": tool_name, 
                        "parameters": parameters,
                        "result": result
                    }
                    if eval(condition, namespace):
                        total_reward += value
                except:
                    pass
        
        return total_reward
    
    def _update_history(self, tool_name: str, parameters: Dict, result: Dict, reward: float):
        self.history.append({
            "step": len(self.history),
            "tool": tool_name,
            "parameters": parameters,
            "result": result,
            "reward": reward,
            "state": self.current_state
        })
    
    def _update_state(self, tool_name: str, parameters: Dict, result: Dict):
        transition_key = (self.current_state, tool_name)
        if transition_key in self.transitions:
            transition = self.transitions[transition_key]
            condition = transition.get("condition")
            
            if condition is True or callable(condition) and condition(self):
                self.current_state = transition.get("to_state", self.current_state)
                effect = transition.get("effect")
                if effect and callable(effect):
                    effect(self, parameters, result)
        
        elif ("*", tool_name) in self.transitions:
            transition = self.transitions[("*", tool_name)]
            self.current_state = transition.get("to_state", self.current_state)
    
    def _check_terminal_condition(self) -> bool:
        if self.current_state == "terminal":
            return True
        
        if len(self.history) > 100:
            return True
        
        return False
    
    def _error_response(self, message: str) -> Tuple[Dict[str, Any], float, bool, Dict]:
        return {
            "error": message,
            "status": "failed"
        }, -1.0, False, {"success": False}
    
    def get_available_tools(self) -> List[str]:
        return list(self.tools.keys())
    
    def get_tool_info(self, tool_name: str) -> Optional[Dict]:
        return self.tools.get(tool_name)
    
    def render(self) -> str:
        return f"""
环境: {self.name}
当前状态: {self.current_state}
步骤数: {len(self.history)}
状态变量: {json.dumps(self.state, indent=2, ensure_ascii=False)}
"""
    
    def __str__(self) -> str:
        return self.render()
'''
    
    def _get_ecommerce_template(self) -> str:
        base_template = self._get_base_template()
        
        ecommerce_extensions = '''
    def _default_tool_execution(self, tool_name: str, **parameters) -> Dict[str, Any]:
        if tool_name == "search_products":
            return self._search_products(**parameters)
        elif tool_name == "add_to_cart":
            return self._add_to_cart(**parameters)
        elif tool_name == "checkout":
            return self._checkout(**parameters)
        elif tool_name == "view_cart":
            return self._view_cart(**parameters)
        elif tool_name == "get_product_info":
            return self._get_product_info(**parameters)
        else:
            return super()._default_tool_execution(tool_name, **parameters)
    
    def _search_products(self, query: str, category: str = None) -> Dict[str, Any]:
        products = self.state.get("products", {})
        results = []
        
        for product_id, product in products.items():
            if query.lower() in product.get("name", "").lower():
                if category is None or product.get("category") == category:
                    results.append({
                        "id": product_id,
                        "name": product.get("name"),
                        "price": product.get("price"),
                        "stock": product.get("stock", 0)
                    })
        
        return {
            "query": query,
            "category": category,
            "results": results,
            "count": len(results)
        }
    
    def _add_to_cart(self, product_id: str, quantity: int = 1) -> Dict[str, Any]:
        if not self._validate_inventory(product_id, quantity):
            return {"error": "库存不足或产品不存在", "status": "failed"}
        
        cart = self.state.get("cart", {})
        product = self.state.get("products", {}).get(product_id, {})
        
        if product_id in cart:
            cart[product_id]["quantity"] += quantity
        else:
            cart[product_id] = {
                "name": product.get("name"),
                "price": product.get("price"),
                "quantity": quantity
            }
        
        self.state["cart"] = cart
        
        return {
            "product_id": product_id,
            "quantity": quantity,
            "cart": cart,
            "status": "success"
        }
    
    def _view_cart(self) -> Dict[str, Any]:
        cart = self.state.get("cart", {})
        total = sum(item["price"] * item["quantity"] for item in cart.values())
        
        return {
            "cart": cart,
            "item_count": sum(item["quantity"] for item in cart.values()),
            "total_amount": total,
            "status": "success"
        }
    
    def _checkout(self, payment_method: str = "credit_card") -> Dict[str, Any]:
        cart = self.state.get("cart", {})
        if not cart:
            return {"error": "购物车为空", "status": "failed"}
        
        total = sum(item["price"] * item["quantity"] for item in cart.values())
        
        if not self._validate_transaction(total):
            return {"error": "余额不足", "status": "failed"}
        
        self.state["user_balance"] = self.state.get("user_balance", 0) - total
        
        inventory = self.state.get("inventory", {})
        for product_id, item in cart.items():
            if product_id in inventory:
                inventory[product_id] -= item["quantity"]
        
        order = {
            "id": f"ORD_{len(self.history)}",
            "items": cart.copy(),
            "total": total,
            "payment_method": payment_method,
            "status": "confirmed"
        }
        
        orders = self.state.get("orders", [])
        orders.append(order)
        self.state["orders"] = orders
        self.state["cart"] = {}
        
        return {
            "order": order,
            "status": "success",
            "message": "订单已确认"
        }
    
    def _get_product_info(self, product_id: str) -> Dict[str, Any]:
        products = self.state.get("products", {})
        if product_id not in products:
            return {"error": "产品不存在", "status": "failed"}
        
        product = products[product_id]
        return {
            "product_id": product_id,
            **product,
            "status": "success"
        }
'''
        
        return base_template.replace(
            "def _default_tool_execution(self, tool_name: str, **parameters) -> Dict[str, Any]:",
            ecommerce_extensions
        )
    
    def _get_os_template(self) -> str:
        base_template = self._get_base_template()
        
        os_extensions = '''
    def _default_tool_execution(self, tool_name: str, **parameters) -> Dict[str, Any]:
        if tool_name == "list_directory":
            return self._list_directory(**parameters)
        elif tool_name == "read_file":
            return self._read_file(**parameters)
        elif tool_name == "write_file":
            return self._write_file(**parameters)
        elif tool_name == "create_directory":
            return self._create_directory(**parameters)
        elif tool_name == "delete_file":
            return self._delete_file(**parameters)
        elif tool_name == "execute_command":
            return self._execute_command(**parameters)
        else:
            return super()._default_tool_execution(tool_name, **parameters)
    
    def _list_directory(self, path: str = ".") -> Dict[str, Any]:
        if not self._validate_file_path(path):
            return {"error": "无效的路径", "status": "failed"}
        
        filesystem = self.state.get("filesystem", {})
        current_dir = self._get_directory(path)
        
        if current_dir is None:
            return {"error": "目录不存在", "status": "failed"}
        
        contents = []
        for name, item in current_dir.items():
            item_type = "directory" if isinstance(item, dict) else "file"
            contents.append({
                "name": name,
                "type": item_type,
                "size": len(item) if item_type == "file" else None
            })
        
        return {
            "path": path,
            "contents": contents,
            "count": len(contents),
            "status": "success"
        }
    
    def _read_file(self, path: str) -> Dict[str, Any]:
        if not self._validate_file_path(path):
            return {"error": "无效的路径", "status": "failed"}
        
        file_content = self._get_file_content(path)
        
        if file_content is None:
            return {"error": "文件不存在", "status": "failed"}
        
        return {
            "path": path,
            "content": file_content,
            "size": len(file_content),
            "status": "success"
        }
    
    def _write_file(self, path: str, content: str) -> Dict[str, Any]:
        if not self._validate_file_path(path):
            return {"error": "无效的路径", "status": "failed"}
        
        if not self._validate_permissions("write"):
            return {"error": "无写入权限", "status": "failed"}
        
        self._set_file_content(path, content)
        
        return {
            "path": path,
            "size": len(content),
            "status": "success",
            "message": "文件已写入"
        }
    
    def _create_directory(self, path: str) -> Dict[str, Any]:
        if not self._validate_file_path(path):
            return {"error": "无效的路径", "status": "failed"}
        
        if not self._validate_permissions("write"):
            return {"error": "无写入权限", "status": "failed"}
        
        if self._get_directory(path) is not None:
            return {"error": "目录已存在", "status": "failed"}
        
        self._create_dir(path)
        
        return {
            "path": path,
            "status": "success",
            "message": "目录已创建"
        }
    
    def _delete_file(self, path: str) -> Dict[str, Any]:
        if not self._validate_file_path(path):
            return {"error": "无效的路径", "status": "failed"}
        
        if not self._validate_permissions("delete"):
            return {"error": "无删除权限", "status": "failed"}
        
        if self._get_file_content(path) is None and self._get_directory(path) is None:
            return {"error": "文件或目录不存在", "status": "failed"}
        
        self._delete_item(path)
        
        return {
            "path": path,
            "status": "success",
            "message": "已删除"
        }
    
    def _execute_command(self, command: str) -> Dict[str, Any]:
        if not self._validate_permissions("execute"):
            return {"error": "无执行权限", "status": "failed"}
        
        output = self._simulate_command_output(command)
        
        return {
            "command": command,
            "output": output,
            "exit_code": 0,
            "status": "success"
        }
    
    def _get_directory(self, path: str) -> Optional[Dict]:
        filesystem = self.state.get("filesystem", {})
        parts = path.strip("/").split("/") if path != "." else []
        
        current = filesystem
        for part in parts:
            if part == "" or part == ".":
                continue
            if part not in current:
                return None
            if not isinstance(current[part], dict):
                return None
            current = current[part]
        
        return current
    
    def _get_file_content(self, path: str) -> Optional[str]:
        filesystem = self.state.get("filesystem", {})
        parts = path.strip("/").split("/")
        
        if len(parts) == 0:
            return None
        
        filename = parts[-1]
        dir_path = "/".join(parts[:-1]) if len(parts) > 1 else "."
        
        directory = self._get_directory(dir_path)
        if directory is None or filename not in directory:
            return None
        
        if isinstance(directory[filename], dict):
            return None
        
        return directory[filename]
    
    def _set_file_content(self, path: str, content: str):
        filesystem = self.state.setdefault("filesystem", {})
        parts = path.strip("/").split("/")
        
        if len(parts) == 0:
            return
        
        filename = parts[-1]
        dir_path = "/".join(parts[:-1]) if len(parts) > 1 else "."
        
        directory = self._get_or_create_directory(dir_path)
        directory[filename] = content
    
    def _get_or_create_directory(self, path: str) -> Dict:
        filesystem = self.state.setdefault("filesystem", {})
        parts = path.strip("/").split("/") if path != "." else []
        
        current = filesystem
        for part in parts:
            if part == "" or part == ".":
                continue
            if part not in current:
                current[part] = {}
            if not isinstance(current[part], dict):
                current[part] = {}
            current = current[part]
        
        return current
    
    def _create_dir(self, path: str):
        self._get_or_create_directory(path)
    
    def _delete_item(self, path: str):
        filesystem = self.state.get("filesystem", {})
        parts = path.strip("/").split("/")
        
        if len(parts) == 0:
            return
        
        item_name = parts[-1]
        dir_path = "/".join(parts[:-1]) if len(parts) > 1 else "."
        
        directory = self._get_directory(dir_path)
        if directory and item_name in directory:
            del directory[item_name]
    
    def _simulate_command_output(self, command: str) -> str:
        cmd_parts = command.split()
        if not cmd_parts:
            return ""
        
        cmd = cmd_parts[0]
        
        if cmd == "pwd":
            return self.state.get("current_dir", "/")
        elif cmd == "echo":
            return " ".join(cmd_parts[1:])
        elif cmd == "whoami":
            return self.state.get("user", "guest")
        elif cmd == "date":
            return "Sat Jan 01 00:00:00 UTC 2026"
        elif cmd == "cat" and len(cmd_parts) > 1:
            content = self._get_file_content(cmd_parts[1])
            return content if content else f"cat: {cmd_parts[1]}: No such file or directory"
        else:
            return f"Command '{cmd}' not found or not implemented"
'''
        
        return base_template.replace(
            "def _default_tool_execution(self, tool_name: str, **parameters) -> Dict[str, Any]:",
            os_extensions
        )


class TemplateEngine:
    """
    模板引擎 - 用于管理和渲染环境代码模板
    """
    
    def __init__(self):
        self.templates = {}
        self.variables = {}
    
    def load_template(self, name: str, content: str):
        self.templates[name] = content
    
    def render(self, template_name: str, variables: Dict[str, Any] = None) -> str:
        if template_name not in self.templates:
            raise ValueError(f"Template '{template_name}' not found")
        
        content = self.templates[template_name]
        vars_to_use = variables or self.variables
        
        for key, value in vars_to_use.items():
            placeholder = "{{" + key + "}}"
            content = content.replace(placeholder, str(value))
        
        return content
    
    def add_variable(self, key: str, value: Any):
        self.variables[key] = value
    
    def get_template(self, name: str) -> Optional[str]:
        return self.templates.get(name)
    
    def list_templates(self) -> List[str]:
        return list(self.templates.keys())
