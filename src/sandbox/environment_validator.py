"""
环境验证器 - 负责验证生成的环境代码的逻辑一致性和安全性
"""

import ast
import re
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum


class ValidationSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationIssue:
    severity: ValidationSeverity
    message: str
    code_snippet: Optional[str] = None
    location: Optional[Tuple[int, int]] = None  # (line, column)
    category: str = "general"


@dataclass
class ValidationResult:
    valid: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def errors(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity in (ValidationSeverity.ERROR, ValidationSeverity.CRITICAL)]
    
    def warnings(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == ValidationSeverity.WARNING]
    
    def info(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == ValidationSeverity.INFO]


class EnvironmentValidator:
    """
    环境验证器
    
    核心功能：
    1. 验证生成的Python代码语法正确性
    2. 检查潜在的安全风险
    3. 验证环境逻辑一致性
    4. 检测状态转换的完整性
    """
    
    DANGEROUS_MODULES = {'os', 'sys', 'subprocess', 'shutil', 'ctypes', 'importlib'}
    DANGEROUS_FUNCTIONS = {'eval', 'exec', 'compile', '__import__', 'open'}
    RESTRICTED_ATTRIBUTES = {'__import__', '__delattr__', '__setattr__', '__getattr__', '__dict__'}
    
    def __init__(self):
        self.issues: List[ValidationIssue] = []
    
    def validate(self, code: str, spec: Dict = None) -> ValidationResult:
        self.issues = []
        spec = spec or {}
        
        self._validate_syntax(code)
        
        if not any(i.severity == ValidationSeverity.CRITICAL for i in self.issues):
            self._validate_safety(code)
            self._validate_logical_consistency(code, spec)
            self._validate_state_transitions(code, spec)
        
        is_valid = len([i for i in self.issues if i.severity in (ValidationSeverity.ERROR, ValidationSeverity.CRITICAL)]) == 0
        
        return ValidationResult(
            valid=is_valid,
            issues=self.issues.copy(),
            metadata={
                "total_checks": len(self.issues),
                "errors": len([i for i in self.issues if i.severity in (ValidationSeverity.ERROR, ValidationSeverity.CRITICAL)]),
                "warnings": len([i for i in self.issues if i.severity == ValidationSeverity.WARNING])
            }
        )
    
    def _validate_syntax(self, code: str):
        try:
            ast.parse(code)
            self.issues.append(ValidationIssue(
                severity=ValidationSeverity.INFO,
                message="代码语法检查通过",
                category="syntax"
            ))
        except SyntaxError as e:
            self.issues.append(ValidationIssue(
                severity=ValidationSeverity.CRITICAL,
                message=f"语法错误: {e.msg}",
                location=(e.lineno, e.offset),
                code_snippet=e.text.strip() if e.text else None,
                category="syntax"
            ))
    
    def _validate_safety(self, code: str):
        tree = ast.parse(code)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for name in node.names:
                    module_name = name.name.split('.')[0]
                    if module_name in self.DANGEROUS_MODULES:
                        self.issues.append(ValidationIssue(
                            severity=ValidationSeverity.ERROR,
                            message=f"危险模块导入: {name.name}",
                            location=(node.lineno, node.col_offset),
                            category="safety"
                        ))
            
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module.split('.')[0] in self.DANGEROUS_MODULES:
                    self.issues.append(ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        message=f"从危险模块导入: {node.module}",
                        location=(node.lineno, node.col_offset),
                        category="safety"
                    ))
            
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                    if func_name in self.DANGEROUS_FUNCTIONS:
                        self.issues.append(ValidationIssue(
                            severity=ValidationSeverity.CRITICAL,
                            message=f"危险函数调用: {func_name}",
                            location=(node.lineno, node.col_offset),
                            category="safety"
                        ))
                
                elif isinstance(node.func, ast.Attribute):
                    attr_name = node.func.attr
                    if attr_name in self.RESTRICTED_ATTRIBUTES:
                        self.issues.append(ValidationIssue(
                            severity=ValidationSeverity.WARNING,
                            message=f"潜在危险的属性访问: {attr_name}",
                            location=(node.lineno, node.col_offset),
                            category="safety"
                        ))
            
            elif isinstance(node, ast.Attribute):
                if node.attr in self.RESTRICTED_ATTRIBUTES:
                    self.issues.append(ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message=f"访问受限属性: {node.attr}",
                        location=(node.lineno, node.col_offset),
                        category="safety"
                    ))
    
    def _validate_logical_consistency(self, code: str, spec: Dict):
        tree = ast.parse(code)
        
        class_names = []
        method_names = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_names.append(node.name)
                for body_item in node.body:
                    if isinstance(body_item, ast.FunctionDef):
                        method_names.append(body_item.name)
        
        required_methods = ['reset', 'execute_tool', 'get_observation']
        missing_methods = [m for m in required_methods if m not in method_names]
        
        if missing_methods:
            self.issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                message=f"缺少必需的方法: {', '.join(missing_methods)}",
                category="logic"
            ))
        else:
            self.issues.append(ValidationIssue(
                severity=ValidationSeverity.INFO,
                message="所有必需方法已实现",
                category="logic"
            ))
        
        state_vars = self._extract_state_variables(code)
        if state_vars:
            self.issues.append(ValidationIssue(
                severity=ValidationSeverity.INFO,
                message=f"检测到状态变量: {', '.join(state_vars)}",
                category="logic"
            ))
        
        tools = self._extract_tools(code)
        if tools:
            self.issues.append(ValidationIssue(
                severity=ValidationSeverity.INFO,
                message=f"检测到工具定义: {', '.join(tools)}",
                category="logic"
            ))
    
    def _validate_state_transitions(self, code: str, spec: Dict):
        tree = ast.parse(code)
        
        state_assignments = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Attribute):
                        if target.attr in ['current_state', 'state']:
                            state_assignments.append({
                                'lineno': node.lineno,
                                'col_offset': node.col_offset
                            })
        
        if state_assignments:
            self.issues.append(ValidationIssue(
                severity=ValidationSeverity.INFO,
                message=f"检测到 {len(state_assignments)} 处状态修改点",
                category="state_machine"
            ))
        
        if spec.get('transitions'):
            transitions = spec.get('transitions', [])
            from_states = set()
            to_states = set()
            
            for t in transitions:
                from_states.add(t.get('from', '*'))
                to_states.add(t.get('to'))
            
            if '*' not in from_states:
                self.issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    message="状态机没有通配符转换，某些状态可能无法转移",
                    category="state_machine"
                ))
        
        self._validate_terminal_states(code)
    
    def _validate_terminal_states(self, code: str):
        tree = ast.parse(code)
        
        terminal_checks = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == '_check_terminal_condition':
                for body_item in ast.walk(node):
                    if isinstance(body_item, ast.Compare):
                        terminal_checks.append({
                            'lineno': body_item.lineno
                        })
        
        if terminal_checks:
            self.issues.append(ValidationIssue(
                severity=ValidationSeverity.INFO,
                message=f"检测到 {len(terminal_checks)} 个终止条件",
                category="state_machine"
            ))
        else:
            self.issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                message="未检测到明确的终止条件，环境可能永远运行",
                category="state_machine"
            ))
    
    def _extract_state_variables(self, code: str) -> List[str]:
        variables = []
        pattern = r'self\.state\.get\(["\'](\w+)["\']'
        matches = re.findall(pattern, code)
        variables.extend(matches)
        
        pattern = r'"(\w+)":\s*\{[^}]*"initial"'
        matches = re.findall(pattern, code)
        variables.extend(matches)
        
        return list(set(variables))
    
    def _extract_tools(self, code: str) -> List[str]:
        tools = []
        pattern = r'"(\w+)":\s*\{[^}]*"description"'
        matches = re.findall(pattern, code, re.DOTALL)
        tools.extend(matches)
        
        pattern = r'tool_name\s*==\s*["\'](\w+)["\']'
        matches = re.findall(pattern, code)
        tools.extend(matches)
        
        return list(set(tools))
    
    def validate_spec(self, spec: Dict) -> ValidationResult:
        self.issues = []
        
        required_fields = ['name', 'type', 'tools']
        missing = [f for f in required_fields if f not in spec]
        
        if missing:
            self.issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                message=f"规格缺少必需字段: {', '.join(missing)}",
                category="spec"
            ))
        
        if 'tools' in spec:
            tools = spec['tools']
            if not isinstance(tools, list):
                self.issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    message="tools 必须是列表",
                    category="spec"
                ))
            else:
                for i, tool in enumerate(tools):
                    if not isinstance(tool, dict):
                        self.issues.append(ValidationIssue(
                            severity=ValidationSeverity.ERROR,
                            message=f"工具 {i} 必须是字典",
                            category="spec"
                        ))
                    elif 'name' not in tool:
                        self.issues.append(ValidationIssue(
                            severity=ValidationSeverity.ERROR,
                            message=f"工具 {i} 缺少 name 字段",
                            category="spec"
                        ))
        
        if 'state_variables' in spec:
            for var_name, var_spec in spec['state_variables'].items():
                if 'type' not in var_spec:
                    self.issues.append(ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message=f"状态变量 {var_name} 未指定类型",
                        category="spec"
                    ))
        
        is_valid = len([i for i in self.issues if i.severity in (ValidationSeverity.ERROR, ValidationSeverity.CRITICAL)]) == 0
        
        return ValidationResult(
            valid=is_valid,
            issues=self.issues.copy(),
            metadata={"spec_validation": True}
        )
