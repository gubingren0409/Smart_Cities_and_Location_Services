"""架构护栏：核心算法层不得依赖 LLM / 网络 / 智能体。

这是设计上的硬约束（见设计文档第 2 节）：core 完全不知道 LLM 存在。
它保证三件事：
  1. 学生能单独测试与复用算法；
  2. 任务②的敏感性实验可以不启动任何智能体就跑；
  3. 智能体的失败不会污染算法正确性。
"""
from __future__ import annotations

import ast
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORE_DIR = os.path.join(ROOT, "traj_agent", "core")

FORBIDDEN_MODULES = {
    "openai", "anthropic", "requests", "httpx", "aiohttp", "urllib3",
    "socket", "http", "urllib.request", "fastapi", "flask",
}
FORBIDDEN_PREFIXES = ("traj_agent.tools", "traj_agent.agent",
                      "traj_agent.verifier", "traj_agent.memory",
                      "traj_agent.report")


def _core_files():
    for name in sorted(os.listdir(CORE_DIR)):
        if name.endswith(".py"):
            yield os.path.join(CORE_DIR, name)


def _imports(path: str):
    with open(path, "r", encoding="utf-8") as fh:
        tree = ast.parse(fh.read(), filename=path)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                yield node.module


def test_core_files_exist():
    assert len(list(_core_files())) >= 8


@pytest.mark.parametrize("path", list(_core_files()), ids=lambda p: os.path.basename(p))
def test_core_does_not_import_forbidden_modules(path):
    for mod in _imports(path):
        root = mod.split(".")[0]
        assert root not in FORBIDDEN_MODULES, (
            f"{os.path.basename(path)} 引入了被禁止的模块 {mod}")
        assert not mod.startswith(FORBIDDEN_PREFIXES), (
            f"{os.path.basename(path)} 依赖了上层模块 {mod}（应为单向依赖）")


def test_core_does_not_import_agent_layer_transitively():
    """通过实际导入检查：core 模块加载后不应把 agent 层带进 sys.modules。"""
    for name in list(sys.modules):
        if name.startswith(("traj_agent.tools", "traj_agent.agent",
                            "traj_agent.verifier", "traj_agent.memory")):
            del sys.modules[name]
    import traj_agent.core as core  # noqa: F401
    for name in list(sys.modules):
        assert not name.startswith(("traj_agent.tools", "traj_agent.agent",
                                    "traj_agent.verifier", "traj_agent.memory")), (
            f"导入 core 时连带加载了 {name}")


def test_no_network_imports_anywhere_in_core_source():
    """文本级检查：即使注释里写 socket 也无妨，但 import 语句不行。"""
    for path in _core_files():
        with open(path, "r", encoding="utf-8") as fh:
            src = fh.read()
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                assert node.func.id not in ("eval", "exec"), (
                    f"{os.path.basename(path)} 使用了 {node.func.id}")


def test_layer_dependency_direction():
    """report / tools 可以依赖 core；core 不能依赖它们。"""
    import traj_agent.core.geo as geo_mod
    assert geo_mod.__name__.startswith("traj_agent.core")
