"""traj_agent：LLM 辅助的轨迹清洗评估工作流。

分层（单向依赖，见 tests/test_architecture.py）：
    core     纯确定性算法，零 LLM 依赖
    road     路网匹配协议与离线实现
    tools    core 的 JSON-Schema 封装，LLM 唯一能接触的层
    verifier 目标函数、搜索、核验（regret 的计算者）
    memory   SQLite 情景/程序记忆 + 特征 kNN 检索
    agent    ReAct 主循环与 provider
    report   出图

核心设计约束
-----------
1. LLM 不做数值计算，也不当优化器。
2. LLM 不能写记忆；只有 verifier 在实测确认有效后写入。
3. 坐标永不进 LLM 上下文，只以句柄 + 诊断卡流动。
4. 提议与核验分离（generator–verifier gap）。
"""
from __future__ import annotations

__version__ = "1.0.0"

__all__ = ["__version__"]
