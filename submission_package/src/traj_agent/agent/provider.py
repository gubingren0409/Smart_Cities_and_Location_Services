"""LLM provider：OpenAI 兼容接口 + 离线 Mock。

用 OpenAI 兼容协议是因为 DeepSeek / Qwen / 自建 vLLM 都支持它，
换模型只改环境变量，不动代码。

MockProvider 的用途不是「假装成功」，而是让整条流水线**在没有 key 时**
也能完整跑通并被测试覆盖：agent 循环、工具派发、核验、记忆写入
全部走真实代码路径，只有「选哪个工具、提什么参数」这一步换成确定性策略。
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

DEFAULT_BASE_URL_ENV = "DSH_TRAJ_LLM_BASE_URL"
DEFAULT_API_KEY_ENV = "DSH_TRAJ_LLM_API_KEY"
DEFAULT_MODEL_ENV = "DSH_TRAJ_LLM_MODEL"


@dataclass
class ToolCall:
    """一次工具调用请求。"""

    id: str
    name: str
    arguments: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "name": self.name, "arguments": self.arguments}


@dataclass
class LLMResponse:
    """一次模型回复。"""

    content: str = ""
    tool_calls: List[ToolCall] = field(default_factory=list)
    finish_reason: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    elapsed_ms: float = 0.0
    raw: Optional[Dict[str, Any]] = None

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "tool_calls": [t.to_dict() for t in self.tool_calls],
            "finish_reason": self.finish_reason,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "elapsed_ms": round(self.elapsed_ms, 2),
        }


class ProviderError(RuntimeError):
    pass


class OpenAICompatProvider:
    """OpenAI 兼容接口的 provider。

    配置来源（优先级从高到低）：显式参数 → 环境变量。
        DSH_TRAJ_LLM_BASE_URL
        DSH_TRAJ_LLM_API_KEY
        DSH_TRAJ_LLM_MODEL
    """

    def __init__(self, model: Optional[str] = None,
                 base_url: Optional[str] = None,
                 api_key: Optional[str] = None,
                 temperature: float = 0.0,
                 max_tokens: int = 1200,
                 timeout_s: float = 180.0,
                 client: Any = None,
                 extra_body: Optional[Dict[str, Any]] = None,
                 tool_choice: Optional[str] = None) -> None:
        self.model = model or os.environ.get(DEFAULT_MODEL_ENV) or "deepseek-chat"
        self.base_url = base_url or os.environ.get(DEFAULT_BASE_URL_ENV)
        self.api_key = api_key or os.environ.get(DEFAULT_API_KEY_ENV)
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.timeout_s = float(timeout_s)
        self._client = client
        # extra_body 用于传递平台特有参数（如 ECNU 的思考模式开关）。
        # 默认关闭思考模式以省 token：ECNU 手册说两款模型默认关闭，
        # 但仍显式传 disabled 做双保险，避免上游默认值变化时悄悄开始烧 token。
        self.extra_body = dict(extra_body) if extra_body is not None else {
            "thinking": {"type": "disabled"}}
        # tool_choice="required" 强制每轮只调一个工具，
        # 避免模型一轮并发多个调用带来额外的输入 token 与等待时间。
        self.tool_choice = tool_choice
        self.name = f"openai-compat:{self.model}"

    @property
    def available(self) -> bool:
        return bool(self.api_key) or self._client is not None

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            from openai import OpenAI
        except ImportError as exc:      # pragma: no cover
            raise ProviderError(
                "未安装 openai 包；请 `pip install openai` 或使用 MockProvider") from exc
        if not self.api_key:
            raise ProviderError(
                f"未设置 API key。请设置环境变量 {DEFAULT_API_KEY_ENV}，"
                "或在构造时传入 api_key。")
        kwargs: Dict[str, Any] = {"api_key": self.api_key, "timeout": self.timeout_s}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        self._client = OpenAI(**kwargs)
        return self._client

    def chat(self, messages: Sequence[Dict[str, Any]],
             tools: Optional[Sequence[Dict[str, Any]]] = None) -> LLMResponse:
        client = self._get_client()
        t0 = time.perf_counter()
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if tools:
            payload["tools"] = list(tools)
            payload["tool_choice"] = self.tool_choice or "auto"
        if self.extra_body:
            payload["extra_body"] = self.extra_body

        try:
            resp = client.chat.completions.create(**payload)
        except Exception as exc:
            raise ProviderError(f"调用 LLM 失败: {type(exc).__name__}: {exc}") from exc
        elapsed = (time.perf_counter() - t0) * 1000.0

        choice = resp.choices[0] if getattr(resp, "choices", None) else None
        if choice is None:
            raise ProviderError("LLM 返回不含 choices")
        msg = choice.message
        content = getattr(msg, "content", "") or ""
        calls: List[ToolCall] = []
        for tc in (getattr(msg, "tool_calls", None) or []):
            fn = getattr(tc, "function", None)
            if fn is None:
                continue
            args: Dict[str, Any] = {}
            raw_args = getattr(fn, "arguments", "") or ""
            if isinstance(raw_args, dict):
                args = raw_args
            elif raw_args.strip():
                try:
                    args = json.loads(raw_args)
                except json.JSONDecodeError:
                    args = {"_raw": raw_args}
            calls.append(ToolCall(id=str(getattr(tc, "id", "") or ""),
                                  name=str(getattr(fn, "name", "") or ""),
                                  arguments=args))
        usage = getattr(resp, "usage", None)
        return LLMResponse(
            content=content,
            tool_calls=calls,
            finish_reason=str(getattr(choice, "finish_reason", "") or ""),
            prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
            elapsed_ms=elapsed,
            raw=None,
        )

    def describe(self) -> Dict[str, Any]:
        return {
            "provider": "openai-compat",
            "model": self.model,
            "base_url": self.base_url or "(默认)",
            "available": self.available,
            "api_key_set": bool(self.api_key),
            "extra_body": self.extra_body,
            "tool_choice": self.tool_choice or "auto",
        }


class MockProvider:
    """离线 provider：确定性策略，让整条流水线无 key 可跑。

    它不是一个「假 LLM」，而是把「按诊断卡选工具、按物理先验提参数」
    这套规则显式写出来。用途：
      1. 单测与端到端验证覆盖真实代码路径；
      2. 作为消融实验里的 `no-llm` 基线；
      3. 无网络环境下的课堂演示。

    策略读起来很直白：
      - 时间轴不可用 -> 跳过一切速度类工具
      - 静止轨迹   -> 先折叠重复点，不做漂移判定
      - 重复点多   -> 先 clean 再 simplify
      - DP 容差    -> 取 GPS CEP 量级
      - 参数方向预测 -> 按物理单调性给出
    """

    def __init__(self, verbose: bool = False) -> None:
        self.verbose = verbose
        self.name = "mock"
        self._turn = 0
        self._state: Dict[str, Any] = {}
        self.transcript: List[Dict[str, Any]] = []

    @property
    def available(self) -> bool:
        return True

    def chat(self, messages: Sequence[Dict[str, Any]],
             tools: Optional[Sequence[Dict[str, Any]]] = None) -> LLMResponse:
        """按**已完成的工具调用次数**推进状态机。

        刻意不依赖对话历史的消息格式：真实 provider 走的是原生 tool_calls
        协议（assistant 消息带 tool_calls 字段、工具结果带 tool_call_id），
        与文本协议不同。若 Mock 去解析历史格式，它会在真实循环里错位——
        实测表现为第 2 轮就跳过全部工具、只调用 1 次工具。
        用计数器则与格式解耦，两种协议下行为一致。
        """
        self._turn += 1
        t0 = time.perf_counter()
        last = messages[-1] if messages else {}
        role = last.get("role", "")
        self.transcript.append({"turn": self._turn, "role": role,
                                "content": str(last.get("content", ""))[:2000]})

        step = int(self._state.get("step", 0))
        resp = self._next_actions(step)
        if resp is None:
            resp = self._propose()
        elif resp.has_tool_calls:
            self._state["step"] = step + 1
        resp.elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return resp

    def observe(self, name: str, payload: Dict[str, Any]) -> None:
        """记录一次工具结果（真实循环在派发后调用，Mock 用来自我推进）。"""
        self._state.setdefault("observations", {})[name] = payload

    def _next_actions(self, step: int) -> Optional[LLMResponse]:
        """第 step 步该做什么工具调用。None 表示进入提议阶段。"""
        st = self._state
        card = st.get("card") or {}
        if not st.get("handle"):
            return None
        timeline = (card.get("timeline") or {}).get("quality", "ok")
        h = st["handle"]
        obs = st.get("observations", {})
        cleaned = st.get("cleaned") or h

        if step == 0:
            return self._call("clean_trajectory", {
                "handle": h,
                "max_speed_mps": 25.0 if timeline == "unusable" else 38.0,
            })
        if step == 1:
            # 时间轴不可用的轨迹跳过切分（dt 无意义）
            if timeline == "unusable":
                return self._call("simplify_trajectory",
                                  {"handle": cleaned, "tolerance_m": 8.0})
            return self._call("split_trajectory", {
                "handle": cleaned, "dt_threshold": 30.0, "dist_threshold": 400.0})
        if step == 2:
            return self._call("run_search", {
                "handle": cleaned, "param": "dp_tolerance", "n": 7,
                "reference_handle": cleaned})
        if step == 3:
            curve = _extract_curve({"content": _obs_text(obs.get("run_search"))})
            st["curve"] = curve
            return self._call("find_knee", {"curve_json": json.dumps(curve)})
        if step == 4:
            knee = _extract_knee({"content": _obs_text(obs.get("find_knee"))})
            st["knee"] = knee
            return None        # 进入提议
        return None

    # ---- 内部 -----------------------------------------------------------
    def seed(self, handle: str, card: Dict[str, Any]) -> None:
        """注入本条轨迹的句柄与诊断卡（真实 provider 不需要）。"""
        self._state = {"handle": handle, "card": card, "step": 0,
                       "cleaned": handle, "observations": {},
                       "curve": None, "knee": None}

    def _call(self, name: str, args: Dict[str, Any], call_id: str = "") -> LLMResponse:
        return LLMResponse(
            content=f"（Mock）调用 {name}",
            tool_calls=[ToolCall(id=call_id or f"mock-{self._turn}", name=name,
                                 arguments=args)],
            finish_reason="tool_calls")

    def _propose(self) -> LLMResponse:
        """给出结构化提议。"""
        st = self._state
        card = st.get("card") or {}
        timeline = (card.get("timeline") or {}).get("quality", "ok")
        stationary = bool(card.get("is_stationary"))
        knee = st.get("knee") or {}
        # 容差：取 knee 值；无 knee 时用 GPS 精度量级
        tol = float(knee.get("value") or 8.0)
        tol = max(0.5, min(30.0, tol))
        if timeline == "unusable":
            tol = 8.0

        params = {
            "dp_tolerance": round(tol, 2),
            "dt_threshold": 30.0 if timeline != "unusable" else 180.0,
            "max_speed_mps": 38.0,
            "dist_threshold": 400.0,
        }
        predicted = {
            "quality": "down" if tol > 5.0 else "same",
            "compression": "up",
            "dp_tolerance": "up" if tol > 5.0 else "same",
        }
        if timeline == "unusable":
            rationale = (
                "时间轴不可用（时间跨度不足以支撑速度量），"
                "跳过基于 dt 的速度类规则，只用几何规则；"
                f"DP 容差取 GPS 精度量级 {tol} m。"
            )
        elif stationary:
            rationale = (
                "静止轨迹：连续重复点占比高属正常，先折叠重复点；"
                "不做漂移与转向角判定（厘米级抖动上无物理意义）。"
                f"DP 容差取 {tol} m。"
            )
        else:
            rationale = (
                f"依据实测容差扫描的 knee point 取 dp_tolerance={tol} m；"
                "该点之后压缩收益递减而偏差开始超过 GPS 精度。"
                "时间阈值沿用 30s（若发现 Δt 双峰应改用 p95）。"
            )
        payload = {
            "params": params,
            "expected_effect": predicted,
            "rationale": rationale,
        }
        return LLMResponse(content=json.dumps(payload, ensure_ascii=False),
                           finish_reason="stop")

    def describe(self) -> Dict[str, Any]:
        return {"provider": "mock", "model": "rule-based", "available": True}


# ---------------------------------------------------------------------------
def _obs_text(payload: Any) -> Any:
    """把工具结果转成 _extract_* 能吃的形式。"""
    if payload is None:
        return ""
    return payload


def _last_handle(msg: Dict[str, Any]) -> Optional[str]:
    content = msg.get("content")
    if isinstance(content, dict):
        return content.get("handle")
    if isinstance(content, str):
        try:
            d = json.loads(content)
            return d.get("handle")
        except json.JSONDecodeError:
            return None
    return None


def _extract_curve(msg: Dict[str, Any]) -> List[Dict[str, Any]]:
    content = msg.get("content")
    d = content if isinstance(content, dict) else None
    if d is None and isinstance(content, str):
        try:
            d = json.loads(content)
        except json.JSONDecodeError:
            return []
    if not isinstance(d, dict):
        return []
    curve = d.get("curve")
    return curve if isinstance(curve, list) else []


def _extract_knee(msg: Dict[str, Any]) -> Dict[str, Any]:
    content = msg.get("content")
    d = content if isinstance(content, dict) else None
    if d is None and isinstance(content, str):
        try:
            d = json.loads(content)
        except json.JSONDecodeError:
            return {}
    if not isinstance(d, dict):
        return {}
    knee = d.get("knee")
    return knee if isinstance(knee, dict) else {}


def build_provider(name: Optional[str] = None, **kwargs) -> Any:
    """按名字或环境变量构造 provider。

    name 未给时：有 key 就用 OpenAICompatProvider，否则回退 MockProvider。
    这个回退是有意的——它让「没配 key」不会变成「跑不起来」。
    """
    key = name or os.environ.get("DSH_TRAJ_LLM_PROVIDER", "")
    if key in ("mock", "none", "offline"):
        return MockProvider()
    if key in ("openai", "openai-compat", "compat"):
        return OpenAICompatProvider(**kwargs)
    # 自动判定
    if os.environ.get(DEFAULT_API_KEY_ENV):
        return OpenAICompatProvider(**kwargs)
    return MockProvider()


def provider_status() -> Dict[str, Any]:
    """当前环境的 provider 配置状态（不泄露 key）。"""
    key = os.environ.get(DEFAULT_API_KEY_ENV)
    return {
        "has_api_key": bool(key),
        "api_key_env": DEFAULT_API_KEY_ENV,
        "base_url": os.environ.get(DEFAULT_BASE_URL_ENV) or "(未设置)",
        "model": os.environ.get(DEFAULT_MODEL_ENV) or "(未设置)",
        "would_use": "OpenAICompatProvider" if key else "MockProvider",
    }
