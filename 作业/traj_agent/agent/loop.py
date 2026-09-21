"""ReAct 主循环：诊断 → 检索 → 提议 → 执行 → 核验 → 记忆。

流程（对应设计文档第 5 节）
--------------------------
  [1] 载入轨迹（坐标留在内存，只取诊断卡）
  [2] 检索记忆（L2 参数区间 + L3 人类知识）
  [3] LLM 提议（结构化：params / expected_effect / rationale）
  [4] 核验器检查约束（纯代码，无 LLM）
  [5] 执行提议 → 实测指标
  [6] 确定性搜索 → ground truth 最优
  [7] regret = score(search_best) - score(proposal)
  [8] 仅当 regret <= 阈值 才写 SQLite 记忆

第 3 步与第 7-8 步之间是整个工作流的价值所在：它把「LLM 说得好不好」
变成一个可自动判分的数字，而不是让人去读一段解释。

提议阶段与核验阶段是两个独立步骤（且理想情况下用不同模型），
这是有意的 generator–verifier 分离。
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..core import clean, diagnosis, geo, metrics, params as params_mod, simplify
from ..memory import retrieve as retrieve_mod
from ..memory.store import MemoryStore
from ..tools.playbook import playbook_digest
from ..tools.registry import ToolContext, ToolRegistry, attach_dataset, attach_memory
from ..verifier import objective as obj_mod, search as search_mod, verify as verify_mod
from . import prompts, provider as provider_mod

# 单条轨迹默认预算。这两个数是成本护栏，不是随手填的：
# 工具调用 25 次足以完成「诊断→去噪→切分→扫描→求 knee→评估」且留有余量；
# LLM 轮次 8 轮覆盖同样的动作序列。
DEFAULT_MAX_TOOL_CALLS = 25
# 轮次上限 4：实测 6 轮里后 3 轮都是无意义的重复探测。
# 每多一轮就多一次完整提示的输入开销。
DEFAULT_MAX_LLM_TURNS = 4

# 发给 LLM 的工具子集。工具 schema 每轮都要重发，是最大的 token 开销来源：
# 全部 15 个约 5700 字符，这 7 个约 2500 字符，省一半以上。
# 被裁掉的都是「执行类」工具（split / clean / simplify / render /
# apply_road_constraint / load_trajectory / evaluate / compare_handles）——
# 它们由框架在核验阶段确定性调用，不需要 LLM 决策。
# 留下的都是「决策所需」的：看诊断、看异常、扫参数、取拐点、查经验。
# 默认只发 3 个。实测教训：发 7 个时模型会在 6 轮里反复调 detect_anomalies，
# 每轮把完整工具 schema 重发一次，累计 25116 prompt tokens 却拿不到有效提议。
# 关键不是「工具够不够」，而是「模型肯不肯一次给答案」——
# 所以工具集压到最小，并在提示里强制先输出 JSON。
ESSENTIAL_TOOLS = (
    "suggest_param_range",
    "run_search",
    "detect_anomalies",
)


@dataclass
class StepRecord:
    """一步的记录，构成可读的 trace。"""

    index: int
    kind: str                      # llm | tool | verify | memory
    name: str = ""
    detail: Dict[str, Any] = field(default_factory=dict)
    elapsed_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {"index": self.index, "kind": self.kind, "name": self.name,
                "detail": self.detail, "elapsed_ms": round(self.elapsed_ms, 2)}


@dataclass
class AgentResult:
    """一条轨迹跑完的完整结果。"""

    vehicle_id: str
    seg_id: str
    ok: bool = False
    error: str = ""
    mode: str = "llm+memory+search"

    # 诊断
    diagnosis: Dict[str, Any] = field(default_factory=dict)

    # 提议
    proposal_params: Dict[str, float] = field(default_factory=dict)
    proposal_raw: Dict[str, Any] = field(default_factory=dict)
    proposal_source: str = ""      # llm | fallback

    # 执行与指标
    proposal_objective: Dict[str, Any] = field(default_factory=dict)
    baseline_objective: Dict[str, Any] = field(default_factory=dict)
    best_objective: Dict[str, Any] = field(default_factory=dict)
    measured: Dict[str, Any] = field(default_factory=dict)

    # 核验
    verification: Dict[str, Any] = field(default_factory=dict)
    search: Dict[str, Any] = field(default_factory=dict)

    # 记忆
    memory_prior: Dict[str, Any] = field(default_factory=dict)
    memory_written: bool = False
    episodic_id: Optional[int] = None

    # 账本
    trace: List[StepRecord] = field(default_factory=list)
    # 总工具调用次数（含 agent 自己发起的 load_trajectory 等）
    tool_calls: int = 0
    # LLM 通过 function-calling 发起的调用次数，**这才是消耗预算的那个数**
    llm_tool_calls: int = 0
    llm_turns: int = 0
    total_elapsed_ms: float = 0.0
    token_usage: Dict[str, int] = field(default_factory=dict)
    provider: str = ""

    def to_dict(self, with_trace: bool = True) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "vehicle_id": self.vehicle_id,
            "seg_id": self.seg_id,
            "ok": self.ok,
            "error": self.error,
            "mode": self.mode,
            "provider": self.provider,
            "regime": self.diagnosis.get("regime"),
            "timeline_quality": (self.diagnosis.get("timeline") or {}).get("quality"),
            "proposal_source": self.proposal_source,
            "proposal_params": {k: round(float(v), 4)
                                for k, v in sorted(self.proposal_params.items())},
            "proposal_objective": self.proposal_objective,
            "baseline_objective": self.baseline_objective,
            "best_objective": self.best_objective,
            "measured": self.measured,
            "verification": self.verification,
            "search": self.search,
            "memory_prior_available": bool(self.memory_prior.get("available")),
            "memory_written": self.memory_written,
            "episodic_id": self.episodic_id,
            "tool_calls": self.tool_calls,
            "llm_tool_calls": self.llm_tool_calls,
            "llm_turns": self.llm_turns,
            "token_usage": self.token_usage,
            "total_elapsed_ms": round(self.total_elapsed_ms, 2),
        }
        if with_trace:
            d["trace"] = [s.to_dict() for s in self.trace]
        return d


class TrajCleaningAgent:
    """轨迹清洗参数决策智能体。"""

    def __init__(self,
                 registry: Optional[ToolRegistry] = None,
                 llm: Optional[Any] = None,
                 memory: Optional[MemoryStore] = None,
                 vault_dir: Optional[str] = None,
                 max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS,
                 max_llm_turns: int = DEFAULT_MAX_LLM_TURNS,
                 regret_threshold: float = 0.05,
                 use_memory: bool = True,
                 use_search: bool = True,
                 use_llm: bool = True,
                 read_only: bool = False,
                 tool_subset: Optional[Sequence[str]] = None,
                 direct_first: bool = True,
                 mode: str = "llm+memory+search") -> None:
        self.registry = registry or ToolRegistry()
        if vault_dir is not None:
            self.registry.ctx.vault_dir = vault_dir
        self.llm = llm or provider_mod.build_provider()
        self.memory = memory
        if memory is not None:
            attach_memory(self.registry.ctx, memory)
        self.max_tool_calls = int(max_tool_calls)
        self.max_llm_turns = int(max_llm_turns)
        self.regret_threshold = float(regret_threshold)
        self.use_memory = bool(use_memory)
        self.use_search = bool(use_search)
        self.use_llm = bool(use_llm)
        # read_only=True 时记忆只读不写。
        # 评测留出集时必须开启：否则会边评测边把结果写回记忆，
        # 下一条 holdout 轨迹就能检索到上一条的结果，产生跨样本泄漏。
        self.read_only = bool(read_only)
        # 发给 LLM 的工具子集；默认只发决策必需的那几个以省 token。
        # 传 () 表示不限制（发全部 15 个）。
        self.tool_subset = (list(tool_subset) if tool_subset
                            else list(ESSENTIAL_TOOLS))
        # direct_first：第一轮**不发工具**，直接要 JSON 提议。
        # 实测依据：带工具时模型会在 4 轮里反复调同一个工具，
        # 累计 14848 prompt tokens 却拿不到有效提议，最后回退默认值。
        # 而决策所需的诊断信息在系统提示里已经全给了，工具并非必需。
        # 只有直答失败时才升级到带工具的路径。
        self.direct_first = bool(direct_first)
        self.mode = mode

    # ---- 工具 -----------------------------------------------------------
    def attach_dataset(self, raw: Dict[str, Any]) -> "TrajCleaningAgent":
        attach_dataset(self.registry.ctx, raw)
        return self

    # ---- 主入口 ---------------------------------------------------------
    def run(self, vehicle_id: str, segment_index: int = 0,
            run_id: str = "") -> AgentResult:
        t_start = time.perf_counter()
        run_id = run_id or uuid.uuid4().hex[:12]
        res = AgentResult(vehicle_id=str(vehicle_id),
                          seg_id=f"{vehicle_id}#{segment_index}",
                          mode=self.mode,
                          provider=getattr(self.llm, "name", "?"))
        ctx = self.registry.ctx
        ctx.call_log.clear()

        # [1] 载入
        load = self.registry.call("load_trajectory", vehicle_id=str(vehicle_id),
                                 segment_index=int(segment_index))
        if not load.get("ok"):
            res.error = str(load.get("error", "载入失败"))
            res.total_elapsed_ms = (time.perf_counter() - t_start) * 1000.0
            return res
        handle = load["handle"]
        card = load["diagnosis"]
        res.diagnosis = card
        self._step(res, "tool", "load_trajectory",
                   {"handle": handle, "n_points": load.get("n_points")})

        # [2] 检索记忆与人类知识
        prior_dict: Dict[str, Any] = {"available": False}
        if self.use_memory and self.memory is not None:
            card_obj = diagnosis.diagnose(self.registry.ctx.store.get(handle))
            prior = retrieve_mod.prior_for(card_obj, self.memory)
            prior_dict = prior.to_dict()
        res.memory_prior = prior_dict
        self._step(res, "memory", "retrieve",
                   {"available": prior_dict.get("available"),
                    "n_neighbors": prior_dict.get("n_neighbors", 0),
                    "n_regions": len(prior_dict.get("suggested_regions", []))})

        # [3] 提议。use_llm=False 时跳过一切 LLM 调用，直接用数据驱动先验，
        # 这是消融实验里「search-only」基线成立的前提：
        # 它必须真的不含 LLM 成分，否则与含 LLM 的模式不可比。
        if self.use_llm:
            proposal, source, n_tools = self._elicit_proposal(
                res, handle, card, prior_dict)
        else:
            proposal = self._prior_only_proposal(handle, prior_dict)
            source, n_tools = "prior-only", 0
            self._step(res, "llm", "skipped", {"reason": "use_llm=False"})
        res.proposal_raw = proposal or {}
        res.proposal_source = source
        if not proposal or not isinstance(proposal.get("params"), dict):
            res.error = "未取得有效提议"
            res.llm_tool_calls = int(n_tools)
            res.tool_calls = sum(1 for s_ in res.trace if s_.kind == "tool")
            res.total_elapsed_ms = (time.perf_counter() - t_start) * 1000.0
            return res
        params, clamped = params_mod.clamp_params(
            {k: float(v) for k, v in proposal["params"].items()
             if isinstance(v, (int, float))})
        res.proposal_params = params
        if clamped:
            self._step(res, "verify", "clamp", {"clamped": clamped})

        # [4]-[7] 执行、搜索、核验
        ver = self._evaluate_and_verify(res, handle, params, proposal)
        res.verification = ver.to_dict()

        # [8] 记忆准入（read_only 模式下跳过写入）
        if self.read_only:
            self._step(res, "memory", "write_skipped",
                       {"reason": "read_only=True（留出集评测不得写回记忆）"})
        elif self.use_memory and self.memory is not None:
            try:
                card_obj = diagnosis.diagnose(ctx.store.get(handle))
                eid = self.memory.write_case(
                    card=card_obj,
                    params=params,
                    metrics=res.measured,
                    objective=res.proposal_objective,
                    verification=res.verification,
                    admitted=bool(ver.admitted),
                    admit_reason=ver.admit_reason,
                    mode=self.mode, run_id=run_id,
                    score=float(res.proposal_objective.get("score", 0.0)),
                    regret=float(ver.regret),
                )
                res.episodic_id = eid
                res.memory_written = True
                self._step(res, "memory", "write_case",
                           {"episodic_id": eid, "admitted": ver.admitted,
                            "reason": ver.admit_reason})
            except Exception as exc:      # 记忆失败不应让整条结果作废
                self._step(res, "memory", "write_case_failed",
                           {"error": f"{type(exc).__name__}: {exc}"})

        res.ok = True
        # 两个口径分开记录：
        #   llm_tool_calls 与预算同源，用于成本核算；
        #   tool_calls 是总调用数，用于展示完整动作序列。
        res.llm_tool_calls = int(n_tools)
        res.tool_calls = sum(1 for s_ in res.trace if s_.kind == "tool")
        res.total_elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        return res

    # ---- 提议 -----------------------------------------------------------
    def _elicit_proposal(self, res: AgentResult, handle: str,
                         card: Dict[str, Any],
                         prior: Dict[str, Any]
                         ) -> Tuple[Optional[Dict[str, Any]], str, int]:
        """返回 (提议, 来源, LLM 发起的工具调用次数)。

        工具次数必须显式返回：它是预算护栏的计量口径，
        与 res.tool_calls 需要同源，不能在两个作用域里各自计数。
        """
        ctx = self.registry.ctx
        card_obj = diagnosis.diagnose(ctx.store.get(handle))
        specs = params_mod.param_bounds_table()
        playbook = playbook_digest(self.registry.ctx.vault_dir)

        text_protocol = not hasattr(self.llm, "chat") or isinstance(
            self.llm, provider_mod.MockProvider) is False and False
        system = prompts.build_system_message(
            card, specs, playbook, prior,
            catalog=(self.registry.catalog(self.tool_subset)
                     if text_protocol else None),
            text_protocol=text_protocol)

        messages: List[Dict[str, Any]] = [{"role": "system", "content": system}]
        messages.append({"role": "user", "content": (
            "请给出你的参数提议。可以先用工具查实测数据（最多 3 次），"
            "然后**必须**只输出一个 JSON 对象，不要任何解释文字：\n"
            '{"params": {...}, "expected_effect": {...}, "rationale": "..."}')})

        # MockProvider 需要预注入句柄与诊断卡
        if isinstance(self.llm, provider_mod.MockProvider):
            self.llm.seed(handle, card)

        # ---- 第一级：无工具直答（最省 token）----
        if self.direct_first:
            t0 = time.perf_counter()
            try:
                reply = self.llm.chat(messages, tools=None)
                res.token_usage["prompt"] = res.token_usage.get("prompt", 0) + reply.prompt_tokens
                res.token_usage["completion"] = (res.token_usage.get("completion", 0)
                                                 + reply.completion_tokens)
                res.llm_turns = 1
                self._step(res, "llm", "direct",
                           {"content": (reply.content or "")[:600],
                            "n_tool_calls": 0},
                           (time.perf_counter() - t0) * 1000.0)
                proposal = prompts.proposal_from_text(reply.content)
                if proposal:
                    return proposal, "llm-direct", 0
                self._step(res, "llm", "direct_no_json",
                           {"reason": "直答未含有效 JSON，升级到带工具路径"})
            except provider_mod.ProviderError as exc:
                self._step(res, "llm", "direct_error", {"error": str(exc)})

        # ---- 第二级 / 第三级：带工具（auto），工具在首次调用后撤掉 ----
        tools = None if text_protocol else self.registry.openai_tools(self.tool_subset)
        n_tools = 0
        used_tool_rounds = 0
        for turn in range(max(1, self.max_llm_turns)):
            res.llm_turns = turn + 1 + (1 if self.direct_first else 0)
            t0 = time.perf_counter()
            try:
                reply = self.llm.chat(messages, tools=tools)
            except provider_mod.ProviderError as exc:
                self._step(res, "llm", "error", {"error": str(exc)},
                           (time.perf_counter() - t0) * 1000.0)
                break
            res.token_usage["prompt"] = res.token_usage.get("prompt", 0) + reply.prompt_tokens
            res.token_usage["completion"] = (res.token_usage.get("completion", 0)
                                             + reply.completion_tokens)
            self._step(res, "llm", f"turn{turn + 1}",
                       {"content": (reply.content or "")[:600],
                        "n_tool_calls": len(reply.tool_calls)},
                       (time.perf_counter() - t0) * 1000.0)

            if reply.has_tool_calls:
                used_tool_rounds += 1
                # 给过一次工具机会后就把工具撤掉，逼模型给最终答案。
                # 否则它会一直调下去（实测车辆 0 连续调了 8 次同一个工具）。
                if used_tool_rounds >= 2:
                    tools = None
                messages.append({"role": "assistant", "content": reply.content or "",
                                 "tool_calls": [
                                     {"id": tc.id or f"c{n_tools + i}",
                                      "type": "function",
                                      "function": {"name": tc.name,
                                                   "arguments": json.dumps(
                                                       tc.arguments, ensure_ascii=False)}}
                                     for i, tc in enumerate(reply.tool_calls)]})
                for tc in reply.tool_calls:
                    if n_tools >= self.max_tool_calls:
                        messages.append({"role": "tool",
                                         "tool_call_id": tc.id or "c",
                                         "content": json.dumps(
                                             {"ok": False, "error": "工具调用预算已用尽"})})
                        continue
                    n_tools += 1
                    t1 = time.perf_counter()
                    payload = self.registry.call(tc.name, **tc.arguments)
                    if hasattr(self.llm, "observe"):
                        self.llm.observe(tc.name, payload)
                    self._step(res, "tool", tc.name,
                               {"args": tc.arguments, "ok": payload.get("ok"),
                                "error": payload.get("error", "")},
                               (time.perf_counter() - t1) * 1000.0)
                    messages.append({"role": "tool",
                                     "tool_call_id": tc.id or f"c{n_tools}",
                                     "content": json.dumps(payload, ensure_ascii=False,
                                                           default=str)[:4000]})
                continue

            proposal = prompts.proposal_from_text(reply.content)
            if proposal:
                return proposal, "llm", n_tools
            messages.append({"role": "assistant", "content": reply.content or ""})
            messages.append({"role": "user", "content": (
                "只输出那个 JSON 对象，不要任何解释文字。")})
            # 撤掉工具，逼出最终答案
            tools = None

        # 兜底：用物理先验默认值，并明确标注来源
        self._step(res, "llm", "fallback", {"reason": "未取得结构化提议"})
        return ({"params": params_mod.default_params(),
                 "expected_effect": {}, "rationale": "兜底：使用物理先验默认值"},
                "fallback", n_tools)

    def _prior_only_proposal(self, handle: str,
                             prior: Dict[str, Any]) -> Dict[str, Any]:
        """不调用 LLM，直接用物理先验 + 记忆区间给出参数。

        这是消融实验的 no-LLM 基线。参数来源优先级：
          1) 记忆库里已验证的参数区间中位数（若有）
          2) 数据驱动建议起点
          3) 物理先验默认值
        """
        p = params_mod.default_params()
        for region in prior.get("suggested_regions", []) or []:
            name = region.get("param")
            if name in p and isinstance(region.get("median"), (int, float)):
                p[name] = float(region["median"])
        # 数据驱动起点只看需要按轨迹自适应的参数，避免覆盖记忆给出的区间
        from ..core import diagnosis as diag_mod
        traj = self.registry.ctx.store.get(handle)
        dt = [d for d in traj.time_deltas()[1:] if d > 0]
        suggested = params_mod.data_driven_priors(
            dt, traj.space_deltas_m()[1:],
            [v for v in traj.speeds_mps() if v > 0]).get("suggested_start", {})
        for k, v in suggested.items():
            if k in p and not any(r.get("param") == k
                                  for r in prior.get("suggested_regions", []) or []):
                p[k] = float(v)
        p, _ = params_mod.clamp_params(p)
        return {"params": p, "expected_effect": {},
                "rationale": "prior-only 基线：物理先验 + 记忆区间，未调用 LLM"}

    # ---- 执行与核验 ------------------------------------------------------
    def _execute(self, handle: str,
                 params: Dict[str, float]) -> Tuple[Dict[str, Any], Any]:
        """按给定参数执行一次完整的「清洗 + 简化」，返回实测指标与简化结果。"""
        traj = self.registry.ctx.store.get(handle)
        cfg = clean.CleanConfig.from_params(params)
        cleaned, clean_report = clean.denoise_trajectory(traj, cfg)
        tol = float(params.get("dp_tolerance", 5.0))
        simp = simplify.simplify_trajectory(cleaned, tol)
        # 口径：以清洗后轨迹为参考，隔离出纯 DP 误差
        m = metrics.compute_metrics(simp.traj, reference=cleaned)
        measured = {
            "n_points_before": len(cleaned),
            "n_points_after": len(simp.traj),
            "compression_ratio": round(simp.compression_ratio, 4),
            "max_deviation_m": round(simp.max_deviation_m, 4),
            "length_before_m": round(cleaned.length_m, 3),
            "length_after_m": round(simp.traj.length_m, 3),
            "hausdorff_m": None if m.hausdorff_m is None else round(m.hausdorff_m, 3),
            "frechet_m": None if m.frechet_m is None else round(m.frechet_m, 3),
            "runtime_ms": round(simp.elapsed_ms, 4),
            "clean_report": clean_report.to_dict(),
        }
        return measured, simp

    def _objective_of(self, measured: Dict[str, Any],
                      params: Dict[str, float]) -> obj_mod.ObjectiveResult:
        return obj_mod.compute_objective(
            n_after=measured["n_points_after"],
            n_before=measured["n_points_before"],
            max_deviation_m=measured["max_deviation_m"],
            tolerance_m=float(params.get("dp_tolerance", 5.0)),
            length_after_m=measured["length_after_m"],
            length_before_m=measured["length_before_m"],
            runtime_ms=measured["runtime_ms"],
            road_match_rate=measured.get("road_match_rate"),
        )

    def _evaluate_and_verify(self, res: AgentResult, handle: str,
                             params: Dict[str, float],
                             proposal: Dict[str, Any]) -> verify_mod.VerificationResult:
        # [5] 执行提议
        t0 = time.perf_counter()
        measured, _ = self._execute(handle, params)
        obj = self._objective_of(measured, params)
        res.measured = measured
        res.proposal_objective = obj.to_dict()
        self._step(res, "verify", "execute_proposal",
                   {"score": obj.to_dict()["score"],
                    "compression": obj.to_dict()["compression"],
                    "deviation_m": measured["max_deviation_m"]},
                   (time.perf_counter() - t0) * 1000.0)

        # 基线：物理先验默认值
        base_params = params_mod.default_params()
        base_measured, _ = self._execute(handle, base_params)
        base_obj = self._objective_of(base_measured, base_params)
        res.baseline_objective = base_obj.to_dict()

        # [6] 确定性搜索（ground truth）
        if self.use_search:
            t1 = time.perf_counter()
            trace = self._run_search(handle, params)
            res.search = trace.to_dict()
            self._step(res, "verify", "coordinate_descent",
                       {"n_evaluations": trace.n_evaluations,
                        "best_score": round(trace.best_score, 6),
                        "best_params": {k: round(float(v), 4)
                                        for k, v in sorted(trace.best_params.items())}},
                       (time.perf_counter() - t1) * 1000.0)
        else:
            # 无搜索时，本模式可达的最优就是「基线」——
            # 它没有探索参数空间的能力。故 best=baseline，
            # regret 退化为「提议相对基线的绝对差距」。
            #
            # 注意：这个数值**不能**与含搜索模式的归一化 regret 直接比较，
            # 因为分母的含义不同（一个是本模式可达上限，一个是全局最优）。
            # 消融表必须标注这一点，否则会得出「关掉搜索更好」的错误结论。
            trace = search_mod.SearchTrace(method="disabled")
            trace.best_score = base_obj.score
            trace.best_params = dict(base_params)
            trace.baseline_score = base_obj.score
            trace.baseline_params = dict(base_params)
            trace.record(base_params, base_obj.score)
            res.search = trace.to_dict()
            res.search["caveat"] = (
                "search 已关闭：最优 = 基线，regret 为绝对口径，"
                "不可与含搜索模式的归一化 regret 直接比较")

        best_measured, _ = self._execute(handle, trace.best_params or base_params)
        best_obj = self._objective_of(best_measured, trace.best_params or base_params)
        res.best_objective = best_obj.to_dict()

        # [7] 核验
        predicted = proposal.get("expected_effect") or {}
        observed = {
            "quality": obj.fidelity,
            "compression": obj.compression,
        }
        baseline_metrics = {
            "quality": base_obj.fidelity,
            "compression": base_obj.compression,
        }
        for k, v in params.items():
            observed[k] = float(v)
            baseline_metrics[k] = float(base_params.get(k, v))

        return verify_mod.verify_proposal(
            proposal_params=params,
            proposal_objective=obj,
            baseline_objective=base_obj,
            search_trace=trace,
            predicted_effects={str(k): str(v) for k, v in predicted.items()},
            baseline_metrics=baseline_metrics,
            observed_metrics=observed,
            regret_threshold=self.regret_threshold,
        )

    def _run_search(self, handle: str,
                    seed: Dict[str, float]) -> search_mod.SearchTrace:
        """在物理先验区间内做坐标下降，得到 ground truth 最优分。"""
        cache: Dict[str, float] = {}

        def evaluator(p: Dict[str, float]) -> float:
            key = json.dumps({k: round(float(v), 6) for k, v in sorted(p.items())})
            if key in cache:
                return cache[key]
            try:
                measured, _ = self._execute(handle, p)
                s = self._objective_of(measured, p).score
            except Exception:
                s = -1.0
            cache[key] = s
            return s

        names = [n for n in ("dp_tolerance", "dt_threshold", "max_speed_mps")
                 if n in params_mod.PARAM_SPECS]
        default_params = params_mod.default_params()
        # 固定参照系：物理先验默认参数的分数。归一化 regret 的分母用它，
        # 保证「可提升空间」不随提议而变化，regret 才能跨轨迹比较。
        trace = search_mod.coordinate_descent(
            base_params=dict(default_params), search_params=names,
            evaluator=evaluator, n_steps=5, rounds=3, max_evals=60)
        trace.baseline_score = evaluator(default_params)
        trace.baseline_params = dict(default_params)

        # 起点：把提议值也纳入评估，保证「最优」不会比提议还差（公平比较）
        start = params_mod.clamp_params({**default_params, **seed})[0]
        prop_score = evaluator(start)
        if prop_score > trace.best_score:
            trace.best_score = prop_score
            trace.best_params = dict(start)
        return trace

    # ---- 辅助 -----------------------------------------------------------
    def _step(self, res: AgentResult, kind: str, name: str,
              detail: Dict[str, Any], elapsed_ms: float = 0.0) -> None:
        res.trace.append(StepRecord(index=len(res.trace), kind=kind, name=name,
                                    detail=detail, elapsed_ms=elapsed_ms))

    def run_many(self, vehicle_ids: Sequence[str],
                 run_id: str = "") -> List[AgentResult]:
        run_id = run_id or uuid.uuid4().hex[:12]
        if self.memory is not None:
            self.memory.start_run(run_id, mode=self.mode,
                                  notes=f"{len(vehicle_ids)} cases")
        out = [self.run(v, run_id=run_id) for v in vehicle_ids]
        if self.memory is not None:
            self.memory.finish_run(run_id, sum(1 for r in out if r.ok))
        return out
