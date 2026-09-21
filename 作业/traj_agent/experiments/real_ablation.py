"""真实 OpenAI 兼容模型的四模式消融实验。

Notebook 只运行 Mock 演示；本模块顺序调用真实模型。每个 case 完成后
立即追加 JSONL 并 fsync。恢复时按 phase/repetition/mode/vehicle_id
跳过已有 case，避免重复计费。
"""
from __future__ import annotations

import json
import os
import platform
import random
import re
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ..agent import provider as prov
from ..agent.loop import ESSENTIAL_TOOLS, TrajCleaningAgent
from ..core import diagnosis, params as params_mod, traj as tm
from ..memory.store import MemoryStore
from ..tools.registry import ToolRegistry, attach_dataset
from ..verifier import objective as objective_mod
from ..verifier.ablation import MODE_SPECS

STAGE1_DEMO = ["0", "1", "3", "7", "18", "62", "68", "129",
               "101", "149", "170", "187"]
STAGE1_HOLDOUT = ["2", "4", "10", "22", "111", "137", "153", "154",
                  "165", "194", "201", "208"]
MODES = ["llm-only", "search-only", "llm+search", "llm+memory+search"]
STRATA = [(r, t) for r in ("stationary", "mixed", "moving")
          for t in ("ok", "degraded")]
DEFAULT_SEED = 20260921


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(dict(payload), ensure_ascii=False, default=str) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path} 第 {line_no} 行不是有效 JSON") from exc
        if isinstance(row, dict):
            rows.append(row)
    return rows


def case_key(row: Mapping[str, Any]) -> Tuple[str, int, str, str]:
    return (
        str(row.get("phase", "")),
        int(row.get("repetition", 0)),
        str(row.get("mode", "")),
        str(row.get("vehicle_id", "")),
    )


def redact(text: Any) -> str:
    out = str(text or "")
    secret = os.environ.get(prov.DEFAULT_API_KEY_ENV, "")
    if secret:
        out = out.replace(secret, "<redacted>")
    return re.sub(r"sk-[A-Za-z0-9_-]{8,}", "<redacted>", out)


class TrackingProvider:
    """记录请求数、token、延迟和错误，不保存请求头或凭据。"""

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.name = getattr(inner, "name", type(inner).__name__)
        self.calls = 0
        self.successes = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.elapsed_ms = 0.0
        self.errors: List[str] = []

    def chat(self, messages: Sequence[Dict[str, Any]],
             tools: Optional[Sequence[Dict[str, Any]]] = None) -> Any:
        self.calls += 1
        t0 = time.perf_counter()
        try:
            reply = self.inner.chat(messages, tools=tools)
        except Exception as exc:
            self.elapsed_ms += (time.perf_counter() - t0) * 1000.0
            self.errors.append(redact(f"{type(exc).__name__}: {exc}"))
            raise
        self.successes += 1
        self.elapsed_ms += float(getattr(reply, "elapsed_ms", 0.0) or 0.0)
        self.prompt_tokens += int(getattr(reply, "prompt_tokens", 0) or 0)
        self.completion_tokens += int(getattr(reply, "completion_tokens", 0) or 0)
        return reply


def transient_error(message: str) -> bool:
    msg = message.lower()
    permanent = (
        "401", "403", "authentication", "unauthorized", "invalid api key",
        "insufficient", "quota",
    )
    if any(x in msg for x in permanent):
        return False
    transient = (
        "429", "timeout", "timed out", "connection", "reset", "tempor",
        "502", "503", "504", "rate limit",
    )
    return any(x in msg for x in transient)


def result_row(result: Any, totals: Mapping[str, Any], *, phase: str,
               repetition: int, model: str, retry_count: int,
               elapsed_ms: float, error: str = "") -> Dict[str, Any]:
    d = result.to_dict(with_trace=False)
    verification = dict(d.get("verification") or {})
    raw_proposal = dict(result.proposal_raw or {})
    prior = dict(result.memory_prior or {})
    neighbors = list(prior.get("neighbors") or [])
    similarities = [float(n.get("similarity", 0.0)) for n in neighbors]
    uses_llm = bool(MODE_SPECS.get(result.mode, {}).get("use_llm"))
    source = str(d.get("proposal_source") or "")
    json_success = (
        source in ("llm", "llm-direct")
        and isinstance(raw_proposal.get("params"), dict)
    ) if uses_llm else None
    return {
        "record_type": "case",
        "phase": phase,
        "vehicle_id": str(d.get("vehicle_id")),
        "regime": d.get("regime"),
        "timeline_quality": d.get("timeline_quality"),
        "mode": result.mode,
        "repetition": int(repetition),
        "provider": "OpenAICompatProvider",
        "model": model,
        "proposal_source": source,
        "proposal_params": d.get("proposal_params") or {},
        "expected_effect": raw_proposal.get("expected_effect") or {},
        "rationale": str(raw_proposal.get("rationale") or ""),
        "constraint_ok": verification.get("constraint_ok"),
        "clamped_params": verification.get("clamped_params") or [],
        "feasible": verification.get("feasible"),
        "applicable": verification.get("applicable"),
        "proposal_score": verification.get("proposal_score"),
        "baseline_score": verification.get("baseline_score"),
        "best_score": verification.get("best_score"),
        "regret": verification.get("regret"),
        "regret_basis": verification.get("regret_basis"),
        "direction_accuracy": verification.get("direction_accuracy"),
        "admitted": verification.get("admitted"),
        "admit_reason": verification.get("admit_reason"),
        "memory_prior_available": bool(d.get("memory_prior_available")),
        "memory_neighbor_count": len(neighbors),
        "memory_neighbor_similarities": similarities,
        "memory_suggested_regions": prior.get("suggested_regions") or [],
        "llm_turns": int(d.get("llm_turns") or 0),
        "llm_tool_calls": int(d.get("llm_tool_calls") or 0),
        "llm_calls": int(totals.get("calls", 0)),
        "prompt_tokens": int(totals.get("prompt_tokens", 0)),
        "completion_tokens": int(totals.get("completion_tokens", 0)),
        "llm_elapsed_ms": round(float(totals.get("llm_elapsed_ms", 0.0)), 2),
        "elapsed_ms": round(float(elapsed_ms), 2),
        "json_success": json_success,
        "agent_ok": bool(d.get("ok")),
        "error": redact(error or d.get("error") or ""),
        "retry_count": int(retry_count),
        "provider_error_count": len(totals.get("errors", [])),
        "attempt_errors": [redact(x) for x in totals.get("errors", [])],
        "proposal_objective": d.get("proposal_objective") or {},
        "baseline_objective": d.get("baseline_objective") or {},
        "best_objective": d.get("best_objective") or {},
        "verification": verification,
        "measured": d.get("measured") or {},
        "search": d.get("search") or {},
        "memory_written": False,
    }


def run_case(raw: Dict[str, Any], vehicle_id: str, mode: str, repetition: int,
             phase: str, base_provider: Any, memory: Optional[MemoryStore],
             max_retries: int, backoff_s: float) -> Tuple[Dict[str, Any], Any]:
    spec = dict(MODE_SPECS[mode])
    totals: Dict[str, Any] = {
        "calls": 0,
        "successes": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "llm_elapsed_ms": 0.0,
        "errors": [],
    }
    started = time.perf_counter()
    last_result = None
    final_error = ""
    retries_used = 0
    for attempt in range(max_retries + 1):
        tracker = TrackingProvider(base_provider)
        registry = ToolRegistry()
        attach_dataset(registry.ctx, raw)
        agent = TrajCleaningAgent(
            registry=registry,
            llm=tracker,
            memory=memory if spec.get("use_memory") else None,
            mode=mode,
            read_only=True,
            direct_first=True,
            tool_subset=ESSENTIAL_TOOLS,
            **spec,
        )
        last_result = agent.run(
            str(vehicle_id), run_id=f"ecnu-r{repetition}-{phase}"
        )
        totals["calls"] += tracker.calls
        totals["successes"] += tracker.successes
        totals["prompt_tokens"] += tracker.prompt_tokens
        totals["completion_tokens"] += tracker.completion_tokens
        totals["llm_elapsed_ms"] += tracker.elapsed_ms
        totals["errors"].extend(tracker.errors)

        if not spec.get("use_llm") or tracker.successes > 0:
            break
        final_error = (
            tracker.errors[-1] if tracker.errors
            else "LLM 未返回有效响应"
        )
        if attempt >= max_retries or not transient_error(final_error):
            break
        retries_used += 1
        time.sleep(backoff_s * (2 ** attempt))

    assert last_result is not None
    if spec.get("use_llm") and totals["successes"] <= 0:
        final_error = final_error or "所有 LLM 请求均失败"
    row = result_row(
        last_result,
        totals,
        phase=phase,
        repetition=repetition,
        model=str(getattr(base_provider, "model", "")),
        retry_count=retries_used,
        elapsed_ms=(time.perf_counter() - started) * 1000.0,
        error=final_error,
    )
    return row, last_result


def build_demo_candidates(
    raw: Dict[str, Any],
    *,
    seed: int = DEFAULT_SEED,
    sizes: Sequence[int] = (24, 48, 100, 200),
    fixed_demo: Sequence[str] = STAGE1_DEMO,
    holdout: Sequence[str] = STAGE1_HOLDOUT,
) -> Dict[str, Any]:
    """生成嵌套分层 demo 候选集，优先补充 moving 层。"""
    if set(fixed_demo) & set(holdout):
        raise ValueError("demo 与 holdout 不能重叠")
    if any(int(n) < len(fixed_demo) for n in sizes):
        raise ValueError("候选规模不能小于固定 demo 数")
    excluded = set(fixed_demo) | set(holdout)
    buckets: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    for vid in sorted(raw, key=lambda x: (len(str(x)), str(x))):
        vid = str(vid)
        if vid in excluded:
            continue
        card = diagnosis.diagnose(tm.traj_from_raw(vid, *raw[vid]))
        buckets[(card.regime, card.timeline_quality)].append(vid)
    rng = random.Random(seed)
    for key in sorted(buckets):
        rng.shuffle(buckets[key])

    # moving:mixed:stationary = 3:2:1；类内交替 ok/degraded。
    schedule: List[Tuple[str, str]] = []
    for regime, weight in (("moving", 3), ("mixed", 2), ("stationary", 1)):
        for _ in range(weight):
            schedule.extend([(regime, "ok"), (regime, "degraded")])
    offsets: Dict[Tuple[str, str], int] = defaultdict(int)
    ordered_extra: List[str] = []
    needed = max(int(n) for n in sizes) - len(fixed_demo)
    while len(ordered_extra) < needed:
        progressed = False
        for key in schedule:
            pos = offsets[key]
            if pos < len(buckets.get(key, [])):
                ordered_extra.append(buckets[key][pos])
                offsets[key] += 1
                progressed = True
                if len(ordered_extra) >= needed:
                    break
        if not progressed:
            raise ValueError("可用轨迹不足以生成候选集")

    def describe(ids: Sequence[str]) -> Dict[str, Any]:
        counts = {f"{r}/{t}": 0 for r, t in STRATA}
        for vid in ids:
            card = diagnosis.diagnose(tm.traj_from_raw(vid, *raw[vid]))
            counts[f"{card.regime}/{card.timeline_quality}"] += 1
        return {"n": len(ids), "vehicle_ids": list(ids), "strata": counts}

    candidates: Dict[str, Any] = {}
    for n in sorted(set(int(x) for x in sizes)):
        ids = list(fixed_demo) + ordered_extra[:n - len(fixed_demo)]
        if set(ids) & set(holdout):
            raise AssertionError("生成的 demo 与 holdout 重叠")
        candidates[str(n)] = describe(ids)
    return {
        "seed": int(seed),
        "strategy": (
            "保留固定12条demo，其余按(regime,timeline_quality)固定种子打乱；"
            "加权轮转优先moving，候选集逐级嵌套。"
        ),
        "fixed_demo": list(fixed_demo),
        "holdout": list(holdout),
        "candidates": candidates,
    }


def git_sha(repo_dir: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_dir,
            text=True,
            encoding="utf-8",
        ).strip()
    except Exception:
        return "unknown"


def build_config(repo_dir: Path, provider: Any, repetitions: int,
                 seed: int) -> Dict[str, Any]:
    return {
        "git_commit_sha": git_sha(repo_dir),
        "date": date.today().isoformat(),
        "provider": "OpenAICompatProvider",
        "model": str(getattr(provider, "model", "")),
        "base_url": str(getattr(provider, "base_url", "") or "(default)"),
        "api_key_present": bool(os.environ.get(prov.DEFAULT_API_KEY_ENV)),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "demo_vehicle_ids": list(STAGE1_DEMO),
        "holdout_vehicle_ids": list(STAGE1_HOLDOUT),
        "repetitions": int(repetitions),
        "seed": int(seed),
        "modes": {k: dict(MODE_SPECS[k]) for k in MODES},
        "tool_subset": list(ESSENTIAL_TOOLS),
        "direct_first": True,
        "search_budget": {
            "method": "coordinate_descent",
            "start": "default_params",
            "dimensions": [
                "dp_tolerance", "dt_threshold", "max_speed_mps"
            ],
            "n_steps": 5,
            "rounds": 3,
            "max_evals": 60,
            "note": "LLM候选独立执行；搜索从默认参数开始建立内部参考。",
        },
        "regret_threshold": 0.05,
        "objective": {
            "weights": objective_mod.ObjectiveWeights().__dict__,
            "length_ratio_floor": 0.98,
            "min_applicable_length_m": objective_mod.MIN_APPLICABLE_LENGTH_M,
            "fidelity_scale_m": 30.0,
            "warning": "确定性搜索是内部目标参考，不是现实道路真值。",
        },
        "parameter_defaults": params_mod.default_params(),
        "parameter_bounds": params_mod.param_bounds_table(),
    }


def hydrate_memory(store: MemoryStore, raw: Dict[str, Any],
                   rows: Sequence[Mapping[str, Any]], repetition: int) -> None:
    """从已保存 demo 结果重建 L1，resume 不重新调用 LLM。"""
    for row in rows:
        if (
            row.get("phase") != "demo"
            or int(row.get("repetition", 0)) != repetition
            or row.get("error")
            or not row.get("memory_written")
        ):
            continue
        vid = str(row["vehicle_id"])
        card = diagnosis.diagnose(tm.traj_from_raw(vid, *raw[vid]))
        store.write_case(
            card=card,
            params=dict(row.get("proposal_params") or {}),
            metrics=dict(row.get("measured") or {}),
            objective=dict(row.get("proposal_objective") or {}),
            verification=dict(row.get("verification") or {}),
            admitted=bool(row.get("admitted")),
            admit_reason=str(row.get("admit_reason") or ""),
            mode="llm+memory+search",
            run_id=f"ecnu-r{repetition}-demo",
            score=row.get("proposal_score"),
            regret=row.get("regret"),
        )


def memory_snapshot(store: MemoryStore,
                    demo_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    admitted = {f"{r}/{t}": 0 for r, t in STRATA}
    total = {f"{r}/{t}": 0 for r, t in STRATA}
    for row in demo_rows:
        key = f"{row.get('regime')}/{row.get('timeline_quality')}"
        total[key] = total.get(key, 0) + 1
        if row.get("admitted"):
            admitted[key] = admitted.get(key, 0) + 1
    return {
        "stats": store.stats(),
        "demo_total_by_stratum": total,
        "demo_admitted_by_stratum": admitted,
        "regions": [r.to_dict() for r in store.query_regions()],
    }


def run_experiment(
    experiment_dir: Path,
    data_path: Path,
    *,
    repetitions: int = 3,
    seed: int = DEFAULT_SEED,
    max_retries: int = 2,
    backoff_s: float = 5.0,
    retry_failed: bool = False,
) -> None:
    from dotenv import load_dotenv

    experiment_dir = experiment_dir.resolve()
    work_dir = experiment_dir.parents[1]
    repo_dir = work_dir.parent
    load_dotenv(work_dir / ".env", override=False)
    status = prov.provider_status()
    if status.get("would_use") != "OpenAICompatProvider":
        raise RuntimeError(
            "未检测到真实 ECNU provider；实验终止，不允许静默回退 Mock"
        )
    base_provider = prov.build_provider()
    if not isinstance(base_provider, prov.OpenAICompatProvider):
        raise RuntimeError("构造的 provider 不是 OpenAICompatProvider")
    if set(STAGE1_DEMO) & set(STAGE1_HOLDOUT):
        raise RuntimeError("demo 与 holdout 重叠")

    experiment_dir.mkdir(parents=True, exist_ok=True)
    raw = tm.load_raw(str(data_path))
    config = build_config(repo_dir, base_provider, repetitions, seed)
    atomic_json(experiment_dir / "config.json", config)
    manifest = build_demo_candidates(raw, seed=seed)
    manifest["stage1"] = {
        "demo": list(STAGE1_DEMO),
        "holdout": list(STAGE1_HOLDOUT),
        "overlap": sorted(set(STAGE1_DEMO) & set(STAGE1_HOLDOUT)),
    }
    atomic_json(experiment_dir / "sample_manifest.json", manifest)

    results_path = experiment_dir / "raw_results.jsonl"
    rows = read_jsonl(results_path)
    completed = {
        case_key(r) for r in rows
        if not retry_failed or not r.get("error")
    }
    state_path = experiment_dir / "run_state.json"
    state = (
        json.loads(state_path.read_text(encoding="utf-8"))
        if state_path.exists()
        else {"memory_diagnostics": {}}
    )
    runtime_dir = experiment_dir / ".runtime"
    runtime_dir.mkdir(exist_ok=True)

    print(json.dumps({
        "provider": "OpenAICompatProvider",
        "model": config["model"],
        "base_url": config["base_url"],
        "repetitions": repetitions,
        "completed_cases": len(completed),
        "api_key_present": True,
    }, ensure_ascii=False), flush=True)

    for rep in range(1, repetitions + 1):
        # 每轮从 JSONL 重建全新的库，保证轮次隔离且 resume 不依赖 SQLite。
        mem_path = runtime_dir / f"memory_rep{rep}.sqlite"
        if mem_path.exists():
            mem_path.unlink()
        memory = MemoryStore(str(mem_path))
        rows = read_jsonl(results_path)
        hydrate_memory(memory, raw, rows, rep)

        for vid in STAGE1_DEMO:
            key = ("demo", rep, "llm+memory+search", vid)
            if key in completed:
                continue
            row, _ = run_case(
                raw, vid, "llm+memory+search", rep, "demo",
                base_provider, memory, max_retries, backoff_s,
            )
            if not row["error"] and row["agent_ok"]:
                card = diagnosis.diagnose(tm.traj_from_raw(vid, *raw[vid]))
                memory.write_case(
                    card=card,
                    params=row["proposal_params"],
                    metrics=row["measured"],
                    objective=row["proposal_objective"],
                    verification=row["verification"],
                    admitted=bool(row["admitted"]),
                    admit_reason=row["admit_reason"],
                    mode="llm+memory+search",
                    run_id=f"ecnu-r{rep}-demo",
                    score=row["proposal_score"],
                    regret=row["regret"],
                )
                row["memory_written"] = True
            append_jsonl(results_path, row)
            completed.add(key)
            print(
                f"demo r{rep} vehicle={vid} source={row['proposal_source']} "
                f"calls={row['llm_calls']} error={bool(row['error'])}",
                flush=True,
            )

        memory.rebuild_procedural(min_samples=1)
        rows = read_jsonl(results_path)
        demo_rows = [
            r for r in rows
            if r.get("phase") == "demo"
            and int(r.get("repetition", 0)) == rep
        ]
        state.setdefault("memory_diagnostics", {})[str(rep)] = memory_snapshot(
            memory, demo_rows
        )
        state["last_updated"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        atomic_json(state_path, state)

        for mode in MODES:
            for vid in STAGE1_HOLDOUT:
                key = ("holdout", rep, mode, vid)
                if key in completed:
                    continue
                mem_arg = (
                    memory if MODE_SPECS[mode].get("use_memory") else None
                )
                row, _ = run_case(
                    raw, vid, mode, rep, "holdout",
                    base_provider, mem_arg, max_retries, backoff_s,
                )
                append_jsonl(results_path, row)
                completed.add(key)
                print(
                    f"holdout r{rep} {mode} vehicle={vid} "
                    f"calls={row['llm_calls']} error={bool(row['error'])}",
                    flush=True,
                )
        memory.close()

    # SQLite 只是断点运行状态，正式证据已经写入 JSONL 与诊断 JSON。
    shutil.rmtree(runtime_dir, ignore_errors=True)
