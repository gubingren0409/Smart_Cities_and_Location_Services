"""工具注册表：core 的 JSON-Schema 封装，LLM 唯一能接触的层。

设计约束
-------
1. **坐标永不返回**。所有响应只含标量统计、计数与句柄字符串。
2. **每个工具都是确定性函数**。没有随机性，没有网络，没有 LLM 调用。
3. **副作用显式标注**。`side_effect` 字段区分只读工具与会创建新句柄的工具。
4. **刻意缺席**：没有 `write_memory`，没有 `write_playbook`。
   LLM 无权写任何记忆——记忆写入只发生在 verifier 判定通过之后。
   这是设计，不是遗漏。
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from ..core import (anomalies, clean, diagnosis, geo, metrics, params as params_mod,
                    segment, simplify)
from ..core.anomalies import RuleParams
from ..core.traj import Traj
from ..road import matcher as road_mod
from ..verifier import search as search_mod
from . import playbook as playbook_mod
from .store import TrajStore, state_hash


@dataclass
class ToolSpec:
    """一个工具的声明。"""

    name: str
    description: str
    parameters: Dict[str, Any]
    handler: Callable[..., Dict[str, Any]]
    side_effect: bool = False
    returns: str = ""

    def to_openai_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


def _obj(props: Dict[str, Any], required: Sequence[str] = ()) -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": props,
        "required": list(required),
        "additionalProperties": False,
    }


@dataclass
class ToolContext:
    """工具执行上下文。持有存储、路网与评估缓存。"""

    store: TrajStore = field(default_factory=TrajStore)
    road_matcher: Optional[road_mod.RoadMatcher] = None
    vault_dir: Optional[str] = None
    # 评估缓存：相同 (state_hash, params) 不重复评估
    eval_cache: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # 每次评估的原始指标，供 verifier 做方向判定
    last_eval: Optional[Dict[str, Any]] = None
    call_log: List[Dict[str, Any]] = field(default_factory=list)

    def log(self, tool: str, args: Dict[str, Any], ok: bool, ms: float,
            error: str = "") -> None:
        self.call_log.append({
            "tool": tool,
            "args": {k: v for k, v in args.items()},
            "ok": ok,
            "elapsed_ms": round(ms, 3),
            "error": error,
        })


class ToolRegistry:
    """工具集合。注册、派发、生成 schema。"""

    def __init__(self, ctx: Optional[ToolContext] = None) -> None:
        self.ctx = ctx or ToolContext()
        self._specs: Dict[str, ToolSpec] = {}
        self._register_all()

    # ---- 查询 -----------------------------------------------------------
    def names(self) -> List[str]:
        return sorted(self._specs)

    def spec(self, name: str) -> ToolSpec:
        if name not in self._specs:
            raise KeyError(f"未知工具 {name!r}；可用: {self.names()}")
        return self._specs[name]

    def openai_tools(self, only: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
        """生成 OpenAI function-calling 的工具声明。

        only 给出子集时只返回这些工具。这直接决定每轮的输入 token：
        全部 15 个工具的 schema 约 5700 字符，而精简到 7 个约 2500 字符。
        工具的 schema 每轮都要重发，所以这是省 token 最有效的一处。
        """
        names = list(only) if only else self.names()
        return [self._specs[n].to_openai_schema() for n in names if n in self._specs]

    def catalog(self, only: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
        """文本形式的工具目录（供 ReAct 协议使用），同样支持子集。"""
        names = list(only) if only else self.names()
        return [{
            "name": self._specs[n].name,
            "description": self._specs[n].description,
            "parameters": self._specs[n].parameters,
            "side_effect": self._specs[n].side_effect,
        } for n in names if n in self._specs]

    # ---- 派发 -----------------------------------------------------------
    def call(self, name: str, **kwargs) -> Dict[str, Any]:
        t0 = time.perf_counter()
        try:
            spec = self.spec(name)
        except KeyError as exc:
            self.ctx.log(name, kwargs, False, 0.0, str(exc))
            return {"ok": False, "error": str(exc)}
        try:
            out = spec.handler(self.ctx, **kwargs)
        except Exception as exc:                      # 工具异常不能中断 agent
            ms = (time.perf_counter() - t0) * 1000.0
            self.ctx.log(name, kwargs, False, ms, f"{type(exc).__name__}: {exc}")
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        ms = (time.perf_counter() - t0) * 1000.0
        self.ctx.log(name, kwargs, True, ms)
        out.setdefault("ok", True)
        out["_elapsed_ms"] = round(ms, 3)
        return out

    def call_json(self, name: str, arguments: Any) -> str:
        """从 JSON 字符串参数调用（OpenAI function-calling 的入参形式）。"""
        if isinstance(arguments, str):
            try:
                kwargs = json.loads(arguments) if arguments.strip() else {}
            except json.JSONDecodeError as exc:
                return json.dumps({"ok": False, "error": f"参数不是合法 JSON: {exc}"},
                                  ensure_ascii=False)
        elif isinstance(arguments, dict):
            kwargs = arguments
        else:
            kwargs = {}
        return json.dumps(self.call(name, **kwargs), ensure_ascii=False, default=str)

    # ---- 注册 -----------------------------------------------------------
    def register(self, spec: ToolSpec) -> None:
        self._specs[spec.name] = spec

    def _register_all(self) -> None:
        self._register_source()
        self._register_inspect()
        self._register_transform()
        self._register_evaluate()
        self._register_memory_read()
        self._register_render()

    # ------------------------------------------------------------------
    # 数据源
    # ------------------------------------------------------------------
    def _register_source(self) -> None:
        def load_trajectory(ctx: ToolContext, vehicle_id: str,
                            segment_index: int = 0) -> Dict[str, Any]:
            raw = getattr(ctx, "_raw_dataset", None)
            if raw is None:
                return {"ok": False,
                        "error": "尚未载入数据集；请先设置 ToolContext._raw_dataset"}
            if vehicle_id not in raw:
                return {"ok": False, "error": f"车辆 {vehicle_id!r} 不存在"}
            ts, coords = raw[vehicle_id]
            traj = Traj(vehicle_id=str(vehicle_id),
                        timestamps=[int(t) for t in ts],
                        coords=[(float(p[0]), float(p[1])) for p in coords],
                        segment_index=int(segment_index))
            handle = ctx.store.put(traj, operation="load")
            card = diagnosis.diagnose(traj, _rule_params(ctx))
            return {
                "handle": handle,
                "n_points": len(traj),
                "duration_s": round(traj.duration_s, 1),
                "diagnosis": card.to_dict(),
            }

        self.register(ToolSpec(
            name="load_trajectory",
            description="载入指定车辆的原始轨迹并返回诊断卡。返回的 handle 是后续所有工具的输入。",
            parameters=_obj({
                "vehicle_id": {"type": "string", "description": "车辆 ID，如 '246'"},
                "segment_index": {"type": "integer", "description": "分段序号，默认 0"},
            }, ["vehicle_id"]),
            handler=load_trajectory,
            returns="handle 字符串 + 诊断卡（无坐标）",
        ))

    # ------------------------------------------------------------------
    # 只读检查
    # ------------------------------------------------------------------
    def _register_inspect(self) -> None:
        def profile_trajectory(ctx: ToolContext, handle: str) -> Dict[str, Any]:
            traj = ctx.store.get(handle)
            card = diagnosis.diagnose(traj, _rule_params(ctx))
            return {"handle": handle, "diagnosis": card.to_dict()}

        def detect_anomalies(ctx: ToolContext, handle: str,
                            max_speed_mps: Optional[float] = None,
                            dist_threshold: Optional[float] = None,
                            max_turn_deg: Optional[float] = None,
                            with_indices: bool = False) -> Dict[str, Any]:
            traj = ctx.store.get(handle)
            r = _rule_params(ctx)
            if max_speed_mps is not None:
                r = _replace(r, max_speed_mps=float(max_speed_mps))
            if dist_threshold is not None:
                r = _replace(r, jump_threshold_m=float(dist_threshold))
            if max_turn_deg is not None:
                r = _replace(r, max_turn_deg=float(max_turn_deg))
            res = anomalies.detect_anomalies(traj, r)
            return {
                "handle": handle,
                "anomalies": res.to_dict(with_indices=bool(with_indices)),
                "window_hits": {
                    k: (len(v) if isinstance(v, list) else v)
                    for k, v in res.window_hits.items()
                },
            }

        def suggest_param_range(ctx: ToolContext, handle: str,
                                param: Optional[str] = None) -> Dict[str, Any]:
            traj = ctx.store.get(handle)
            table = params_mod.param_bounds_table()
            if param:
                table = [t for t in table if t["name"] == param]
                if not table:
                    return {"ok": False,
                            "error": f"未知参数 {param!r}；可用 "
                                     f"{[t['name'] for t in params_mod.param_bounds_table()]}"}
            dt = [d for d in traj.time_deltas()[1:] if d > 0]
            prior = params_mod.data_driven_priors(
                dt, traj.space_deltas_m()[1:],
                [v for v in traj.speeds_mps() if v > 0])
            return {"handle": handle, "param_specs": table, "data_driven": prior}

        self.register(ToolSpec(
            name="profile_trajectory",
            description="重新计算并返回指定句柄的诊断卡（标量统计，无坐标）。",
            parameters=_obj({"handle": {"type": "string"}}, ["handle"]),
            handler=profile_trajectory,
            returns="诊断卡",
        ))
        self.register(ToolSpec(
            name="detect_anomalies",
            description=("按规则检测异常点，返回各原因（reason）的计数。"
                         "可临时覆盖速度阈值、跳跃阈值与转向角阈值做对照。"),
            parameters=_obj({
                "handle": {"type": "string"},
                "max_speed_mps": {"type": "number", "description": "可选，覆盖速度阈值"},
                "dist_threshold": {"type": "number", "description": "可选，覆盖跳跃阈值"},
                "max_turn_deg": {"type": "number", "description": "可选，覆盖转向角阈值"},
                "with_indices": {"type": "boolean", "description": "是否返回点索引"},
            }, ["handle"]),
            handler=detect_anomalies,
            returns="各异常原因的计数与受影响点数",
        ))
        self.register(ToolSpec(
            name="suggest_param_range",
            description=("返回参数的物理先验区间与基于本条轨迹实测分布的建议起点。"
                         "建议起点只是参考，最终取值由你决定。"),
            parameters=_obj({
                "handle": {"type": "string"},
                "param": {"type": "string", "description": "可选，只看某个参数"},
            }, ["handle"]),
            handler=suggest_param_range,
            returns="参数区间表 + 数据驱动建议",
        ))

    # ------------------------------------------------------------------
    # 变换（有副作用：创建新句柄）
    # ------------------------------------------------------------------
    def _register_transform(self) -> None:
        def split_trajectory(ctx: ToolContext, handle: str,
                            dt_threshold: float = 30.0,
                            dist_threshold: float = 400.0,
                            min_points: int = 5,
                            min_length_m: float = 65.0) -> Dict[str, Any]:
            traj = ctx.store.get(handle)
            res = segment.split_trajectory(traj, dt_threshold, dist_threshold,
                                          int(min_points), min_length_m)
            handles = [ctx.store.put(s, parent=handle, operation="split",
                                     params={"dt_threshold": dt_threshold,
                                             "dist_threshold": dist_threshold})
                       for s in res.segments]
            return {
                "handle": handle,
                "summary": res.summary(),
                "segment_handles": handles,
                "segment_summaries": [s.summary() for s in res.segments[:20]],
                "n_segment_handles": len(handles),
            }

        def clean_trajectory(ctx: ToolContext, handle: str,
                            smooth_window: Optional[int] = None,
                            max_speed_mps: Optional[float] = None,
                            dist_threshold: Optional[float] = None) -> Dict[str, Any]:
            traj = ctx.store.get(handle)
            p = params_mod.default_params()
            if smooth_window is not None:
                p["smooth_window"] = float(smooth_window)
            if max_speed_mps is not None:
                p["max_speed_mps"] = float(max_speed_mps)
            if dist_threshold is not None:
                p["dist_threshold"] = float(dist_threshold)
            p, clamped = params_mod.clamp_params(p)
            cfg = clean.CleanConfig.from_params(p)
            out, report = clean.denoise_trajectory(traj, cfg)
            new_handle = ctx.store.put(out, parent=handle, operation="clean", params=p)
            return {
                "handle": new_handle,
                "parent": handle,
                "clean_report": report.to_dict(),
                "clamped_params": clamped,
            }

        def simplify_trajectory(ctx: ToolContext, handle: str,
                               tolerance_m: float = 5.0,
                               algorithm: str = "dp") -> Dict[str, Any]:
            traj = ctx.store.get(handle)
            spec = params_mod.get_spec("dp_tolerance")
            clamped = []
            if not spec.in_range(tolerance_m):
                clamped.append("dp_tolerance")
                tolerance_m = spec.clamp(tolerance_m)
            res = simplify.simplify_trajectory(traj, tolerance_m, algorithm)
            new_handle = ctx.store.put(
                res.traj, parent=handle, operation="simplify",
                params={"dp_tolerance": tolerance_m})
            return {
                "handle": new_handle,
                "parent": handle,
                "result": res.to_dict(),
                "clamped_params": clamped,
            }

        def apply_road_constraint(ctx: ToolContext, handle: str,
                                 tolerance_ratio: float = 0.1) -> Dict[str, Any]:
            traj = ctx.store.get(handle)
            tol = params_mod.get_spec("dp_tolerance").clamp(
                traj.length_m * float(tolerance_ratio) / 100.0) if traj.length_m else 15.0
            tol = max(5.0, min(30.0, tol))
            roads = road_mod.roads_from_trajectory(traj, simplify_tolerance_m=tol)
            if not roads:
                return {"ok": False, "error": "轨迹过短，无法构造道路折线约束"}
            m = road_mod.PolylineRoadMatcher(roads, match_tolerance_m=30.0)
            report = m.match(traj)
            snapped = Traj(traj.vehicle_id,
                           list(traj.timestamps),
                           list(report.matched_coords),
                           traj.segment_index,
                           list(traj.reasons))
            new_handle = ctx.store.put(
                snapped, parent=handle, operation="road_snap",
                params={"simplify_tolerance_m": tol})
            return {
                "handle": new_handle,
                "parent": handle,
                "road_stats": report.stats(),
                "n_roads": len(roads),
                "note": ("这是用轨迹自身几何构造的近似道路约束，"
                         "用于无真实路网时的对照实验"),
            }

        self.register(ToolSpec(
            name="split_trajectory",
            description="按时间间隔与空间跳跃阈值切分轨迹，返回每段的新句柄。",
            parameters=_obj({
                "handle": {"type": "string"},
                "dt_threshold": {"type": "number", "description": "时间间隔阈值（秒）"},
                "dist_threshold": {"type": "number", "description": "空间跳跃阈值（米）"},
                "min_points": {"type": "integer", "description": "每段最少点数"},
                "min_length_m": {"type": "number", "description": "每段最短长度（米）"},
            }, ["handle"]),
            handler=split_trajectory, side_effect=True,
            returns="切分摘要 + 各段句柄",
        ))
        self.register(ToolSpec(
            name="clean_trajectory",
            description=("执行去噪：折叠连续重复点、删非法坐标、修时间戳伪影、"
                         "删跳跃伪影、中值平滑速度毛刺。返回新句柄与清洗账本"
                         "（含每个被删点的异常原因）。"),
            parameters=_obj({
                "handle": {"type": "string"},
                "smooth_window": {"type": "integer", "description": "中值平滑窗口点数"},
                "max_speed_mps": {"type": "number", "description": "速度阈值（m/s）"},
                "dist_threshold": {"type": "number", "description": "跳跃阈值（米）"},
            }, ["handle"]),
            handler=clean_trajectory, side_effect=True,
            returns="新句柄 + 清洗账本 + 原因直方图",
        ))
        self.register(ToolSpec(
            name="simplify_trajectory",
            description=("用 Douglas-Peucker 压缩轨迹。tolerance_m 单位为米，"
                         "物理上限约为 GPS 定位精度 CEP（8m）。"
                         "algorithm 可选 dp / perp / dp_broken（后者仅用于对照）。"),
            parameters=_obj({
                "handle": {"type": "string"},
                "tolerance_m": {"type": "number", "description": "DP 容差（米）"},
                "algorithm": {"type": "string", "enum": ["dp", "perp", "dp_broken"]},
            }, ["handle"]),
            handler=simplify_trajectory, side_effect=True,
            returns="新句柄 + 压缩率 + 最大偏差",
        ))
        self.register(ToolSpec(
            name="apply_road_constraint",
            description=("把轨迹吸附到道路约束上（本轮用轨迹自身几何构造近似路网）。"
                         "用于比较「仅几何规则」与「加路网约束」两种方案的差异。"),
            parameters=_obj({
                "handle": {"type": "string"},
                "tolerance_ratio": {"type": "number", "description": "构造路网时的简化强度"},
            }, ["handle"]),
            handler=apply_road_constraint, side_effect=True,
            returns="吸附后句柄 + 路网匹配统计",
        ))

    # ------------------------------------------------------------------
    # 评估
    # ------------------------------------------------------------------
    def _register_evaluate(self) -> None:
        def evaluate(ctx: ToolContext, handle: str,
                     reference_handle: Optional[str] = None,
                     with_dtw: bool = False) -> Dict[str, Any]:
            traj = ctx.store.get(handle)
            ref = ctx.store.get(reference_handle) if reference_handle else None
            m = metrics.compute_metrics(traj, reference=ref, with_dtw=bool(with_dtw))
            out = {"handle": handle, "reference": reference_handle,
                   "metrics": m.to_dict()}
            if ctx.road_matcher is not None and getattr(ctx.road_matcher, "available", False):
                out["road"] = road_mod.road_constraint_report(traj, ctx.road_matcher)
            return out

        def compare_handles(ctx: ToolContext, handle_a: str,
                           handle_b: Optional[str] = None) -> Dict[str, Any]:
            """对比两个句柄的指标，并给出压缩率与保真度。"""
            ta = ctx.store.get(handle_a)
            tb = ctx.store.get(handle_b) if handle_b else ta
            ma = metrics.compute_metrics(ta, reference=tb if handle_b else None)
            n_before = len(tb)
            n_after = len(ta)
            comp = (1.0 - n_after / n_before) if n_before else 0.0
            fidelity = None
            if handle_b:
                scale = 30.0
                fidelity = max(0.0, 1.0 - (ma.hausdorff_m or 0.0) / scale)
            return {
                "handle_a": handle_a,
                "handle_b": handle_b,
                "n_points_a": n_after,
                "n_points_b": n_before,
                "compression_ratio": round(comp, 4),
                "length_ratio": round(ma.length_m / tb.length_m, 4) if tb.length_m else 1.0,
                "hausdorff_m": ma.hausdorff_m,
                "frechet_m": ma.frechet_m,
                "fidelity_proxy": None if fidelity is None else round(fidelity, 4),
                "metrics_a": ma.to_dict(),
                "caveat": (
                    "Hausdorff/Fréchet 是 handle_a 相对 handle_b 的总形变，"
                    "同时包含『清洗删点』与『几何简化』两部分贡献。"
                    "若 handle_b 是原始轨迹而你只想评价简化质量，"
                    "应以清洗后的句柄为参考，否则大偏差主要来自被删掉的跳变点，"
                    "会掩盖简化本身的误差。"
                ),
            }

        def run_search(ctx: ToolContext, handle: str,
                       param: str = "dp_tolerance",
                       n: int = 8,
                       reference_handle: Optional[str] = None) -> Dict[str, Any]:
            """在物理先验区间内扫描某参数，返回「质量—压缩率曲线」。"""
            traj = ctx.store.get(handle)
            ref = ctx.store.get(reference_handle) if reference_handle else None
            spec = params_mod.get_spec(param)
            curve: List[Dict[str, Any]] = []
            for v in search_mod.linspace(spec.low, spec.high, int(n),
                                         integer=spec.integer):
                if param == "dp_tolerance":
                    res = simplify.simplify_trajectory(traj, v)
                    m = metrics.compute_metrics(res.traj, reference=ref or traj)
                    curve.append({
                        "value": float(v),
                        "n_points": len(res.traj),
                        "compression_ratio": round(res.compression_ratio, 4),
                        "max_deviation_m": round(res.max_deviation_m, 3),
                        "hausdorff_m": None if m.hausdorff_m is None
                                       else round(m.hausdorff_m, 3),
                        "length_ratio": round(m.length_m / traj.length_m, 4)
                        if traj.length_m else 1.0,
                    })
                else:
                    curve.append({"value": float(v), "note": "该参数暂无扫描实现"})
            return {"handle": handle, "param": param, "spec": spec.to_dict(),
                    "curve": curve,
                    "note": ("曲线上的 knee point 是压缩率与保真度的最优折衷，"
                             "用 find_knee 工具可直接取得")}

        def find_knee(ctx: ToolContext, curve_json: str,
                      deviation_scale_m: float = 30.0) -> Dict[str, Any]:
            """对 run_search 返回的曲线求 knee point。

            保真度把最大偏差折算到 [0,1]：deviation_scale_m 是「偏差大到
            多少就算完全失真」。默认 30 m ≈ 3-4 倍 GPS CEP，
            含义是「偏差超过它就等于把点扔到了另一条路上」。
            """
            try:
                rows = json.loads(curve_json)
            except json.JSONDecodeError as exc:
                return {"ok": False, "error": f"curve_json 不是合法 JSON: {exc}"}
            if not isinstance(rows, list) or not rows:
                return {"ok": False, "error": "curve_json 应为非空数组"}
            from ..verifier import objective as obj_mod
            pts = []
            values: List[float] = []
            for r in rows:
                values.append(float(r.get("value", 0.0)))
                comp = float(r.get("compression_ratio", 0.0))
                dev = float(r.get("max_deviation_m", 0.0))
                pts.append((comp, obj_mod.fidelity_from_deviation(dev, 0.0,
                                                                 scale_m=deviation_scale_m)))
            kp = obj_mod.find_knee_point(pts)
            d = kp.to_dict()
            # 把参数值补回去：曲线里的字段名是 value，不是 tolerance_m
            if 0 <= kp.index < len(values):
                d["value"] = round(values[kp.index], 4)
                row = rows[kp.index]
                d["max_deviation_m"] = round(float(row.get("max_deviation_m", 0.0)), 3)
                d["n_points"] = int(row.get("n_points", 0))
            d["deviation_scale_m"] = deviation_scale_m
            # 同时给出 max_score 口径的结果，便于对照两种准则
            kp2 = obj_mod.find_knee_point(pts, method="max_score")
            if 0 <= kp2.index < len(values):
                d["max_score_alternative"] = {
                    "value": round(values[kp2.index], 4),
                    "compression": round(kp2.compression, 4),
                    "fidelity": round(kp2.fidelity, 4),
                }
            return {"knee": d, "curve_points": len(pts)}

        self.register(ToolSpec(
            name="evaluate",
            description="计算句柄的指标：点数、长度、Hausdorff/Fréchet 距离、运行时间。给出 reference_handle 时计算与它的偏差。",
            parameters=_obj({
                "handle": {"type": "string"},
                "reference_handle": {"type": "string", "description": "参考轨迹句柄"},
                "with_dtw": {"type": "boolean"},
            }, ["handle"]),
            handler=evaluate,
            returns="指标字典",
        ))
        self.register(ToolSpec(
            name="compare_handles",
            description="对比两个句柄：压缩率、长度比、Hausdorff、Fréchet 与保真度代理。",
            parameters=_obj({
                "handle_a": {"type": "string"},
                "handle_b": {"type": "string", "description": "参考基准句柄"},
            }, ["handle_a"]),
            handler=compare_handles,
            returns="对比指标",
        ))
        self.register(ToolSpec(
            name="run_search",
            description=("在物理先验区间内扫描单个参数，返回质量—压缩率曲线。"
                         "这是判断参数取值好坏的实测依据，不要凭直觉下结论。"),
            parameters=_obj({
                "handle": {"type": "string"},
                "param": {"type": "string", "description": "要扫描的参数名"},
                "n": {"type": "integer", "description": "扫描点数"},
                "reference_handle": {"type": "string"},
            }, ["handle"]),
            handler=run_search, side_effect=True,
            returns="曲线数据点",
        ))
        self.register(ToolSpec(
            name="find_knee",
            description="对 run_search 返回的曲线求 knee point（压缩率—保真度最优折衷点）。",
            parameters=_obj({
                "curve_json": {"type": "string", "description": "run_search 返回的 curve 字段的 JSON"},
            }, ["curve_json"]),
            handler=find_knee,
            returns="knee point 参数",
        ))

    # ------------------------------------------------------------------
    # 记忆读（刻意不含写）
    # ------------------------------------------------------------------
    def _register_memory_read(self) -> None:
        def query_memory(ctx: ToolContext, handle: str, k: int = 5) -> Dict[str, Any]:
            store = getattr(ctx, "_memory_store", None)
            if store is None:
                return {"ok": True, "available": False,
                        "note": "记忆库未接入（仅 L3 人类知识可用）",
                        "playbook": playbook_mod.playbook_digest(ctx.vault_dir)}
            traj = ctx.store.get(handle)
            card = diagnosis.diagnose(traj, _rule_params(ctx))
            cases = store.retrieve_similar(card, k=int(k))
            return {
                "available": True,
                "n_cases": len(cases),
                "cases": [c.to_dict() for c in cases],
                "playbook": playbook_mod.playbook_digest(ctx.vault_dir),
            }

        def query_playbook(ctx: ToolContext,
                          param: Optional[str] = None) -> Dict[str, Any]:
            return playbook_mod.playbook_digest(
                ctx.vault_dir, params=[param] if param else None)

        self.register(ToolSpec(
            name="query_memory",
            description=("按诊断特征检索相似历史案例（含已验证的参数区间），"
                         "并返回人类知识笔记（Obsidian playbook）。"
                         "注意：本工具只读，写入记忆由核验器负责。"),
            parameters=_obj({
                "handle": {"type": "string"},
                "k": {"type": "integer", "description": "检索条数"},
            }, ["handle"]),
            handler=query_memory,
            returns="相似案例 + playbook 摘要",
        ))
        self.register(ToolSpec(
            name="query_playbook",
            description="读取人类知识笔记（Obsidian vault，只读）。",
            parameters=_obj({
                "param": {"type": "string", "description": "可选，按参数名过滤"},
            }),
            handler=query_playbook,
            returns="playbook 摘要",
        ))

    # ------------------------------------------------------------------
    # 出图
    # ------------------------------------------------------------------
    def _register_render(self) -> None:
        def render(ctx: ToolContext, handle: str, kind: str = "clean_overlay",
                   reference_handle: Optional[str] = None,
                   out_dir: str = "figures") -> Dict[str, Any]:
            from ..report import figures
            traj = ctx.store.get(handle)
            ref = ctx.store.get(reference_handle) if reference_handle else None
            path = figures.render(kind, traj, reference=ref, out_dir=out_dir,
                                  road_matcher=ctx.road_matcher)
            return {"handle": handle, "kind": kind, "path": path}

        self.register(ToolSpec(
            name="render",
            description=("出图。kind 可选：clean_overlay（清洗前后叠加）、"
                         "anomaly_scatter（异常点分布）、heatmap（热力图）。"
                         "返回文件路径。"),
            parameters=_obj({
                "handle": {"type": "string"},
                "kind": {"type": "string",
                         "enum": ["clean_overlay", "anomaly_scatter", "heatmap"]},
                "reference_handle": {"type": "string"},
                "out_dir": {"type": "string"},
            }, ["handle"]),
            handler=render, side_effect=True,
            returns="图片路径",
        ))


# ---------------------------------------------------------------------------
# 辅助
# ---------------------------------------------------------------------------
def _rule_params(ctx: ToolContext) -> RuleParams:
    p = getattr(ctx, "_rule_params", None)
    return p if p is not None else RuleParams()


def _replace(rp: RuleParams, **kw) -> RuleParams:
    from dataclasses import replace as _dc_replace
    return _dc_replace(rp, **kw)


def attach_dataset(ctx: ToolContext, raw: Dict[str, Any]) -> ToolContext:
    """把原始数据集挂到上下文上（唯一需要坐标的地方，且不经 LLM）。"""
    ctx._raw_dataset = raw                      # type: ignore[attr-defined]
    return ctx


def attach_memory(ctx: ToolContext, memory_store) -> ToolContext:
    ctx._memory_store = memory_store            # type: ignore[attr-defined]
    return ctx
