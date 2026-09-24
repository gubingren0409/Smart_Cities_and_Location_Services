"""OpenStreetMap 道路缓存与基于空间索引的真实路网匹配。

正式实验使用 OSM API 的 ``/api/0.6/map`` 小范围请求。下载后只保留可驾驶
``highway`` geometry，并写成 gzip GeoJSON；重复实验直接读取本地缓存。
距离与吸附均在 WGS84 / UTM zone 51N (EPSG:32651) 中计算。
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import time
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import xml.etree.ElementTree as ET

from pyproj import Transformer
import requests
from shapely.geometry import LineString, MultiLineString, Point, box, mapping, shape
from shapely.ops import transform
from shapely.strtree import STRtree

from ..core import geo
from ..core.traj import Traj
from .matcher import HIGHWAY_SPEED_LIMIT_MPS, MatchResult


WGS84 = "EPSG:4326"
UTM51N = "EPSG:32651"
OSM_MAP_URL = "https://api.openstreetmap.org/api/0.6/map"
DRIVABLE_EXCLUDED = {
    "bridleway", "construction", "corridor", "cycleway", "elevator",
    "footway", "path", "pedestrian", "platform", "proposed", "raceway",
    "steps",
}

_TO_UTM = Transformer.from_crs(WGS84, UTM51N, always_xy=True)
_TO_WGS84 = Transformer.from_crs(UTM51N, WGS84, always_xy=True)


def parse_maxspeed_mps(value: Any) -> Optional[float]:
    """解析 OSM ``maxspeed``，返回 m/s；不猜测非数值符号。"""
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text or text in {"none", "signals", "walk", "variable", "national"}:
        return None
    # 条件/双向标注时取第一个可读数值，原始 tag 仍保存在缓存中。
    found = re.search(r"(\d+(?:\.\d+)?)", text)
    if not found:
        return None
    number = float(found.group(1))
    if number <= 0:
        return None
    if "mph" in text:
        return number * 0.44704
    # OSM 数字默认单位为 km/h。
    return number / 3.6


def buffered_bbox(
    bbox_wgs84: Sequence[float], buffer_m: float = 300.0,
) -> Tuple[float, float, float, float]:
    """在 UTM 米制空间中给经纬度 bbox 增加 buffer。"""
    min_lon, min_lat, max_lon, max_lat = map(float, bbox_wgs84)
    corners = [
        _TO_UTM.transform(min_lon, min_lat),
        _TO_UTM.transform(max_lon, max_lat),
    ]
    min_x = min(p[0] for p in corners) - float(buffer_m)
    min_y = min(p[1] for p in corners) - float(buffer_m)
    max_x = max(p[0] for p in corners) + float(buffer_m)
    max_y = max(p[1] for p in corners) + float(buffer_m)
    out = [_TO_WGS84.transform(min_x, min_y), _TO_WGS84.transform(max_x, max_y)]
    return out[0][0], out[0][1], out[1][0], out[1][1]


def parse_osm_xml(data: bytes) -> List[Dict[str, Any]]:
    """把 OSM XML 转为可驾驶道路 GeoJSON features。"""
    root = ET.fromstring(data)
    nodes: Dict[str, Tuple[float, float]] = {}
    for node in root.findall("node"):
        node_id = node.attrib.get("id")
        if not node_id:
            continue
        try:
            nodes[node_id] = (float(node.attrib["lon"]), float(node.attrib["lat"]))
        except (KeyError, TypeError, ValueError):
            continue

    features: List[Dict[str, Any]] = []
    for way in root.findall("way"):
        tags = {
            str(tag.attrib.get("k")): str(tag.attrib.get("v"))
            for tag in way.findall("tag") if tag.attrib.get("k")
        }
        highway = tags.get("highway", "")
        if not highway or highway in DRIVABLE_EXCLUDED:
            continue
        coords = [
            nodes[nd.attrib["ref"]]
            for nd in way.findall("nd")
            if nd.attrib.get("ref") in nodes
        ]
        if len(coords) < 2:
            continue
        features.append({
            "type": "Feature",
            "id": f"way/{way.attrib.get('id', len(features))}",
            "properties": {
                "osm_way_id": str(way.attrib.get("id", "")),
                "highway": highway,
                "name": tags.get("name"),
                "maxspeed": tags.get("maxspeed"),
                "maxspeed_mps": parse_maxspeed_mps(tags.get("maxspeed")),
                "oneway": tags.get("oneway"),
            },
            "geometry": {"type": "LineString", "coordinates": coords},
        })
    return features


def _clip_features(
    features: Iterable[Mapping[str, Any]], bbox_wgs84: Sequence[float],
) -> List[Dict[str, Any]]:
    clip = box(*map(float, bbox_wgs84))
    out: List[Dict[str, Any]] = []
    for feature in features:
        geometry = shape(feature["geometry"])
        clipped = geometry.intersection(clip)
        if clipped.is_empty:
            continue
        if clipped.geom_type not in {"LineString", "MultiLineString"}:
            continue
        copied = dict(feature)
        copied["properties"] = dict(feature.get("properties") or {})
        copied["geometry"] = mapping(clipped)
        out.append(copied)
    return out


def write_geojson_gz(path: Path, features: Sequence[Mapping[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"type": "FeatureCollection", "features": list(features)}
    with gzip.open(path, "wt", encoding="utf-8", compresslevel=9) as handle:
        json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_geojson(path: Path) -> Dict[str, Any]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("type") != "FeatureCollection":
        raise ValueError(f"不是 GeoJSON FeatureCollection: {path}")
    return payload


def download_osm_cache(
    bbox_wgs84: Sequence[float],
    cache_path: Path,
    *,
    buffer_m: float = 300.0,
    metadata_path: Optional[Path] = None,
    implementation_commit: str = "unknown",
    timeout_s: float = 120.0,
    max_retries: int = 3,
    session: Optional[requests.Session] = None,
) -> Dict[str, Any]:
    """下载一个小 bbox 的 OSM map 数据并落盘；已有缓存时不访问网络。"""
    cache_path = Path(cache_path)
    metadata_path = metadata_path or cache_path.with_suffix(".metadata.json")
    if cache_path.exists() and metadata_path.exists():
        return json.loads(metadata_path.read_text(encoding="utf-8"))

    query_bbox = buffered_bbox(bbox_wgs84, buffer_m)
    client = session or requests.Session()
    headers = {"User-Agent": "ECNU-smart-city-lab1/1.0 (course experiment)"}
    last_error: Optional[Exception] = None
    response: Optional[requests.Response] = None
    for attempt in range(int(max_retries)):
        try:
            response = client.get(
                OSM_MAP_URL,
                params={"bbox": ",".join(f"{value:.7f}" for value in query_bbox)},
                headers=headers,
                timeout=float(timeout_s),
            )
            response.raise_for_status()
            break
        except (requests.RequestException, OSError) as exc:
            last_error = exc
            if attempt + 1 >= int(max_retries):
                raise RuntimeError(f"OSM 下载失败: {type(exc).__name__}: {exc}") from exc
            time.sleep(2 ** attempt)
    if response is None:
        raise RuntimeError(f"OSM 下载失败: {last_error}")

    features = _clip_features(parse_osm_xml(response.content), query_bbox)
    if not features:
        raise RuntimeError(f"OSM 查询返回 0 条可驾驶道路: bbox={query_bbox}")
    digest = write_geojson_gz(cache_path, features)
    highway_counts = Counter(
        str((feature.get("properties") or {}).get("highway") or "unknown")
        for feature in features
    )
    n_true_maxspeed = sum(
        (feature.get("properties") or {}).get("maxspeed_mps") is not None
        for feature in features
    )
    metadata = {
        "source": "OpenStreetMap",
        "source_url": OSM_MAP_URL,
        "license": "ODbL 1.0",
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_bbox_wgs84": list(map(float, bbox_wgs84)),
        "query_bbox_wgs84": list(map(float, query_bbox)),
        "buffer_m": float(buffer_m),
        "network_filter": "highway present; non-motorized/construction classes excluded",
        "road_feature_count": len(features),
        "highway_distribution": dict(sorted(highway_counts.items())),
        "features_with_true_maxspeed": int(n_true_maxspeed),
        "true_maxspeed_feature_rate": n_true_maxspeed / len(features),
        "cache_file": cache_path.name,
        "cache_sha256": digest,
        "implementation_commit": str(implementation_commit),
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata


class OSMRoadMatcher:
    """真实 OSM 道路 matcher，使用 UTM 米制几何和 Shapely STRtree。"""

    source = "OpenStreetMap"
    crs = UTM51N
    spatial_index = "Shapely STRtree"

    def __init__(
        self,
        features: Sequence[Mapping[str, Any]],
        match_tolerance_m: float = 30.0,
    ) -> None:
        self.match_tolerance_m = float(match_tolerance_m)
        self._lines: List[LineString] = []
        self._properties: List[Dict[str, Any]] = []
        for feature in features:
            geometry = shape(feature["geometry"])
            parts = list(geometry.geoms) if isinstance(geometry, MultiLineString) else [geometry]
            for part_index, part in enumerate(parts):
                if not isinstance(part, LineString) or len(part.coords) < 2:
                    continue
                projected = transform(_TO_UTM.transform, part)
                if projected.is_empty or projected.length <= 0:
                    continue
                props = dict(feature.get("properties") or {})
                props["road_id"] = str(
                    feature.get("id") or props.get("osm_way_id") or len(self._lines))
                props["part_index"] = part_index
                if props.get("maxspeed_mps") is not None:
                    try:
                        props["maxspeed_mps"] = float(props["maxspeed_mps"])
                    except (TypeError, ValueError):
                        props["maxspeed_mps"] = None
                else:
                    props["maxspeed_mps"] = parse_maxspeed_mps(props.get("maxspeed"))
                self._lines.append(projected)
                self._properties.append(props)
        self._tree = STRtree(self._lines) if self._lines else None

    @classmethod
    def from_geojson(
        cls, path: Path, match_tolerance_m: float = 30.0,
    ) -> "OSMRoadMatcher":
        payload = read_geojson(Path(path))
        return cls(payload["features"], match_tolerance_m=match_tolerance_m)

    @property
    def available(self) -> bool:
        return bool(self._lines)

    @property
    def n_roads(self) -> int:
        return len(self._lines)

    def _nearest(self, point: geo.LonLat) -> Tuple[Optional[int], float, Optional[Point]]:
        if self._tree is None or not geo.is_legal_lonlat(point[0], point[1]):
            return None, float("inf"), None
        projected_point = Point(*_TO_UTM.transform(float(point[0]), float(point[1])))
        index = int(self._tree.nearest(projected_point))
        line = self._lines[index]
        distance = float(projected_point.distance(line))
        snapped = line.interpolate(line.project(projected_point))
        return index, distance, snapped

    def nearest_info(self, point: geo.LonLat) -> Dict[str, Any]:
        index, distance, snapped = self._nearest(point)
        if index is None or snapped is None:
            return {
                "matched": False, "distance_m": float("inf"),
                "snapped": point, "road_id": None, "highway": None,
                "speed_limit_mps": None, "speed_limit_source": None,
            }
        props = self._properties[index]
        true_limit = props.get("maxspeed_mps")
        highway = str(props.get("highway") or "unclassified")
        fallback_limit = HIGHWAY_SPEED_LIMIT_MPS.get(highway)
        limit = float(true_limit) if true_limit is not None else fallback_limit
        limit_source = "osm_maxspeed" if true_limit is not None else (
            "highway_fallback" if fallback_limit is not None else None)
        is_match = distance <= self.match_tolerance_m
        lon, lat = _TO_WGS84.transform(float(snapped.x), float(snapped.y))
        return {
            "matched": is_match,
            "distance_m": distance,
            "snapped": (lon, lat) if is_match else point,
            "road_id": props.get("road_id"),
            "highway": highway,
            "speed_limit_mps": limit,
            "speed_limit_source": limit_source,
        }

    def match(self, traj: Traj) -> MatchResult:
        result = MatchResult()
        for point in traj.coords:
            info = self.nearest_info(point)
            result.matched_coords.append(info["snapped"])
            result.distances_m.append(float(info["distance_m"]))
            result.matched.append(bool(info["matched"]))
            result.road_ids.append(info["road_id"])
            result.highways.append(info["highway"])
            result.speed_limits_mps.append(info["speed_limit_mps"])
            result.speed_limit_sources.append(info["speed_limit_source"])
        return result

    def snap(self, point: geo.LonLat) -> geo.LonLat:
        return self.nearest_info(point)["snapped"]

    def distance_to_road_m(self, point: geo.LonLat) -> float:
        return float(self.nearest_info(point)["distance_m"])

    def speed_limit_info_at(self, point: geo.LonLat) -> Dict[str, Any]:
        info = self.nearest_info(point)
        return {
            "speed_limit_mps": info["speed_limit_mps"] if info["matched"] else None,
            "source": info["speed_limit_source"] if info["matched"] else None,
            "highway": info["highway"] if info["matched"] else None,
        }

    def speed_limit_at(self, point: geo.LonLat) -> Optional[float]:
        return self.speed_limit_info_at(point)["speed_limit_mps"]


def speed_violation_stats(traj: Traj, match: MatchResult) -> Dict[str, Any]:
    """按相邻点终点所在道路限速计算速度超限统计。"""
    speeds = traj.speeds_mps()
    comparable = 0
    violations = 0
    true_count = 0
    fallback_count = 0
    for index in range(1, min(len(speeds), len(match.matched))):
        limit = match.speed_limits_mps[index] if index < len(match.speed_limits_mps) else None
        source = match.speed_limit_sources[index] if index < len(match.speed_limit_sources) else None
        if not match.matched[index] or limit is None or speeds[index] <= 0:
            continue
        comparable += 1
        if source == "osm_maxspeed":
            true_count += 1
        elif source == "highway_fallback":
            fallback_count += 1
        if speeds[index] > float(limit):
            violations += 1
    return {
        "n_speed_comparable": comparable,
        "road_speed_violation_count": violations,
        "road_speed_violation_rate": violations / comparable if comparable else None,
        "true_maxspeed_point_count": true_count,
        "fallback_speed_point_count": fallback_count,
        "true_maxspeed_point_rate": true_count / comparable if comparable else None,
        "fallback_speed_point_rate": fallback_count / comparable if comparable else None,
    }
