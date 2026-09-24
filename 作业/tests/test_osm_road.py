from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from traj_agent.core import geo
from traj_agent.core.traj import Traj
from traj_agent.road.osm import (OSMRoadMatcher, parse_maxspeed_mps,
                                 parse_osm_xml, read_geojson)
from traj_agent.tools.registry import ToolContext, ToolRegistry


OSM_FIXTURE = b"""<?xml version='1.0' encoding='UTF-8'?>
<osm version='0.6'>
  <node id='1' lat='31.2300' lon='121.4700'/>
  <node id='2' lat='31.2300' lon='121.4710'/>
  <node id='3' lat='31.2310' lon='121.4710'/>
  <node id='4' lat='31.2295' lon='121.4700'/>
  <way id='10'>
    <nd ref='1'/><nd ref='2'/><nd ref='3'/>
    <tag k='highway' v='primary'/><tag k='maxspeed' v='60'/>
  </way>
  <way id='11'>
    <nd ref='4'/><nd ref='1'/>
    <tag k='highway' v='footway'/>
  </way>
</osm>"""


def test_parse_osm_fixture_and_maxspeed():
    features = parse_osm_xml(OSM_FIXTURE)
    assert len(features) == 1
    assert features[0]["properties"]["highway"] == "primary"
    assert features[0]["properties"]["maxspeed_mps"] == pytest.approx(60 / 3.6)
    assert parse_maxspeed_mps("30 mph") == pytest.approx(13.4112)
    assert parse_maxspeed_mps("50;70") == pytest.approx(50 / 3.6)
    assert parse_maxspeed_mps("signals") is None


def test_read_gzip_geojson_and_multiline(tmp_path: Path):
    path = tmp_path / "roads.geojson.gz"
    payload = {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature", "id": "way/1",
            "properties": {"highway": "residential"},
            "geometry": {"type": "MultiLineString", "coordinates": [
                [[121.4700, 31.2300], [121.4710, 31.2300]],
                [[121.4710, 31.2300], [121.4710, 31.2310]],
            ]},
        }],
    }
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle)
    assert read_geojson(path)["type"] == "FeatureCollection"
    matcher = OSMRoadMatcher.from_geojson(path, match_tolerance_m=30)
    assert matcher.n_roads == 2


def test_strtree_distance_snap_match_and_speed_source():
    matcher = OSMRoadMatcher(parse_osm_xml(OSM_FIXTURE), match_tolerance_m=30)
    point = (121.4705, 31.2301)
    info = matcher.nearest_info(point)
    assert info["matched"]
    assert 5 < info["distance_m"] < 20
    assert info["highway"] == "primary"
    assert info["speed_limit_source"] == "osm_maxspeed"
    assert info["speed_limit_mps"] == pytest.approx(60 / 3.6)
    snapped = matcher.snap(point)
    assert matcher.distance_to_road_m(snapped) < 0.01
    traj = Traj("x", [0, 10], [point, (121.4708, 31.2301)])
    result = matcher.match(traj)
    assert result.match_rate == 1.0
    assert result.stats()["n_unmatched"] == 0


def test_highway_fallback_and_far_point_remains_unsnapped():
    feature = {
        "type": "Feature", "id": "way/2",
        "properties": {"highway": "residential"},
        "geometry": {"type": "LineString", "coordinates": [
            [121.4700, 31.2300], [121.4710, 31.2300],
        ]},
    }
    matcher = OSMRoadMatcher([feature], match_tolerance_m=20)
    near = matcher.nearest_info((121.4705, 31.23005))
    assert near["speed_limit_source"] == "highway_fallback"
    assert near["speed_limit_mps"] == pytest.approx(8.3)
    far_point = (121.4705, 31.2400)
    far = matcher.nearest_info(far_point)
    assert not far["matched"]
    assert far["snapped"] == far_point


def test_apply_road_constraint_executes_real_backend_and_keeps_objective_separate():
    matcher = OSMRoadMatcher(parse_osm_xml(OSM_FIXTURE), match_tolerance_m=30)
    ctx = ToolContext(road_matcher=matcher)
    registry = ToolRegistry(ctx)
    traj = Traj("x", [0, 10, 20], [
        (121.4701, 31.23008),
        (121.4705, 31.23008),
        (121.4709, 31.23008),
    ])
    handle = ctx.store.put(traj, operation="test")
    out = registry.call("apply_road_constraint", handle=handle)
    assert out["ok"]
    assert out["road_backend"] == "OSMRoadMatcher"
    assert out["road_source"] == "OpenStreetMap"
    snapped = ctx.store.get(out["handle"])
    assert snapped.coords != traj.coords
    assert max(geo.local_distance_m(a, b) for a, b in zip(traj.coords, snapped.coords)) < 30
    # 工具只生成新 handle，没有修改 Objective 定义或任何 Objective 权重。
    assert "objective" not in out
