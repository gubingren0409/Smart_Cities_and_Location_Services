"""可选道路约束后端。"""

from .matcher import (HIGHWAY_SPEED_LIMIT_MPS, MatchResult, NoRoadMatcher,
                      PolylineRoadMatcher, RoadMatcher, SegmentRoad,
                      road_constraint_report, roads_from_trajectory)
from .osm import (OSMRoadMatcher, buffered_bbox, download_osm_cache,
                  parse_maxspeed_mps, parse_osm_xml, read_geojson,
                  speed_violation_stats)

__all__ = [
    "HIGHWAY_SPEED_LIMIT_MPS", "MatchResult", "NoRoadMatcher",
    "PolylineRoadMatcher", "RoadMatcher", "SegmentRoad", "OSMRoadMatcher",
    "buffered_bbox", "download_osm_cache", "parse_maxspeed_mps",
    "parse_osm_xml", "read_geojson", "road_constraint_report",
    "roads_from_trajectory", "speed_violation_stats",
]
