import os
import json
from utils import util


def geo_json_generate_traj_from_dict(link_wkts, type_style="LineString"):
    res = {
        "type": "FeatureCollection",
        "features": []
    }
    for k, v in link_wkts.items():
        t = {
            "type": "Feature",
            "geometry": {
                "type": type_style,
                "coordinates": util.transform_points_mercator_to_wgs84(v[1]) if v[1][0][0] > 1000 else v[1]
            },
            "properties": {
                "order_id": k,
                # /
            }

        }

        res["features"].append(t)

    return res




def visual_raw_traj(traj, path, type_style="LineString"):
    """
    Args:
        list traj
        String path
        String type_style(Multipoint, LineString, Point)
    Returns:
        .json
    """
    if os.path.exists(path):
        os.remove(path)
    with open(path, "w") as f:
        json.dump(geo_json_generate_traj_from_dict(traj, type_style), f)



