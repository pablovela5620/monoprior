from dataclasses import dataclass
from pathlib import Path
import json


@dataclass
class PolycamCameraData:
    blur_score: float
    cx: float
    cy: float
    fx: float
    fy: float
    height: int
    manual_keyframe: bool
    neighbors: list[int]
    t_00: float
    t_01: float
    t_02: float
    t_03: float
    t_10: float
    t_11: float
    t_12: float
    t_13: float
    t_20: float
    t_21: float
    t_22: float
    t_23: float
    timestamp: int
    width: int


def load_raw_polycam_data(camera_path: Path) -> PolycamCameraData:
    assert camera_path.name.endswith(".json"), "Camera parameters must be a JSON file"
    with open(camera_path, "r") as f:
        data = json.load(f)

    polycam_camera_data = PolycamCameraData(**data)
    return polycam_camera_data
