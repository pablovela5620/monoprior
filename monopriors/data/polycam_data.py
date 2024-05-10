from dataclasses import dataclass, field
from pathlib import Path
import json
import numpy as np
from jaxtyping import Float32
import calibur


@dataclass
class PolycamCameraData:
    blur_score: float
    cx: float
    cy: float
    fx: float
    fy: float
    height: int
    manual_keyframe: bool
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
    neighbors: list[int] = field(default_factory=list)  # default value is an empty list

    def __post_init__(self) -> None:
        self.K_33: Float32[np.ndarray, "3 3"] = self.calculate_k33()
        self.world_T_cam_44: Float32[np.ndarray, "4 4"] = (
            self.calculate_world_t_cam_44()
        )
        self.cam_T_world_44: Float32[np.ndarray, "4 4"] = np.linalg.inv(
            self.world_T_cam_44
        )

    def calculate_k33(self) -> Float32[np.ndarray, "3 3"]:
        return np.array(
            [
                [self.fx, 0, self.cx],
                [0, self.fy, self.cy],
                [0, 0, 1],
            ],
            dtype=np.float32,
        )

    def calculate_world_t_cam_44(self) -> Float32[np.ndarray, "4 4"]:
        world_T_cam_44_gl = np.array(
            [
                [self.t_00, self.t_01, self.t_02, self.t_03],
                [self.t_10, self.t_11, self.t_12, self.t_13],
                [self.t_20, self.t_21, self.t_22, self.t_23],
                [0, 0, 0, 1],
            ],
            dtype=np.float32,
        )
        world_T_cam_44_cv = calibur.convert_pose(
            world_T_cam_44_gl,
            src_convention=calibur.CC.GL,
            dst_convention=calibur.CC.CV,
        )
        return world_T_cam_44_cv


def load_raw_polycam_data(camera_path: Path) -> PolycamCameraData:
    assert camera_path.name.endswith(".json"), "Camera parameters must be a JSON file"
    with open(camera_path, "r") as f:
        data = json.load(f)

    polycam_camera_data = PolycamCameraData(**data)
    return polycam_camera_data
