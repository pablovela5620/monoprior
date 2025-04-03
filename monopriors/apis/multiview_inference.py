from dataclasses import dataclass
from pathlib import Path
from timeit import default_timer as timer
from typing import Literal

import cv2
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import torch
from jaxtyping import Float32, UInt8, UInt16
from numpy import ndarray
from simplecv.rerun_log_utils import RerunTyroConfig, log_pinhole

from monopriors.multiview_models.vggt_model import MultiviewPred, VGGTPredictor

np.set_printoptions(suppress=True)

SUPPORTED_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")
device = "cuda" if torch.cuda.is_available() else "cpu"


def create_blueprint(parent_log_path: Path, image_paths: list[Path]) -> rrb.Blueprint:
    view3d = rrb.Spatial3DView(
        origin=f"{parent_log_path}",
        contents=[
            "+ $origin/**",
            # don't include depths in the 3D view, as they can be very noisy
            *[f"- /{parent_log_path}/camera_{i}/pinhole/depth" for i in range(len(image_paths))],
        ],
    )
    view2d = rrb.Vertical(
        contents=[
            rrb.Horizontal(
                contents=[
                    rrb.Spatial2DView(
                        origin=f"{parent_log_path}/camera_{i}/pinhole/",
                        contents=[
                            "+ $origin/**",
                        ],
                        name="Pinhole Content",
                    ),
                    rrb.Spatial2DView(
                        origin=f"{parent_log_path}/camera_{i}/pinhole/confidence",
                        contents=[
                            "+ $origin/**",
                        ],
                        name="Confidence Map",
                    ),
                ]
            )
            # show at most 4 cameras
            for i in range(min(4, len(image_paths)))
        ]
    )

    blueprint = rrb.Blueprint(rrb.Horizontal(contents=[view3d, view2d], column_shares=[3, 1]), collapse_panels=True)
    return blueprint


@dataclass
class VGGTInferenceConfig:
    rr_config: RerunTyroConfig
    image_dir: Path
    confidence_threshold: int | float = 50.0
    """Confidence threshold value between 0 and 100.0"""
    preprocessing_mode: Literal["crop", "pad"] = "crop"
    """Mode for image preprocessing: 'crop' preserves aspect ratio, 'pad' adds white padding"""


def run_inference(config: VGGTInferenceConfig) -> None:
    print("Running inference on images in", config.image_dir)

    start: float = timer()
    image_paths = []

    for ext in SUPPORTED_IMAGE_EXTENSIONS:
        image_paths.extend(config.image_dir.glob(f"*{ext}"))
    image_paths: list[Path] = sorted(image_paths)
    assert len(image_paths) > 0, (
        f"No images found in {config.image_dir} in supported formats {SUPPORTED_IMAGE_EXTENSIONS}"
    )

    bgr_list: list[UInt8[ndarray, "H W 3"]] = [cv2.imread(str(image_path)) for image_path in image_paths]
    rgb_list: list[UInt8[ndarray, "H W 3"]] = [cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) for bgr in bgr_list]

    # initialize rerun
    parent_log_path = Path("world")
    blueprint = create_blueprint(parent_log_path=parent_log_path, image_paths=image_paths)
    rr.send_blueprint(blueprint=blueprint)
    rr.log(f"{parent_log_path}", rr.ViewCoordinates.RDF, static=True)
    # Apply the rotation to the root coordinate system
    rr.log(
        f"{parent_log_path}",
        rr.Transform3D(rotation=rr.RotationAxisAngle(axis=(0, 1, 0), radians=-np.pi / 4)),
        static=True,
    )

    vggt_predictor = VGGTPredictor(
        device=device,
        confidence_threshold=config.confidence_threshold,
        preprocessing_mode=config.preprocessing_mode,
    )
    calibration_data: list[MultiviewPred] = vggt_predictor(rgb_list=rgb_list)

    rr.log(
        f"{parent_log_path}/point_cloud",
        rr.Points3D(
            calibration_data[0].pointcloud.points,
            colors=calibration_data[0].pointcloud.colors,
        ),
        static=True,
    )
    calib_data: MultiviewPred
    for calib_data in calibration_data:
        cam_log_path: Path = parent_log_path / calib_data.cam_name

        mask: Float32[ndarray, "H W"] = calib_data.confidence_mask.astype(np.float32)
        depth_map: UInt16[ndarray, "H W"] = calib_data.depth_map

        log_pinhole(
            calib_data.pinhole_param,
            cam_log_path=cam_log_path,
            image_plane_distance=100.0,
            static=True,
        )

        rr.log(
            f"{cam_log_path}/pinhole/image", rr.Image(calib_data.rgb_image, color_model=rr.ColorModel.RGB), static=True
        )
        rr.log(
            f"{cam_log_path}/pinhole/confidence",
            rr.Image(mask),
            static=True,
        )
        rr.log(
            f"{cam_log_path}/pinhole/depth",
            rr.DepthImage(depth_map, draw_order=1),
            static=True,
        )
    print(f"Inference completed in {timer() - start:.2f} seconds")
