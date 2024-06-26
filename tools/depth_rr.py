from pathlib import Path
from argparse import ArgumentParser
import rerun as rr
import cv2
import numpy as np
from beartype import beartype

# from monopriors.depth_models import UniDepthPredictor
from monopriors.relative_depth_models import (
    DepthAnythingV2Predictor,
    RelativeDepthPrediction,
)
from jaxtyping import Float32


def main(image_path: Path) -> int:
    parent_log_path = Path("world")
    bgr_hw3 = cv2.imread(str(image_path))
    rgb_hw3 = cv2.cvtColor(bgr_hw3, cv2.COLOR_BGR2RGB)

    predictor = DepthAnythingV2Predictor(device="cuda")
    relative_pred: RelativeDepthPrediction = predictor.__call__(rgb=rgb_hw3, K_33=None)

    cam_log_path: Path = parent_log_path / "camera"
    pinhole_path: Path = cam_log_path / "pinhole"
    h, w, _ = rgb_hw3.shape

    cam_T_world_44: int = np.eye(4)

    rr.log(f"{parent_log_path}", rr.ViewCoordinates.RDF, timeless=True)

    rr.log(
        f"{cam_log_path}",
        rr.Transform3D(
            translation=cam_T_world_44[:3, 3],
            mat3x3=cam_T_world_44[:3, :3],
            from_parent=True,
        ),
    )
    rr.log(
        f"{pinhole_path}",
        rr.Pinhole(
            image_from_camera=relative_pred.K_33,
            width=w,
            height=h,
            camera_xyz=rr.ViewCoordinates.RDF,
        ),
    )
    rr.log(f"{pinhole_path}/image", rr.Image(rgb_hw3))
    rr.log(f"{pinhole_path}/depth", rr.DepthImage(relative_pred.depth))
    # log to cam_log_path to avoid backprojecting disparity
    rr.log(f"{cam_log_path}/disparity", rr.DepthImage(relative_pred.disparity))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--image-path", type=Path, default="data/39703866203.jpg")
    rr.script_add_args(parser)
    args = parser.parse_args()
    rr.script_setup(args, "depth view")
    main(args.image_path)
    rr.script_teardown(args)
