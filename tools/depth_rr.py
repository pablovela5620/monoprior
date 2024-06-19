from pathlib import Path
from argparse import ArgumentParser
import rerun as rr
import cv2
import numpy as np

# from monopriors.depth_models import UniDepthPredictor
from monopriors.relative_depth_models import (
    UniDepthPredictor,
    DepthAnythingV2Predictor,
    RelativeDepthPrediction,
)
from icecream import ic
from einops import rearrange


# def get_intrinsics(H, W):
#     """
#     Intrinsics for a pinhole camera model.
#     Assume fov of 55 degrees and central principal point.
#     """
#     f = 0.5 * W / np.tan(0.5 * 55 * np.pi / 180.0)
#     cx = 0.5 * W
#     cy = 0.5 * H
#     return np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])


# def disparity_to_depth(disparity):
#     range1 = np.minimum(disparity.max() / (disparity.min() + 1e-6), 10.0)
#     max1 = disparity.max()
#     min1 = max1 / range1

#     depth = 1 / np.maximum(disparity, min1)
#     depth = (depth - depth.min()) / (depth.max() - depth.min())
#     depth = np.power(depth, 1.0 / 2.2)  # gamma correction for better visualization
#     depth = (depth * 255).astype(np.uint8)
#     return depth


# def call_unidepth(image_path):
#     predictor = UniDepthPredictor(device="cuda", backbone="vitl14")
#     bgr_hw3 = cv2.imread(str(image_path))
#     rgb_hw3 = cv2.cvtColor(bgr_hw3, cv2.COLOR_BGR2RGB)

#     depth_b1hw, K_b33, conf_b1hw = predictor(rgb_hw3, None)

#     # filter out low confidence regions
#     depth_b1hw = depth_b1hw * (conf_b1hw > 0.25).float()

#     depth_hw1 = rearrange(depth_b1hw, "1 c h w -> h w c").numpy(force=True)
#     K_33 = rearrange(K_b33, "1 r c -> r c").numpy(force=True)

#     # normalize the depth to 0-1
#     depth_hw1 = (depth_hw1 - depth_hw1.min()) / (depth_hw1.max() - depth_hw1.min())
#     depth_hw1 = (depth_hw1 * 255).astype(np.uint8)

#     return depth_hw1, K_33, rgb_hw3


def main(image_path: Path):
    parent_log_path = Path("world")
    bgr_hw3 = cv2.imread(str(image_path))
    rgb_hw3 = cv2.cvtColor(bgr_hw3, cv2.COLOR_BGR2RGB)

    predictor = DepthAnythingV2Predictor(device="cuda")
    relative_pred: RelativeDepthPrediction = predictor.__call__(rgb=rgb_hw3, K_33=None)

    cam_log_path = parent_log_path / "camera"
    pinhole_path = cam_log_path / "pinhole"
    h, w, _ = rgb_hw3.shape

    cam_T_world_44 = np.eye(4)

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
