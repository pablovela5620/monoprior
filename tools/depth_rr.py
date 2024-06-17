from pathlib import Path
from argparse import ArgumentParser
import rerun as rr
import cv2
import numpy as np
from monopriors.depth_models import DepthAnythingPredictor, UniDepthPredictor
from icecream import ic


def get_intrinsics(H, W):
    """
    Intrinsics for a pinhole camera model.
    Assume fov of 55 degrees and central principal point.
    """
    f = 0.5 * W / np.tan(0.5 * 55 * np.pi / 180.0)
    cx = 0.5 * W
    cy = 0.5 * H
    return np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])


def main(image_path: Path):
    parent_log_path = Path("world")
    # predictor = DepthAnythingPredictor(device="cuda")
    predictor = UniDepthPredictor(device="cuda", backbone="vitl14")
    bgr_hw3 = cv2.imread(str(image_path))
    rgb_hw3 = cv2.cvtColor(bgr_hw3, cv2.COLOR_BGR2RGB)

    depth, K_33_pred = predictor(rgb_hw3, None)
    # depth *= -1.0
    # depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    # depth = depth.astype(np.uint8)

    cam_log_path = parent_log_path / "camera"
    pinhole_path = cam_log_path / "pinhole"
    h, w, _ = rgb_hw3.shape

    cam_T_world_44 = np.eye(4)
    # K_33 = get_intrinsics(h, w)

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
            image_from_camera=K_33_pred.numpy(force=True),
            width=w,
            height=h,
            camera_xyz=rr.ViewCoordinates.RDF,
        ),
    )
    rr.log(f"{pinhole_path}/image", rr.Image(rgb_hw3))
    rr.log(f"{pinhole_path}/depth", rr.DepthImage(depth.numpy(force=True)))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--image-path", type=Path, default="data/39703866203.jpg")
    rr.script_add_args(parser)
    args = parser.parse_args()
    rr.script_setup(args, "depth view")
    main(args.image_path)
    rr.script_teardown(args)
