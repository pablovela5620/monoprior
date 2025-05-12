import os
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
from scipy.spatial.transform import Rotation
from simplecv.camera_parameters import Extrinsics
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


def write_colmap_cameras_txt(
    file_path: str, intrinsics: Float32[ndarray, "n 3 3"], image_width: int, image_height: int
) -> None:
    """Write camera intrinsics to COLMAP cameras.txt format."""
    with open(file_path, "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"# Number of cameras: {len(intrinsics)}\n")

        for i, intrinsic in enumerate(intrinsics):
            camera_id = i + 1  # COLMAP uses 1-indexed camera IDs
            model = "PINHOLE"

            fx = intrinsic[0, 0]
            fy = intrinsic[1, 1]
            cx = intrinsic[0, 2]
            cy = intrinsic[1, 2]

            f.write(f"{camera_id} {model} {image_width} {image_height} {fx} {fy} {cx} {cy}\n")


def write_colmap_images_txt(
    file_path: str,
    quaternions: np.ndarray,
    translations: np.ndarray,
    image_points2D: list[list],  # empty list for now
    image_names: list[str],
):
    """Write camera poses and keypoints to COLMAP images.txt format."""
    with open(file_path, "w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")

        # num_points = sum(len(points) for points in image_points2D)
        # avg_points = num_points / len(image_points2D) if image_points2D else 0
        avg_points = 0  # Placeholder for now
        f.write(f"# Number of images: {len(quaternions)}, mean observations per image: {avg_points:.1f}\n")

        for i in range(len(quaternions)):
            image_id = i + 1
            camera_id = i + 1

            qw, qx, qy, qz = quaternions[i]
            tx, ty, tz = translations[i]

            f.write(f"{image_id} {qw} {qx} {qy} {qz} {tx} {ty} {tz} {camera_id} {os.path.basename(image_names[i])}\n")

            # points_line = " ".join([f"{x} {y} {point3d_id + 1}" for x, y, point3d_id in image_points2D[i]])
            points_line = " ".join([""])  # Placeholder for now
            f.write(f"{points_line}\n")


def write_colmap_points3D_txt(file_path: str, points3D: list) -> None:
    """Write 3D points and tracks to COLMAP points3D.txt format."""
    with open(file_path, "w") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")

        # set the average track length to 0 for now
        avg_track_length: Literal[0] = 0
        f.write(f"# Number of points: {len(points3D)}, mean track length: {avg_track_length:.4f}\n")

        for point in points3D:
            point_id = point["id"] + 1
            x, y, z = point["xyz"]
            r, g, b = point["rgb"]
            error = point["error"]

            # track = " ".join([f"{img_id + 1} {point2d_idx}" for img_id, point2d_idx in point["track"]])
            track = " ".join([""])

            f.write(f"{point_id} {x} {y} {z} {int(r)} {int(g)} {int(b)} {error} {track}\n")


def extrinsic_to_colmap_format(mv_pred_list: list[MultiviewPred]) -> tuple[np.ndarray, np.ndarray]:
    """Convert extrinsic matrices to COLMAP format (quaternion + translation)."""
    quaternions = []
    translations = []

    for mv_pred in mv_pred_list:
        extrinsic: Extrinsics = mv_pred.pinhole_param.extrinsics
        # VGGT's extrinsic is camera-to-world (R|t) format
        R = extrinsic.cam_R_world
        t = extrinsic.cam_t_world

        # Convert rotation matrix to quaternion
        # COLMAP quaternion format is [qw, qx, qy, qz]
        rot = Rotation.from_matrix(R)
        quat = rot.as_quat()  # scipy returns [x, y, z, w]
        quat = np.array([quat[3], quat[0], quat[1], quat[2]])  # Convert to [w, x, y, z]

        quaternions.append(quat)
        translations.append(t)

    return np.array(quaternions), np.array(translations)


@dataclass
class VGGTInferenceConfig:
    rr_config: RerunTyroConfig
    image_dir: Path
    confidence_threshold: int | float = 50.0
    """Confidence threshold value between 0 and 100.0"""
    preprocessing_mode: Literal["crop", "pad"] = "crop"
    """Mode for image preprocessing: 'crop' preserves aspect ratio, 'pad' adds white padding"""
    output_dir: Path | None = None
    """Output directory for colmap version. If None, results are not saved."""


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
    mv_pred_list: list[MultiviewPred] = vggt_predictor(rgb_list=rgb_list)

    pointcloud = mv_pred_list[0].pointcloud
    pc_conf_mask = mv_pred_list[0].pointcloud_conf

    filtered_points = np.asarray(pointcloud.points)[pc_conf_mask]
    filtered_colors = np.asarray(pointcloud.colors)[pc_conf_mask]

    rr.log(
        f"{parent_log_path}/point_cloud",
        rr.Points3D(
            filtered_points,
            colors=filtered_colors,
        ),
        static=True,
    )

    intri_stack_list: list[Float32[ndarray, "3 3"]] = []
    mv_pred: MultiviewPred
    for mv_pred in mv_pred_list:
        cam_log_path: Path = parent_log_path / mv_pred.cam_name

        mask: Float32[ndarray, "H W"] = mv_pred.confidence_mask.astype(np.float32)
        depth_map: UInt16[ndarray, "H W"] = mv_pred.depth_map

        # Filter the depth map based on confidence mask
        filtered_depth_map = np.where(mask > 0, depth_map, 0)

        log_pinhole(
            mv_pred.pinhole_param,
            cam_log_path=cam_log_path,
            image_plane_distance=100.0,
            static=True,
        )

        intri_stack_list.append(mv_pred.pinhole_param.intrinsics.k_matrix)

        rr.log(f"{cam_log_path}/pinhole/image", rr.Image(mv_pred.rgb_image, color_model=rr.ColorModel.RGB), static=True)
        rr.log(
            f"{cam_log_path}/pinhole/confidence",
            rr.Image(mask, draw_order=-10),
            static=True,
        )
        rr.log(
            f"{cam_log_path}/pinhole/depth",
            rr.DepthImage(filtered_depth_map, draw_order=1),
            static=True,
        )

    intri_stack = np.stack(intri_stack_list, axis=0, dtype=np.float32)

    if config.output_dir is not None:
        config.output_dir.mkdir(parents=True, exist_ok=True)
        write_colmap_cameras_txt(
            file_path=str(config.output_dir / "cameras.txt"),
            intrinsics=intri_stack,
            image_width=mv_pred_list[0].pinhole_param.intrinsics.width,
            image_height=mv_pred_list[0].pinhole_param.intrinsics.height,
        )
        quaternions, translations = extrinsic_to_colmap_format(mv_pred_list)
        image_points2D_empty = [[] for _ in range(len(mv_pred_list))]  # Initialize with empty lists
        write_colmap_images_txt(
            file_path=str(config.output_dir / "images.txt"),
            quaternions=quaternions,
            translations=translations,
            image_points2D=image_points2D_empty,
            image_names=[image_path.name for image_path in image_paths],
        )

        write_colmap_points3D_txt(
            file_path=str(config.output_dir / "points3D.txt"),
            points3D=[
                {
                    "id": i,
                    "xyz": xyz,
                    "rgb": rgb * 255,
                    "error": 1.0,
                    "track": [],
                }
                for i, (xyz, rgb) in enumerate(
                    zip(
                        filtered_points,
                        filtered_colors,
                        strict=True,
                    )
                )
            ],
        )
    print(f"Inference completed in {timer() - start:.2f} seconds")
