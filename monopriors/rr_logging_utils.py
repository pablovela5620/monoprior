import numpy as np
from pathlib import Path
import rerun as rr
import rerun.blueprint as rrb
from jaxtyping import UInt8, Float64, Float32, Bool
from monopriors.relative_depth_models import RelativeDepthPrediction
from monopriors.relative_depth_models.base_relative_depth import BaseRelativePredictor
from einops import rearrange
from monopriors.depth_utils import depth_to_points, clip_disparity, depth_edges_mask


def log_relative_pred(
    parent_log_path: Path,
    relative_pred: RelativeDepthPrediction,
    rgb_hw3: UInt8[np.ndarray, "h w 3"],
    remove_flying_pixels: bool = True,
    jpeg_quality: int = 90,
) -> None:
    cam_log_path: Path = parent_log_path / "camera"
    pinhole_path: Path = cam_log_path / "pinhole"

    # assume camera is at the origin
    cam_T_world_44: Float64[np.ndarray, "4 4"] = np.eye(4)

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
            width=rgb_hw3.shape[1],
            height=rgb_hw3.shape[0],
            camera_xyz=rr.ViewCoordinates.RDF,
        ),
    )
    rr.log(
        f"{pinhole_path}/image", rr.Image(rgb_hw3).compress(jpeg_quality=jpeg_quality)
    )

    depth_hw: Float32[np.ndarray, "h w"] = relative_pred.depth
    if remove_flying_pixels:
        edges_mask: Bool[np.ndarray, "h w"] = depth_edges_mask(depth_hw, threshold=0.1)
        depth_hw: Float32[np.ndarray, "h w"] = depth_hw * ~edges_mask

    rr.log(f"{pinhole_path}/depth", rr.DepthImage(depth_hw))

    # removes outliers from disparity (sometimes we can get weirdly large values)
    clipped_disparity: UInt8[np.ndarray, "h w"] = clip_disparity(
        relative_pred.disparity
    )

    # log to cam_log_path to avoid backprojecting disparity
    rr.log(f"{cam_log_path}/disparity", rr.DepthImage(clipped_disparity))

    depth_1hw: Float32[np.ndarray, "h w"] = rearrange(depth_hw, "h w -> 1 h w")
    pts_3d: Float32[np.ndarray, "h w 3"] = depth_to_points(
        depth_1hw, relative_pred.K_33
    )

    rr.log(
        f"{parent_log_path}/point_cloud",
        rr.Points3D(
            positions=pts_3d.reshape(-1, 3),
            colors=rgb_hw3.reshape(-1, 3),
        ),
    )


def create_relative_depth_blueprint(
    models: list[BaseRelativePredictor],
) -> rrb.Blueprint:
    model_names: list[str] = [model.__class__.__name__ for model in models]
    contents = [
        rrb.Spatial3DView(origin=f"{model_names[0]}"),
        rrb.Vertical(
            rrb.Spatial2DView(
                origin=f"{model_names[0]}/camera/pinhole/image",
            ),
            rrb.Spatial2DView(
                origin=f"{model_names[0]}/camera/pinhole/depth",
            ),
            rrb.Spatial2DView(
                origin=f"{model_names[0]}/camera/disparity",
            ),
        ),
        rrb.Spatial3DView(origin=f"{model_names[1]}"),
        rrb.Vertical(
            rrb.Spatial2DView(
                origin=f"{model_names[1]}/camera/pinhole/image",
            ),
            rrb.Spatial2DView(
                origin=f"{model_names[1]}/camera/pinhole/depth",
            ),
            rrb.Spatial2DView(
                origin=f"{model_names[1]}/camera/disparity",
            ),
        ),
    ]
    blueprint = rrb.Blueprint(
        rrb.Horizontal(
            contents=contents,
            column_shares=(3, 1, 3, 1),
        ),
        collapse_panels=True,
    )
    return blueprint
