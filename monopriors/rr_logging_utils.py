import numpy as np
from pathlib import Path
import rerun as rr
import rerun.blueprint as rrb
from jaxtyping import UInt8, Float64
from monopriors.relative_depth_models import RelativeDepthPrediction
from monopriors.relative_depth_models.base_relative_depth import BaseRelativePredictor


def log_relative_pred(
    parent_log_path: Path,
    relative_pred: RelativeDepthPrediction,
    rgb_hw3: UInt8[np.ndarray, "h w 3"],
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

    rr.log(f"{pinhole_path}/depth", rr.DepthImage(relative_pred.depth))

    # Get the 95th percentile of the disparity values
    p95 = np.percentile(relative_pred.disparity, 99)
    # Clip the disparity values at 105% of the 95th percentile
    clipped_disparity = np.clip(relative_pred.disparity, a_min=None, a_max=1.01 * p95)
    # normalize the disparity to 0-1
    clipped_disparity = (clipped_disparity - clipped_disparity.min()) / (
        clipped_disparity.max() - clipped_disparity.min()
    )
    clipped_disparity = (clipped_disparity * 255).astype(np.uint8)
    # log to cam_log_path to avoid backprojecting disparity
    rr.log(f"{cam_log_path}/disparity", rr.DepthImage(clipped_disparity))


def create_depth_comparison_blueprint(
    models: list[BaseRelativePredictor],
) -> rrb.Blueprint:
    model_names: list[str] = [model.__class__.__name__ for model in models]
    blueprint = rrb.Blueprint(
        rrb.Horizontal(
            contents=[
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
            ],
            column_shares=(3, 1, 3, 1),
        ),
        collapse_panels=True,
    )
    return blueprint
