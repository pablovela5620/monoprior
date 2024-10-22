from pathlib import Path
from argparse import ArgumentParser
import rerun as rr
import mmcv
import cv2
import numpy as np
from typing import get_args

from monopriors.relative_depth_models import (
    get_relative_predictor,
    RelativeDepthPrediction,
    RELATIVE_PREDICTORS,
)
from monopriors.relative_depth_models.base_relative_depth import BaseRelativePredictor
from monopriors.rr_logging_utils import log_relative_pred


def resize_image(image: np.ndarray, max_dim: int = 1024) -> np.ndarray:
    current_dim = max(image.shape[0], image.shape[1])
    if current_dim > max_dim:
        scale_factor = max_dim / current_dim
        image = mmcv.imrescale(img=image, scale=scale_factor)
    return image


def relative_depth_from_img(
    image_path: Path, depth_predictor_name: RELATIVE_PREDICTORS
) -> None:
    parent_log_path = Path("world")
    bgr_hw3 = cv2.imread(str(image_path))
    rgb_hw3 = cv2.cvtColor(bgr_hw3, cv2.COLOR_BGR2RGB)

    max_dim = 1024 // 2
    rgb_hw3 = resize_image(rgb_hw3, max_dim)

    predictor: BaseRelativePredictor = get_relative_predictor(depth_predictor_name)(
        device="cuda"
    )
    relative_pred: RelativeDepthPrediction = predictor.__call__(rgb=rgb_hw3, K_33=None)
    rr.log(f"{parent_log_path}", rr.ViewCoordinates.RDF, timeless=True)
    log_relative_pred(parent_log_path, relative_pred, rgb_hw3)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--image-path", type=Path, default="data/39703866203.jpg")
    parser.add_argument(
        "--depth-predictor-name",
        choices=get_args(RELATIVE_PREDICTORS),
        default="DepthAnythingV2Predictor",
    )
    rr.script_add_args(parser)
    args = parser.parse_args()
    rr.script_setup(args, "depth view")
    relative_depth_from_img(args.image_path, args.depth_predictor_name)
    rr.script_teardown(args)
