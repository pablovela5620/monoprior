from pathlib import Path
from argparse import ArgumentParser
import rerun as rr
import cv2
from typing import get_args

# from monopriors.depth_models import UniDepthPredictor
from monopriors.relative_depth_models import (
    get_relative_predictor,
    RelativeDepthPrediction,
    RELATIVE_PREDICTORS,
)
from monopriors.relative_depth_models.base_relative_depth import BaseRelativePredictor
from monopriors.rr_logging_utils import log_relative_pred


def relative_depth_from_img(
    image_path: Path, depth_predictor_name: RELATIVE_PREDICTORS
) -> None:
    parent_log_path = Path("world")
    bgr_hw3 = cv2.imread(str(image_path))
    rgb_hw3 = cv2.cvtColor(bgr_hw3, cv2.COLOR_BGR2RGB)

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
