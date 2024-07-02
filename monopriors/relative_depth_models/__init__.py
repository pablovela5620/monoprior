from .base_relative_depth import RelativeDepthPrediction, BaseRelativePredictor
from .depth_anything_v2 import DepthAnythingV2Predictor
from .depth_anything_v1 import DepthAnythingV1Predictor
from .unidepth import UniDepthRelativePredictor
from .metric3d_relative import Metric3DRelativePredictor
from typing import Literal, get_args, Callable

# Define predictor names as a list of strings
RELATIVE_PREDICTORS = Literal[
    "DepthAnythingV1Predictor",
    "DepthAnythingV2Predictor",
    "UniDepthRelativePredictor",
    "Metric3DRelativePredictor",
]

# Use the list to generate the __all__ list
__all__: list[str] = list(get_args(RELATIVE_PREDICTORS)) + [
    "RelativeDepthPrediction",
]


def get_relative_predictor(
    predictor_type: RELATIVE_PREDICTORS,
) -> Callable[..., BaseRelativePredictor]:
    match predictor_type:
        case "UniDepthRelativePredictor":
            return UniDepthRelativePredictor
        case "DepthAnythingV2Predictor":
            return DepthAnythingV2Predictor
        case "DepthAnythingV1Predictor":
            return DepthAnythingV1Predictor
        case "Metric3DRelativePredictor":
            return Metric3DRelativePredictor
        case _:
            raise ValueError(f"Unknown predictor type: {predictor_type}")
