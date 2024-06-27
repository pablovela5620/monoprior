from .base_relative_depth import RelativeDepthPrediction, BaseRelativePredictor
from .depth_anything_v2 import DepthAnythingV2Predictor
from .unidepth import UniDepthPredictor
from .metric3d_relative import Metric3DPredictor
from typing import Literal, get_args, Callable

# Define predictor names as a list of strings
RELATIVE_PREDICTORS = Literal[
    "UniDepthPredictor", "DepthAnythingV2Predictor", "Metric3DPredictor"
]

# Use the list to generate the __all__ list
__all__: list[str] = list(get_args(RELATIVE_PREDICTORS)) + [
    "RelativeDepthPrediction",
]


def get_predictor(
    predictor_type: RELATIVE_PREDICTORS,
) -> Callable[..., BaseRelativePredictor]:
    match predictor_type:
        case "UniDepthPredictor":
            return UniDepthPredictor
        case "DepthAnythingV2Predictor":
            return DepthAnythingV2Predictor
        case "Metric3DPredictor":
            return Metric3DPredictor
        case _:
            raise ValueError(f"Unknown predictor type: {predictor_type}")
