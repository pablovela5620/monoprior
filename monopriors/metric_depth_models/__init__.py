from .base_metric_depth import MetricDepthPrediction, BaseMetricPredictor
from .unidepth import UniDepthMetricPredictor
from .metric3d import Metric3DPredictor
from typing import Literal, get_args, Callable

# Define predictor names as a list of strings
METRIC_PREDICTORS = Literal["UniDepthMetricPredictor", "Metric3DPredictor"]

# Use the list to generate the __all__ list
__all__: list[str] = list(get_args(METRIC_PREDICTORS)) + [
    "MetricDepthPrediction",
]


def get_metric_predictor(
    predictor_type: METRIC_PREDICTORS,
) -> Callable[..., BaseMetricPredictor]:
    match predictor_type:
        case "UniDepthMetricPredictor":
            return UniDepthMetricPredictor
        case "Metric3DPredictor":
            return Metric3DPredictor
        case _:
            raise ValueError(f"Unknown predictor type: {predictor_type}")
