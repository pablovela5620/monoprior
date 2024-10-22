from abc import ABC, abstractmethod
from dataclasses import dataclass
from jaxtyping import Float32, UInt8
from typing import Literal
import numpy as np


@dataclass
class RelativeDepthPrediction:
    disparity: Float32[np.ndarray, "h w"]
    # relative disparity
    depth: Float32[np.ndarray, "h w"]
    # relative depth
    confidence: Float32[np.ndarray, "h w"]
    # confidence values
    K_33: Float32[np.ndarray, "3 3"]
    # intrinsics


class BaseRelativePredictor(ABC):
    @abstractmethod
    def __call__(
        self, rgb: UInt8[np.ndarray, "h w 3"], K_33: Float32[np.ndarray, "3 3"] | None
    ) -> RelativeDepthPrediction:
        raise NotImplementedError

    def set_model_device(self, device: Literal["cpu", "cuda"] = "cuda") -> None:
        self.model.to(device)
