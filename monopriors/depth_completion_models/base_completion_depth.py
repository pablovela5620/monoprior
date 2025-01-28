from abc import ABC, abstractmethod
from dataclasses import dataclass
from jaxtyping import Float, UInt8
from typing import Literal
import numpy as np


@dataclass
class CompletionDepthPrediction:
    depth_meters: Float[np.ndarray, "h w"]
    # metric depth in meters
    confidence: Float[np.ndarray, "h w"]
    # confidence values
    K_33: Float[np.ndarray, "3 3"]
    # intrinsics


class BaseCompletionPredictor(ABC):
    @abstractmethod
    def __call__(
        self, rgb: UInt8[np.ndarray, "h w 3"], K_33: Float[np.ndarray, "3 3"] | None
    ) -> CompletionDepthPrediction:
        raise NotImplementedError

    def set_model_device(self, device: Literal["cpu", "cuda"] = "cuda") -> None:
        self.model.to(device)
