from abc import ABC, abstractmethod
from dataclasses import dataclass
from jaxtyping import Float, UInt8
from typing import Literal
import numpy as np


@dataclass
class SurfaceNormalPrediction:
    normal_hw3: Float[np.ndarray, "h w 3"]
    # surface normal prediction
    confidence_hw1: Float[np.ndarray, "h w 1"]
    # confidence values


class BaseNormalPredictor(ABC):
    @abstractmethod
    def __call__(
        self, rgb: UInt8[np.ndarray, "h w 3"], K_33: Float[np.ndarray, "3 3"] | None
    ) -> SurfaceNormalPrediction:
        raise NotImplementedError

    def set_model_device(self, device: Literal["cpu", "cuda"] = "cuda") -> None:
        self.model.to(device)
