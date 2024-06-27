from abc import ABC, abstractmethod
from typing import Literal
import torch
import numpy as np
from dataclasses import dataclass
from jaxtyping import Float, UInt8
from monopriors.normal_models import DSineNormalPredictor, OmniNormalPredictor
from monopriors.depth_models import UniDepthPredictor
from icecream import ic
from einops import rearrange


@dataclass
class MonoPriorPrediction:
    depth_b1hw: Float[torch.Tensor, "b 1 h w"]
    normal_b3hw: Float[torch.Tensor, "b 3 h w"]
    K_b33: Float[torch.Tensor, "b 3 3"] | None = None
    depth_conf_b1hw: Float[torch.Tensor, "b 1 h w"] | None = None
    normal_conf_b1hw: Float[torch.Tensor, "b 1 h w"] | None = None

    def to_numpy(
        self,
    ) -> tuple[
        Float[np.ndarray, "b h w 1"],
        Float[np.ndarray, "b h w 3"],
        Float[np.ndarray, "b 3 3"] | None,
        Float[np.ndarray, "b h w 1"] | None,
        Float[np.ndarray, "b h w 1"] | None,
    ]:
        depth_np_bhw1 = rearrange(self.depth_b1hw, "b c h w -> b h w c").numpy(
            force=True
        )
        normal_np_bhw3 = rearrange(self.normal_b3hw, "b c h w -> b h w c").numpy(
            force=True
        )
        K_np_b33 = self.K_b33.numpy(force=True) if self.K_b33 is not None else None
        depth_conf_np_bhw1 = (
            rearrange(self.depth_conf_b1hw, "b c h w -> b h w c").numpy(force=True)
            if self.depth_conf_b1hw is not None
            else None
        )
        normal_conf_np_bhw1 = (
            rearrange(self.normal_conf_b1hw, "b c h w -> b h w c").numpy(force=True)
            if self.normal_conf_b1hw is not None
            else None
        )
        return (
            depth_np_bhw1,
            normal_np_bhw3,
            K_np_b33,
            depth_conf_np_bhw1,
            normal_conf_np_bhw1,
        )


class MonoPriorModel(ABC):
    def __init__(self) -> None:
        self.device: Literal["cuda", "cpu"] = (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    @abstractmethod
    def __call__(self, rgb, K_33) -> MonoPriorPrediction:
        pass


class DsineAndUnidepth(MonoPriorModel):
    def __init__(self) -> None:
        super().__init__()
        self.depth_model = UniDepthPredictor(device=self.device)
        self.surface_model = DSineNormalPredictor(device=self.device)

    def __call__(
        self,
        rgb: UInt8[np.ndarray, "h w 3"],
        K_33: Float[np.ndarray, "3 3"] | None = None,
    ) -> MonoPriorPrediction:
        depth_b1hw: Float[torch.Tensor, "b 1 h w"]
        K_b33: Float[torch.Tensor, "b 3 3"]

        depth_b1hw, K_b33, depth_conf_b1hw = self.depth_model(rgb, K_33)
        # use depth_model intrinsics for surface_model if not provided
        if K_33 is None:
            K_33 = rearrange(K_b33, "b r c -> r c").numpy(force=True)

        normal_b3hw, normal_conf_b1hw = self.surface_model(rgb, K_33)
        prediction = MonoPriorPrediction(
            depth_b1hw=depth_b1hw,
            normal_b3hw=normal_b3hw,
            K_b33=K_b33,
            depth_conf_b1hw=depth_conf_b1hw,
            normal_conf_b1hw=normal_conf_b1hw,
        )
        return prediction
