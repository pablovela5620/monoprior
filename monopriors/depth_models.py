from typing import Literal
import torch
import numpy as np
from jaxtyping import Float, UInt8


class UniDepthPredictor:
    def __init__(self, device: Literal["cpu", "cuda"]):
        self.model = torch.hub.load(
            "lpiccinelli-eth/UniDepth",
            "UniDepth",
            version="v1",
            backbone="ConvNextL",
            pretrained=True,
            trust_repo=True,
        ).to(device)

    def __call__(
        self, rgb: UInt8[np.ndarray, "h w 3"], K_33: Float[np.ndarray, "3 3"] | None
    ) -> Float[torch.Tensor, "b 1 h w"]:
        # Load the RGB image and the normalization will be taken care of by the model
        rgb = torch.from_numpy(rgb).permute(2, 0, 1)  # C, H, W

        if K_33 is None:
            predictions = self.model.infer(rgb)
        else:
            K_33 = torch.from_numpy(K_33)
            predictions = self.model.infer(rgb, K_33)

        depth_b1hw: Float[torch.Tensor, "b 1 h w"] = predictions["depth"]
        K_b33: Float[torch.Tensor, "b 3 3"] = predictions["intrinsics"]
        return depth_b1hw, K_b33
