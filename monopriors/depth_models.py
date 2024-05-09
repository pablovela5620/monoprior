from typing import Literal
import torch
import numpy as np
from jaxtyping import Float, UInt8
from timeit import default_timer as timer


class UniDepthPredictor:
    def __init__(
        self,
        device: Literal["cpu", "cuda"],
        version: Literal["v1", "v2"] = "v2",
        backbone: Literal["cnvnxtl", "vitl14"] = "vitl14",
    ):
        self.k_output_key: str = "intrinsics" if version == "v1" else "K"
        if version == "v2":
            assert backbone == "vitl14", "Only vitl14 backbone is supported for v2"

        print("Loading UniDepth model...")
        start = timer()
        self.model = torch.hub.load(
            "lpiccinelli-eth/UniDepth",
            "UniDepth",
            version=version,
            backbone=backbone,
            pretrained=True,
            trust_repo=True,
        ).to(device)
        print(f"UniDepth model loaded. Time: {timer() - start:.2f}s")

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
        K_b33: Float[torch.Tensor, "b 3 3"] = predictions[self.k_output_key]
        return depth_b1hw, K_b33
