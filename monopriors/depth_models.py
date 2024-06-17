from typing import Literal
import torch
import numpy as np
from jaxtyping import Float, UInt8
from timeit import default_timer as timer
from huggingface_hub import hf_hub_download
from monopriors.depth_anything_v2.dpt import DepthAnythingV2
from icecream import ic


class UniDepthPredictor:
    def __init__(
        self,
        device: Literal["cpu", "cuda"],
        version: Literal["v1", "v2"] = "v2",
        backbone: Literal["vits14", "vitl14"] = "vitl14",
    ):
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
        self,
        rgb: UInt8[np.ndarray, "h w 3"],  # noqa: F722
        K_33: Float[np.ndarray, "3 3"] | None,  # noqa: F722
    ) -> tuple[
        Float[torch.Tensor, "b 1 h w"],  # noqa: F722
        Float[torch.Tensor, "b 3 3"],  # noqa: F722
        Float[torch.Tensor, "b 1 h w"],  # noqa: F722
    ]:
        # Load the RGB image and the normalization will be taken care of by the model
        rgb = torch.from_numpy(rgb).permute(2, 0, 1)  # C, H, W

        if K_33 is None:
            predictions = self.model.infer(rgb)
        else:
            K_33 = torch.from_numpy(K_33)
            predictions = self.model.infer(rgb, K_33)

        depth_b1hw: Float[torch.Tensor, "b 1 h w"] = predictions["depth"]  # noqa: F722
        K_b33: Float[torch.Tensor, "b 3 3"] = predictions["intrinsics"]  # noqa: F722
        conf_b1hw: Float[torch.Tensor, "b 1 h w"] = predictions["confidence"]  # noqa: F722
        # normalize the confidence to 0-1
        conf_b1hw = (conf_b1hw - conf_b1hw.min()) / (conf_b1hw.max() - conf_b1hw.min())
        return depth_b1hw, K_b33, conf_b1hw


model_configs = {
    "vits": {
        "encoder": "vits",
        "features": 64,
        "out_channels": [48, 96, 192, 384],
    },
    "vitb": {
        "encoder": "vitb",
        "features": 128,
        "out_channels": [96, 192, 384, 768],
    },
    "vitl": {
        "encoder": "vitl",
        "features": 256,
        "out_channels": [256, 512, 1024, 1024],
    },
    "vitg": {
        "encoder": "vitg",
        "features": 384,
        "out_channels": [1536, 1536, 1536, 1536],
    },
}
encoder2name = {
    "vits": "Small",
    "vitb": "Base",
    "vitl": "Large",
    "vitg": "Giant",
}


class DepthAnythingPredictor:
    def __init__(
        self,
        device: Literal["cpu", "cuda"],
        encoder: Literal["vits", "vitb", "vitl"] = "vitl",
    ):
        print("Loading DepthAnything model...")
        start = timer()
        model_name: str = encoder2name[encoder]
        self.model = DepthAnythingV2(**model_configs[encoder])
        filepath: str = hf_hub_download(
            repo_id=f"depth-anything/Depth-Anything-V2-{model_name}",
            filename=f"depth_anything_v2_{encoder}.pth",
            repo_type="model",
        )
        state_dict = torch.load(filepath, map_location="cpu")
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(device).eval()
        print(f"DepthAnything model loaded. Time: {timer() - start:.2f}s")

    def __call__(
        self, rgb: UInt8[np.ndarray, "h w 3"], K_33: Float[np.ndarray, "3 3"] | None
    ) -> Float[torch.Tensor, "b 1 h w"]:  # noqa: F722
        predictions = self.model.infer_image(rgb)

        return predictions
