from abc import ABC, abstractmethod
from typing import Literal
import torch
import numpy as np
from dataclasses import dataclass
from jaxtyping import Float, UInt8
from torchvision import transforms
import torch.nn.functional as F
from monopriors.dsine.dsine import DSINE
from monopriors.dsine.utils.utils import get_intrins_from_fov, pad_input


def _load_state_dict(local_file_path: str | None = None):
    if local_file_path is not None and os.path.exists(local_file_path):
        # Load state_dict from local file
        state_dict = torch.load(local_file_path, map_location=torch.device("cpu"))
    else:
        # Load state_dict from the default URL
        file_name = "dsine.pt"
        url = f"https://huggingface.co/camenduru/DSINE/resolve/main/dsine.pt"
        state_dict = torch.hub.load_state_dict_from_url(
            url, file_name=file_name, map_location=torch.device("cpu")
        )

    return state_dict["model"]


@dataclass
class MonoPriorPrediction:
    depth_b1hw: Float[torch.Tensor, "b 1 h w"]
    normal_b3hw: Float[torch.Tensor, "b 3 h w"]
    K_b33: Float[torch.Tensor, "b 3 3"] | None = None

    def to_numpy(self) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        depth_np_bhw1 = self.depth_b1hw.permute(0, 2, 3, 1).numpy(force=True)
        normal_np_bhw3 = self.normal_b3hw.permute(0, 2, 3, 1).numpy(force=True)
        K_np_b33 = self.K_b33.numpy(force=True) if self.K_b33 is not None else None
        return depth_np_bhw1, normal_np_bhw3, K_np_b33


class MonoPriorModel(ABC):
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @abstractmethod
    def __call__(self, rgb, K_33) -> MonoPriorPrediction:
        pass


class DSinePredictor:
    def __init__(self, device: Literal["cpu", "cuda"]):
        self.device = device
        self.transform = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        self.model = self.load_model()

    def load_model(self):
        state_dict = _load_state_dict(None)
        model = DSINE()
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        model = model.to(self.device)
        model.pixel_coords = model.pixel_coords.to(self.device)

        return model

    def __call__(
        self, rgb: UInt8[np.ndarray, "h w 3"], K_33: Float[np.ndarray, "3 3"] | None
    ) -> Float[torch.Tensor, "b 1 h w"]:
        rgb = rgb.astype(np.float32) / 255.0
        rgb = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(self.device)
        _, _, h, w = rgb.shape

        # zero-pad the input image so that both the width and height are multiples of 32
        left, right, top, bottom = pad_input(h, w)
        rgb = F.pad(rgb, (left, right, top, bottom), mode="constant", value=0.0)
        rgb = self.transform(rgb)

        if K_33 is None:
            K_b33: Float[torch.Tensor, "b 3 3"] = get_intrins_from_fov(
                new_fov=60.0, H=h, W=w, device=self.device
            ).unsqueeze(0)
        else:
            K_b33 = torch.from_numpy(K_33).unsqueeze(0).to(self.device)

        # add padding to intrinsics
        K_b33[:, 0, 2] += left
        K_b33[:, 1, 2] += top

        with torch.no_grad():
            normal_b3hw: Float[torch.Tensor, "b 3 h-t w-l"] = self.model(
                rgb, intrins=K_b33
            )[-1]
            normal_b3hw: Float[torch.Tensor, "b 3 h w"] = normal_b3hw[
                :, :, top : top + h, left : left + w
            ]

        return normal_b3hw


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


class DsineAndUnidepth(MonoPriorModel):
    def __init__(self) -> None:
        super().__init__()
        self.depth_model = UniDepthPredictor(device=self.device)
        self.surface_model = DSinePredictor(device=self.device)

    def __call__(
        self,
        rgb: UInt8[np.ndarray, "h w 3"],
        K_33: Float[np.ndarray, "3 3"] | None = None,
    ) -> MonoPriorPrediction:
        depth_b1hw, K_b33 = self.depth_model(rgb, K_33)
        # use depth_model intrinsics for surface_model if not provided
        if K_33 is None:
            K_33 = K_b33[0].numpy(force=True)
        normal_b3hw = self.surface_model(rgb, K_33)
        prediction = MonoPriorPrediction(depth_b1hw=depth_b1hw, normal_b3hw=normal_b3hw)
        return prediction
