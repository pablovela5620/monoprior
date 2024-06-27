from pathlib import Path
import os
from typing import Literal
import torch
import numpy as np
from jaxtyping import Float, UInt8
from torchvision import transforms
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from monopriors.dsine.dsine import DSINE
from monopriors.dsine.dsine_kappa import DSINE_v02_kappa
from monopriors.dsine.utils.utils import get_intrins_from_fov, pad_input
from omnidata_tools.torch.modules.midas.dpt_depth import DPTDepthModel
from abc import ABC, abstractmethod
from einops import rearrange


class NormalPredictor(ABC):
    @abstractmethod
    def __call__(
        self, rgb: UInt8[np.ndarray, "h w 3"], K_33: Float[np.ndarray, "3 3"] | None
    ) -> tuple[Float[torch.Tensor, "b 3 h w"], Float[torch.Tensor, "b 1 h w"] | None]:
        raise NotImplementedError


class DSineNormalPredictor(NormalPredictor):
    def __init__(
        self,
        device: Literal["cpu", "cuda"],
        model_type: Literal["dsine", "dsine_kappa"] = "dsine_kappa",
    ):
        self.device = device
        self.transform = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        self.model = self.load_model(model_type=model_type)

    def load_model(self, model_type: Literal["dsine", "dsine_kappa"]):
        state_dict = self._load_state_dict(
            local_file_path="checkpoints/dsine_kappa.pt", model_type=model_type
        )
        if model_type == "dsine":
            model = DSINE()
        else:
            model = DSINE_v02_kappa()

        model.load_state_dict(state_dict, strict=True)
        model.eval()
        model = model.to(self.device)
        model.pixel_coords = model.pixel_coords.to(self.device)

        return model

    def _load_state_dict(
        self,
        local_file_path: str | None = None,
        model_type: Literal["dsine", "dsine_kappa"] = "dsine_kappa",
    ):
        if local_file_path is not None and os.path.exists(local_file_path):
            # Load state_dict from local file
            state_dict = torch.load(local_file_path, map_location=torch.device("cpu"))
        else:
            # Load state_dict from the default URL
            file_name = "dsine.pt" if model_type == "dsine" else "dsine_kappa.pt"
            url = (
                f"https://huggingface.co/camenduru/DSINE/resolve/main/dsine.pt"
                if model_type == "dsine"
                else f"https://huggingface.co/pablovela5620/dsine_kappa/resolve/main/dsine_kappa.pt"
            )
            state_dict = torch.hub.load_state_dict_from_url(
                url, file_name=file_name, map_location=torch.device("cpu")
            )

        return state_dict["model"]

    def __call__(
        self, rgb: UInt8[np.ndarray, "h w 3"], K_33: Float[np.ndarray, "3 3"] | None
    ) -> tuple[Float[torch.Tensor, "b 3 h w"], Float[torch.Tensor, "b 1 h w"] | None]:
        # preprocess the input image
        rgb: Float[np.ndarray, "h w 3"] = rgb.astype(np.float32) / 255.0
        rgb: Float[torch.Tensor, "1 c h w"] = torch.from_numpy(
            rearrange(rgb, "h w c -> 1 c h w")
        ).to(self.device)

        _, _, h, w = rgb.shape

        # zero-pad the input image so that both the width and height are multiples of 32
        left, right, top, bottom = pad_input(h, w)
        rgb = F.pad(rgb, (left, right, top, bottom), mode="constant", value=0.0)
        rgb = self.transform(rgb)

        if K_33 is None:
            K_33: Float[torch.Tensor, "3 3"] = get_intrins_from_fov(
                new_fov=60.0, H=h, W=w, device=self.device
            )
            K_b33 = rearrange(K_33, "r c -> 1 r c")
        else:
            K_b33 = torch.from_numpy(rearrange(K_33, "r c -> 1 r c")).to(self.device)

        # add padding to intrinsics
        K_b33[:, 0, 2] += left
        K_b33[:, 1, 2] += top

        with torch.no_grad():
            normal_list: list[torch.Tensor] = self.model(rgb, intrins=K_b33)
            # last value in the list is the normal map
            normal_bchw: (
                Float[torch.Tensor, "b 3 _ _"] | Float[torch.Tensor, "b 4 _ _"]
            ) = normal_list[-1]
            # undo padding
            normal_bchw: (
                Float[torch.Tensor, "b 3 _ _"] | Float[torch.Tensor, "b 4 _ _"]
            ) = normal_bchw[:, :, top : top + h, left : left + w]

            normal_b3hw: Float[torch.Tensor, "b 3 h w"]
            conf_b1hw: Float[torch.Tensor, "b 1 h w"] | None

            normal_b3hw, conf_b1hw = normal_bchw[:, :3, :, :], normal_bchw[:, 3:, :, :]
            conf_b1hw = None if conf_b1hw.size(1) == 0 else conf_b1hw

        return normal_b3hw, conf_b1hw


class OmniNormalPredictor(NormalPredictor):
    def __init__(
        self,
        device: Literal["cpu", "cuda"],
        omnidata_pretrained_weights_path: Path = Path(
            "/home/pablo/0Dev/personal/forked-repos/dn-splatter/omnidata_ckpt"
        ),
    ):
        self.device = device
        self.model = self._load_model(omnidata_pretrained_weights_path)
        self.image_size = 384

    def _load_model(self, omnidata_pretrained_weights_path: Path):
        model = DPTDepthModel(backbone="vitb_rn50_384", num_channels=3)  # DPT Hybrid
        omnidata_pretrained_weights_path = (
            omnidata_pretrained_weights_path / "omnidata_dpt_normal_v2.ckpt"
        )
        map_location = (
            (lambda storage, loc: storage.cuda())
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        checkpoint = torch.load(
            omnidata_pretrained_weights_path, map_location=map_location
        )

        if "state_dict" in checkpoint:
            state_dict = {}
            for k, v in checkpoint["state_dict"].items():
                state_dict[k[6:]] = v
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict)
        model.to(self.device)
        return model

    def __call__(
        self, rgb: UInt8[np.ndarray, "h w 3"], K_33: Float[np.ndarray, "3 3"] | None
    ) -> tuple[Float[torch.Tensor, "b 3 h w"], Float[torch.Tensor, "b 1 h w"] | None]:
        rgb = rgb.astype(np.float32) / 255.0
        img_3hw = torch.from_numpy(rgb).permute(2, 0, 1)
        _, H, W = img_3hw.shape
        # omnidata normal model expects 384x384 images
        img_3hw = TF.resize(img_3hw, (self.image_size, self.image_size), antialias=None)
        img_b3hw = img_3hw.unsqueeze(0).to(self.device)

        # return normal_b3hw
        normal_b3hw = self.model(img_b3hw).clamp(min=0, max=1)
        normal_3hw = normal_b3hw.squeeze(0)
        # reshape back to original size
        if normal_3hw.shape[1] != H and normal_3hw.shape[2] != W:
            normal_3hw = TF.resize(normal_3hw, (H, W), antialias=None)

        normal_b3hw = normal_3hw.unsqueeze(0)
        # generates normal that is between 0-1 in the OpenCV Format (RDF)
        # no confidence map is returned from omnidata model
        return normal_b3hw, None
