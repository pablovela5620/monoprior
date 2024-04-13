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
from monopriors.dsine.utils.utils import get_intrins_from_fov, pad_input
from omnidata_tools.torch.modules.midas.dpt_depth import DPTDepthModel


class DSineNormalPredictor:
    def __init__(self, device: Literal["cpu", "cuda"]):
        self.device = device
        self.transform = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        self.model = self.load_model()

    def load_model(self):
        state_dict = self._load_state_dict(None)
        model = DSINE()
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        model = model.to(self.device)
        model.pixel_coords = model.pixel_coords.to(self.device)

        return model

    def _load_state_dict(self, local_file_path: str | None = None):
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

    def __call__(
        self, rgb: UInt8[np.ndarray, "h w 3"], K_33: Float[np.ndarray, "3 3"] | None
    ) -> Float[torch.Tensor, "b 1 h w"]:
        # preprocess the input image
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
            # undo padding
            normal_b3hw: Float[torch.Tensor, "b 3 h w"] = normal_b3hw[
                :, :, top : top + h, left : left + w
            ]

        return normal_b3hw


class OmniNormalPredictor:
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
    ) -> Float[torch.Tensor, "b 1 h w"]:
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
        return normal_b3hw
