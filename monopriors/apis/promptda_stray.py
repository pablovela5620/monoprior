from monopriors.third_party.promptda.promptda import PromptDA
from monopriors.third_party.promptda.io_wrapper import (
    load_depth,
    load_image,
)
from simplecv.rerun_log_utils import RerunTyroConfig
from dataclasses import dataclass
import rerun as rr
import torch
from torch import Tensor
from jaxtyping import Float


@dataclass
class SanityConfig:
    rr_config: RerunTyroConfig


def sanity_check(config: SanityConfig) -> None:
    DEVICE = "cuda"
    # image_path: str = "data/promptda/example_images/image.jpg"
    # prompt_depth_path: str = "data/promptda/example_images/arkit_depth.png"
    image_path: str = "/home/pablo/0Dev/repos/PromptDA/data/e15294bf65/rgb/000000.jpg"
    prompt_depth_path: str = (
        "/home/pablo/0Dev/repos/PromptDA/data/e15294bf65/depth/000000.png"
    )
    image: Float[Tensor, "B 3 H W"] = load_image(image_path).to(DEVICE)
    prompt_depth: Float[Tensor, "B 1 192 256"] = load_depth(prompt_depth_path).to(
        DEVICE
    )  # 192x256, ARKit LiDAR depth in meters
    print(prompt_depth.shape)
    input()

    model: PromptDA = (
        PromptDA.from_pretrained("depth-anything/prompt-depth-anything-vitl")
        .to(DEVICE)
        .eval()
    )
    depth = model.predict(image, prompt_depth)  # HxW, depth in meters
    rr.log(
        "depth",
        rr.DepthImage(depth, meter=1),
    )
    rr.log(
        "prompt_depth",
        rr.DepthImage(prompt_depth, meter=1),
    )

    image_numpy = image.squeeze().cpu().numpy().transpose(1, 2, 0)

    rr.log(
        "image",
        rr.Image(image_numpy),
    )

    # save_depth(depth, prompt_depth=prompt_depth, image=image)


if __name__ == "__main__":
    sanity_check()
