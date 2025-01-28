import numpy as np
import imageio
import torch
import os
import matplotlib
import cv2


def visualize_depth(
    depth: np.ndarray,
    depth_min=None,
    depth_max=None,
    percentile=2,
    ret_minmax=False,
    cmap="Spectral",
):
    if depth_min is None:
        depth_min = np.percentile(depth, percentile)
    if depth_max is None:
        depth_max = np.percentile(depth, 100 - percentile)
    if depth_min == depth_max:
        depth_min = depth_min - 1e-6
        depth_max = depth_max + 1e-6
    cm = matplotlib.colormaps[cmap]
    depth = ((depth - depth_min) / (depth_max - depth_min)).clip(0, 1)
    img_colored_np = cm(depth[None], bytes=False)[:, :, :, 0:3]  # value from 0 to 1
    img_colored_np = (img_colored_np[0] * 255.0).astype(np.uint8)
    if ret_minmax:
        return img_colored_np, depth_min, depth_max
    else:
        return img_colored_np


def to_tensor_func(arr):
    if arr.ndim == 2:
        arr = arr[:, :, np.newaxis]
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def to_numpy_func(tensor):
    arr = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    if arr.shape[2] == 1:
        arr = arr[:, :, 0]
    return arr


def ensure_multiple_of(x, multiple_of=14):
    return int(x // multiple_of * multiple_of)


def load_image(image_path, to_tensor=True, max_size=1008, multiple_of=14):
    """
    Load image from path and convert to tensor
    max_size // 14 = 0
    """
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.asarray(image).astype(np.float32)
    image = image / 255.0

    max_size = max_size // multiple_of * multiple_of
    if max(image.shape) > max_size:
        h, w = image.shape[:2]
        scale = max_size / max(h, w)
        tar_h = ensure_multiple_of(h * scale)
        tar_w = ensure_multiple_of(w * scale)
        image = cv2.resize(image, (tar_w, tar_h), interpolation=cv2.INTER_AREA)
    if to_tensor:
        return to_tensor_func(image)
    return image


def load_depth(depth_path, to_tensor=True):
    """
    Load depth from path and convert to tensor
    """
    if depth_path.endswith(".png"):
        depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
        depth = np.asarray(depth).astype(np.float32)
        # depth = np.asarray(imageio.imread(depth_path)).astype(np.float32)
        depth = depth / 1000.0
    elif depth_path.endswith(".npz"):
        depth = np.load(depth_path)["depth"]
    else:
        raise ValueError(f"Unsupported depth format: {depth_path}")
    if to_tensor:
        return to_tensor_func(depth)
    return depth


def save_depth(
    depth,
    prompt_depth=None,
    image=None,
    output_path="results/example_depth.png",
    save_vis=True,
):
    """
    Save depth to path
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    depth = to_numpy_func(depth)
    uint16_depth = (depth * 1000.0).astype(np.uint16)
    imageio.imwrite(output_path, uint16_depth)
    print(f"Saved depth to {output_path}")

    if not save_vis:
        return
    output_path_ = output_path
    output_path = output_path_.replace(".png", "_depth.jpg")
    depth_vis, depth_min, depth_max = visualize_depth(depth, ret_minmax=True)
    imageio.imwrite(output_path, depth_vis)

    if prompt_depth is not None:
        prompt_depth = to_numpy_func(prompt_depth)
        output_path = output_path_.replace(".png", "_prompt_depth.jpg")
        prompt_depth_vis = visualize_depth(
            prompt_depth, depth_min=depth_min, depth_max=depth_max
        )
        imageio.imwrite(output_path, prompt_depth_vis)

    if image is not None:
        output_path = output_path_.replace(".png", "_image.jpg")
        image = to_numpy_func(image)
        imageio.imwrite(output_path, (image * 255).astype(np.uint8))
