from dataclasses import dataclass
from timeit import default_timer as timer
from typing import Literal

import cv2
import numpy as np
import open3d as o3d
import torch
from einops import rearrange
from jaxtyping import Float32, UInt8, UInt16
from numpy import ndarray
from PIL import Image
from serde import field as serde_field
from serde import from_dict, serde
from simplecv.camera_parameters import Extrinsics, Intrinsics, PinholeParameters, rescale_intri
from torch import Tensor
from torchvision import transforms as TF
from vggt.models.vggt import VGGT
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.pose_enc import pose_encoding_to_extri_intri


@serde(deny_unknown_fields=True)
class VGGTPredictions:
    pose_enc: UInt8[ndarray, "*batch num_cams 9"]
    depth: Float32[ndarray, "*batch num_cams H W 1"]
    depth_conf: Float32[ndarray, "*batch num_cams H W"]
    world_points: Float32[ndarray, "*batch num_cams H W 3"]
    world_points_conf: Float32[ndarray, "*batch num_cams H W"]
    images: Float32[ndarray, "*batch num_cams 3 H W"]
    intrinsic: Float32[ndarray, "*batch num_cams 3 3"]
    cam_T_world: Float32[ndarray, "*batch num_cams 3 4"] = serde_field(rename="extrinsic")

    def remove_batch_dim_if_one(self) -> "VGGTPredictions":
        """
        Removes the batch dimension from all arrays if batch size is 1.

        Returns:
            VGGTPredictions: A new instance with batch dimension removed if batch=1
        """
        if self.pose_enc.shape[0] != 1:
            return self

        result = VGGTPredictions(
            pose_enc=self.pose_enc.squeeze(0),
            depth=self.depth.squeeze(0),
            depth_conf=self.depth_conf.squeeze(0),
            world_points=self.world_points.squeeze(0),
            world_points_conf=self.world_points_conf.squeeze(0),
            images=self.images.squeeze(0),
            cam_T_world=self.cam_T_world.squeeze(0),
            intrinsic=self.intrinsic.squeeze(0),
        )
        return result


def preprocess_images(
    rgb_list: list[UInt8[ndarray, "H W 3"]],
    mode: Literal["crop", "pad"] = "crop",
) -> Float32[torch.Tensor, "N 3 H W"]:
    """
    A quick start function to preprocess images for model input.

    Args:
        rgb_list (list): List of RGB images as numpy arrays

    Returns:
        torch.Tensor: Batched tensor of preprocessed images with shape (N, 3, H, W)

    Raises:
        ValueError: If the input list is empty

    Notes:
        - Images with different dimensions will be padded with white (value=1.0)
        - A warning is printed when images have different shapes
        - The function ensures width=518px while maintaining aspect ratio
        - Height is adjusted to be divisible by 14 for compatibility with model requirements
    """
    # Check for empty list
    if len(rgb_list) == 0:
        raise ValueError("At least 1 image is required")

    # Disable pad mode for now
    if mode == "pad":
        raise NotImplementedError(
            "Pad mode is currently not fully supported due to issues with post-processing padded outputs. "
            "Please use 'crop' mode instead."
        )

    images = []
    shapes = set()
    to_tensor = TF.ToTensor()
    target_size = 518

    # First process all images and collect their shapes
    for rgb in rgb_list:
        # Convert the numpy array to PIL Image to ensure identical processing
        pil_img = Image.fromarray(rgb)

        width, height = pil_img.size
        if mode == "pad":
            # Make the largest dimension 518px while maintaining aspect ratio
            if width >= height:
                new_width = target_size
                new_height: int = round(height * (new_width / width) / 14) * 14  # Make divisible by 14
            else:
                new_height = target_size
                new_width: int = round(width * (new_height / height) / 14) * 14  # Make divisible by 14
        else:  # mode == "crop"
            # Original behavior: set width to 518px
            new_width = target_size
            # Calculate height maintaining aspect ratio, divisible by 14
            new_height = round(height * (new_width / width) / 14) * 14

        # Resize with new dimensions using PIL's BICUBIC for exact matching
        pil_img = pil_img.resize((new_width, new_height), Image.BICUBIC)

        # Convert to tensor using the same to_tensor transform
        img = to_tensor(pil_img)  # Convert to tensor (0, 1)

        # Center crop height if it's larger than 518 (only in crop mode)
        if mode == "crop" and new_height > target_size:
            start_y = (new_height - target_size) // 2
            img = img[:, start_y : start_y + target_size, :]

        # For pad mode, pad to make a square of target_size x target_size
        if mode == "pad":
            h_padding = target_size - img.shape[1]
            w_padding = target_size - img.shape[2]

            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left

                # Pad with white (value=1.0)
                img = torch.nn.functional.pad(
                    img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
                )

        shapes.add((img.shape[1], img.shape[2]))
        images.append(img)

    # Check if we have different shapes
    if len(shapes) > 1:
        print(f"Warning: Found images with different shapes: {shapes}")
        # Find maximum dimensions
        max_height = max(shape[0] for shape in shapes)
        max_width = max(shape[1] for shape in shapes)

        # Pad images if necessary
        padded_images = []
        for img in images:
            h_padding = max_height - img.shape[1]
            w_padding = max_width - img.shape[2]

            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left

                img = torch.nn.functional.pad(
                    img,
                    (pad_left, pad_right, pad_top, pad_bottom),
                    mode="constant",
                    value=1.0,
                )
            padded_images.append(img)
        images = padded_images

    images = torch.stack(images)  # concatenate images

    # Ensure correct shape when single image is (1, C, H, W)
    if len(rgb_list) == 1 and images.dim() == 3:
        images = images.unsqueeze(0)

    return images


def remove_padding_from_prediction(
    padded_rgb: np.ndarray,
    original_size: tuple[int, int],  # (width, height)
    target_size: int = 518,
) -> np.ndarray:
    """
    Crop out the padding from a prediction that was produced in pad mode.

    Args:
        pred (torch.Tensor): The prediction tensor with shape (..., target_size, target_size).
        original_size (tuple[int, int]): Original (width, height) of the image.
        target_size (int): The size used during preprocessing (default 518).

    Returns:
        torch.Tensor: The cropped tensor corresponding to the resized image.
    """
    width, height = original_size
    # Compute new dimensions using the same logic as in preprocessing:
    if width >= height:
        new_width = target_size
        new_height = round(height * (target_size / width) / 14) * 14
    else:
        new_height = target_size
        new_width = round(width * (target_size / height) / 14) * 14

    # Determine the padding applied during preprocessing
    pad_top = (target_size - new_height) // 2
    pad_left = (target_size - new_width) // 2

    # Crop out the padded borders
    cropped_rgb = padded_rgb[..., pad_top : pad_top + new_height, pad_left : pad_left + new_width]
    return cropped_rgb


@dataclass
class MultiviewPred:
    """
    Multiview Consistent Depth Prediction.

    Attributes:
        cam_name (str): Name of the camera.
        rgb_image (UInt8[ndarray, "H W 3"]): RGB image.
        depth_map (UInt16[ndarray, "H W"]): Depth map computed from multi-view structure-from-motion.
            The depth values are scale-consistent across views, but only accurate up to an unknown global scale factor.
        confidence_mask (UInt8[ndarray, "H W"]): Confidence mask.
        pointcloud (o3d.geometry.PointCloud): Point cloud derived from the depth maps.
        pinhole_param (PinholeParameters): Pinhole camera parameters.
    """

    cam_name: str
    rgb_image: UInt8[ndarray, "H W 3"]
    depth_map: UInt16[ndarray, "H W"]
    confidence_mask: UInt8[ndarray, "H W"]
    pointcloud: o3d.geometry.PointCloud
    pinhole_param: PinholeParameters


def generate_multiview_pred(
    pred_class: VGGTPredictions,
    img_tensors: Float32[Tensor, "num_img 3 resized_h resized_w"],
    rgb_list: list[UInt8[ndarray, "original_h original_w 3"]],
    confidence_threshold: int | float,
) -> list[MultiviewPred]:
    pred_class = pred_class.remove_batch_dim_if_one()
    assert len(pred_class.cam_T_world.shape) == 3, "Currently batch size of 1 is only supported"

    # Generate world points from depth map,this is usually more accurate than the world points from pose encoding
    depth_maps: Float32[ndarray, "num_cams resized_h resized_w 1"] = pred_class.depth
    world_points: Float32[ndarray, "num_cams resized_h resized_w 3"] = unproject_depth_map_to_point_map(
        depth_maps, pred_class.cam_T_world, pred_class.intrinsic
    ).astype(np.float32)

    # Get colors from original images and reshape them to match points
    processed_imgs: Float32[ndarray, "num_cams 3 resized_h resized_w"] = img_tensors.numpy(force=True)
    # Rearrange to match point shape expectation
    processed_imgs: Float32[ndarray, "num_cams resized_h resized_w 3"] = rearrange(
        processed_imgs,
        "num_cams C resized_h resized_w -> num_cams resized_h resized_w C",
    )
    # Flatten both points and colors
    flattened_points: Float32[ndarray, "num_points 3"] = rearrange(
        world_points,
        "num_cams resized_h resized_w C -> (num_cams resized_h resized_w) C",
    )
    flattened_colors: Float32[ndarray, "num_points 3"] = rearrange(
        processed_imgs,
        "num_cams resized_h resized_w C -> (num_cams resized_h resized_w) C",
    )

    depth_confs: Float32[ndarray, "num_cams resized_h resized_w"] = pred_class.depth_conf
    conf: Float32[ndarray, "num_points"] = depth_confs.reshape(-1)  # noqa UP037

    # Convert percentage threshold to actual confidence value
    conf_threshold = 0.0 if confidence_threshold == 0.0 else np.percentile(conf, confidence_threshold)
    conf_mask = (conf >= conf_threshold) & (conf > 1e-5)

    vertices_3d: Float32[ndarray, "num_points 3"] = flattened_points[conf_mask]
    colors_rgb: Float32[ndarray, "num_points 3"] = flattened_colors[conf_mask]

    # Create an empty point cloud
    pcd = o3d.geometry.PointCloud()

    # Ensure your positions and colors are of the appropriate type (typically float64 for points)
    pcd.points = o3d.utility.Vector3dVector(vertices_3d * 1000)  # Scale to allow saving as uint16 later on
    pcd.colors = o3d.utility.Vector3dVector(colors_rgb)

    mv_pred_list: list[MultiviewPred] = []
    for idx, (intri, extri, processed_img, original_img, depth_map, depth_conf) in enumerate(
        zip(
            pred_class.intrinsic,
            pred_class.cam_T_world,
            processed_imgs,
            rgb_list,
            depth_maps,
            depth_confs,
            strict=True,
        )
    ):
        cam_name: str = f"camera_{idx}"
        intri_param = Intrinsics(
            camera_conventions="RDF",
            fl_x=float(intri[0, 0]),
            fl_y=float(intri[1, 1]),
            cx=float(intri[0, 2]),
            cy=float(intri[1, 2]),
            width=processed_img.shape[1],
            height=processed_img.shape[0],
        )
        extri_param = Extrinsics(
            cam_R_world=extri[:, :3],
            cam_t_world=extri[:, 3] * 1000,  # to allow saving as uint16 later on
        )
        pinhole_param = PinholeParameters(name=cam_name, intrinsics=intri_param, extrinsics=extri_param)
        conf_threshold = 0.0 if confidence_threshold == 0.0 else np.percentile(depth_conf, confidence_threshold)
        conf_mask = (depth_conf >= conf_threshold) & (depth_conf > 1e-5)
        # filter depth map based on confidence
        depth_map = depth_map.squeeze()
        depth_map[~conf_mask] = 0.0
        # resize image, confidence mask and depth map to original image size
        # Use INTER_LINEAR for the processed RGB image (standard for color images)

        processed_img = cv2.resize(
            processed_img, (original_img.shape[1], original_img.shape[0]), interpolation=cv2.INTER_LINEAR
        )

        # Use INTER_NEAREST for the confidence mask to preserve binary values
        conf_mask = cv2.resize(
            conf_mask.astype(np.float32),
            (original_img.shape[1], original_img.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

        # Use INTER_NEAREST for depth map to preserve discontinuities and avoid floating artifacts
        depth_map = cv2.resize(
            depth_map, (original_img.shape[1], original_img.shape[0]), interpolation=cv2.INTER_NEAREST
        )

        # rescale camera parameters to original image size
        pinhole_param.intrinsics = rescale_intri(
            pinhole_param.intrinsics,
            target_width=original_img.shape[1],
            target_height=original_img.shape[0],
        )

        # Normalize the processed image to [0, 1] range
        normalized = (processed_img - processed_img.min()) / (processed_img.max() - processed_img.min())
        rgb_image = (normalized * 255).clip(0, 255).astype(np.uint8)
        # convert depth map to UInt16, this means we need to multiply by 1000 the point cloud, extrinsics, and depth map
        mv_pred_list.append(
            MultiviewPred(
                cam_name=cam_name,
                rgb_image=rgb_image,
                depth_map=(depth_map * 1000).astype(np.uint16),  # convert to uint16
                confidence_mask=(conf_mask * 255).astype(np.uint8),
                pointcloud=pcd,
                pinhole_param=pinhole_param,
            )
        )

    return mv_pred_list


class VGGTPredictor:
    def __init__(
        self,
        device: Literal["cpu", "cuda"],
        confidence_threshold: int | float = 50.0,
    ) -> None:
        self.device = device
        self.confidence_threshold = confidence_threshold
        load_start: float = timer()
        print("Loading model...")
        self.model = VGGT.from_pretrained("facebook/VGGT-1B").to(self.device)
        print("Model loaded in", timer() - load_start, "seconds")
        self.dtype: torch.dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    def __call__(self, rgb_list: list[UInt8[ndarray, "H W 3"]]) -> list[MultiviewPred]:
        img_tensors: Float32[Tensor, "num_img 3 H W"] = preprocess_images(rgb_list).to(self.device)
        # Run inference
        print("Running inference...")
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=self.dtype):
            # run model and convert to dataclass for type validaton + easy access
            predictions: dict = self.model(img_tensors)

        # Convert pose encoding to extrinsic and intrinsic matrices
        print("Converting pose encoding to extrinsic and intrinsic matrices...")
        extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], img_tensors.shape[-2:])
        predictions["extrinsic"] = extrinsic
        predictions["intrinsic"] = intrinsic

        # Tensor -> Numpy conversion
        for key in predictions:
            if isinstance(predictions[key], torch.Tensor):
                predictions[key] = predictions[key].numpy(force=True)

        # Convert from dict to dataclass and performs runtime type validation for easy access
        pred_class: VGGTPredictions = from_dict(VGGTPredictions, predictions)
        calibration_data: list[MultiviewPred] = generate_multiview_pred(
            pred_class,
            img_tensors=img_tensors,
            rgb_list=rgb_list,
            confidence_threshold=self.confidence_threshold,
        )
        return calibration_data
