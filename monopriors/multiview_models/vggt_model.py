from dataclasses import dataclass
from timeit import default_timer as timer
from typing import Literal, TypedDict

import cv2
import numpy as np
import open3d as o3d
import torch
from einops import rearrange
from jaxtyping import Bool, Float32, UInt8, UInt16
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


class PreprocessingMetadata(TypedDict):
    original_size: tuple[int, int]  # (width, height)
    mode: Literal["crop", "pad"]  # Processing mode
    target_size: int  # Target width (usually 518px)
    padding: dict[Literal["top", "left", "right", "bottom"], int]  # Padding values
    new_size: tuple[int, int]  # (width, height) after resizing


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
) -> tuple[Float32[torch.Tensor, "N 3 H W"], list[PreprocessingMetadata]]:
    """
    A quick start function to preprocess images for model input.

    Args:
        rgb_list (list): List of RGB images as numpy arrays
        mode (str): Processing mode, either "crop" or "pad"

    Returns:
        tuple: (
            torch.Tensor: Batched tensor of preprocessed images with shape (N, 3, H, W),
            list: List of preprocessing metadata dictionaries for each image
        )

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

    images = []
    shapes = set()
    to_tensor = TF.ToTensor()
    target_size = 518
    metadata_list: list[PreprocessingMetadata] = []

    # First process all images and collect their shapes
    for rgb in rgb_list:
        # Convert the numpy array to PIL Image to ensure identical processing
        pil_img = Image.fromarray(rgb)
        original_width, original_height = pil_img.size

        # Initialize metadata as TypedDict with explicit constructor
        metadata = PreprocessingMetadata(
            original_size=(original_width, original_height),
            mode=mode,
            target_size=target_size,
            padding={"top": 0, "left": 0, "right": 0, "bottom": 0},
            new_size=(0, 0),  # Will be filled later
        )

        if mode == "pad":
            # Make the largest dimension 518px while maintaining aspect ratio
            if original_width >= original_height:
                new_width = target_size
                new_height = round(original_height * (new_width / original_width) / 14) * 14  # Make divisible by 14
            else:
                new_height = target_size
                new_width = round(original_width * (new_height / original_height) / 14) * 14  # Make divisible by 14

            metadata["new_size"] = (new_width, new_height)
            # Calculate padding
            pad_top = (target_size - new_height) // 2
            pad_bottom = target_size - new_height - pad_top
            pad_left = (target_size - new_width) // 2
            pad_right = target_size - new_width - pad_left

            metadata["padding"] = {"top": pad_top, "bottom": pad_bottom, "left": pad_left, "right": pad_right}

            # Resize with new dimensions using PIL's BICUBIC
            pil_img = pil_img.resize((new_width, new_height), Image.BICUBIC)

            # Convert to tensor
            img = to_tensor(pil_img)

            # Apply padding
            img = torch.nn.functional.pad(
                img,
                (pad_left, pad_right, pad_top, pad_bottom),
                mode="constant",
                value=1.0,
            )
        else:  # mode == "crop"
            # Original behavior: set width to target_size
            new_width = target_size
            # Calculate height maintaining aspect ratio, divisible by 14
            new_height = round(original_height * (new_width / original_width) / 14) * 14
            metadata["new_size"] = (new_width, new_height)

            # Resize with new dimensions using PIL's BICUBIC for exact matching
            pil_img = pil_img.resize((new_width, new_height), Image.BICUBIC)

            # Convert to tensor using the same to_tensor transform
            img = to_tensor(pil_img)  # Convert to tensor (0, 1)

            # Center crop height if it's larger than target_size
            if new_height > target_size:
                start_y = (new_height - target_size) // 2
                metadata["padding"]["top"] = -start_y  # Negative value indicates cropping
                img = img[:, start_y : start_y + target_size, :]
                metadata["new_size"] = (new_width, target_size)

        shapes.add((img.shape[1], img.shape[2]))
        images.append(img)
        metadata_list.append(metadata)

    # Check if we have different shapes
    if len(shapes) > 1:
        print(f"Warning: Found images with different shapes: {shapes}")
        # Find maximum dimensions
        max_height = max(shape[0] for shape in shapes)
        max_width = max(shape[1] for shape in shapes)

        # Pad images if necessary
        padded_images = []
        for i, img in enumerate(images):
            h_padding = max_height - img.shape[1]
            w_padding = max_width - img.shape[2]

            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left

                # Update metadata with additional padding
                metadata_list[i]["padding"]["top"] += pad_top
                metadata_list[i]["padding"]["bottom"] += pad_bottom
                metadata_list[i]["padding"]["left"] += pad_left
                metadata_list[i]["padding"]["right"] += pad_right

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

    return images, metadata_list


def remove_padding_from_prediction(
    pred: np.ndarray,
    metadata: PreprocessingMetadata,
) -> np.ndarray:
    """
    Remove padding from a prediction tensor based on preprocessing metadata.

    Args:
        pred: The prediction tensor/array with padding
        metadata: Dictionary containing padding information

    Returns:
        The unpadded array/tensor
    """
    # Get padding values
    pad_top = metadata["padding"]["top"]
    pad_left = metadata["padding"]["left"]
    new_width, new_height = metadata["new_size"]

    if metadata["mode"] == "pad":
        # For pad mode, we need to crop out the padding
        if pred.ndim == 2:  # For 2D arrays like depth maps or masks
            return pred[pad_top : pad_top + new_height, pad_left : pad_left + new_width]
        elif pred.ndim == 3:  # For RGB images (H, W, C)
            return pred[pad_top : pad_top + new_height, pad_left : pad_left + new_width, :]
        else:
            raise ValueError(f"Unsupported tensor dimension: {pred.ndim}")
    else:  # For crop mode
        # In crop mode, padding values are used differently and might be negative
        # But we generally don't need to uncrop - we just need to resize later
        return pred


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
    pointcloud_conf: Bool[ndarray, "num_points"]
    pinhole_param: PinholeParameters


def generate_multiview_pred(
    pred_class: VGGTPredictions,
    img_tensors: Float32[Tensor, "num_img 3 resized_h resized_w"],
    rgb_list: list[UInt8[ndarray, "original_h original_w 3"]],
    confidence_threshold: int | float,
    metadata_list: list[PreprocessingMetadata] | None = None,
) -> list[MultiviewPred]:
    pred_class = pred_class.remove_batch_dim_if_one()
    assert len(pred_class.cam_T_world.shape) == 3, "Currently batch size of 1 is only supported"

    # Generate world points from depth map, this is usually more accurate than the world points from pose encoding
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

    # Process each image's data first - remove padding if metadata is available
    if metadata_list:
        unpadded_depth_maps = []
        unpadded_world_points = []
        unpadded_processed_imgs = []
        unpadded_depth_confs = []

        for i in range(len(processed_imgs)):
            # Remove padding from depths, world points, processed images, and confidence maps
            unpadded_depth_maps.append(remove_padding_from_prediction(depth_maps[i], metadata_list[i]))
            unpadded_world_points.append(remove_padding_from_prediction(world_points[i], metadata_list[i]))
            unpadded_processed_imgs.append(remove_padding_from_prediction(processed_imgs[i], metadata_list[i]))
            unpadded_depth_confs.append(remove_padding_from_prediction(pred_class.depth_conf[i], metadata_list[i]))

            # Also need to update camera intrinsics to account for removed padding
            if metadata_list[i]["mode"] == "pad":
                pad_left = metadata_list[i]["padding"]["left"]
                pad_top = metadata_list[i]["padding"]["top"]

                # Adjust principal point to account for removed padding
                pred_class.intrinsic[i, 0, 2] -= pad_left
                pred_class.intrinsic[i, 1, 2] -= pad_top

        # Replace the padded data with unpadded versions
        depth_maps = np.array(unpadded_depth_maps)
        world_points = np.array(unpadded_world_points)
        processed_imgs = np.array(unpadded_processed_imgs)
        depth_confs = np.array(unpadded_depth_confs)
    else:
        depth_confs = pred_class.depth_conf

    # Now create the point cloud from all unpadded data
    # Flatten both points and colors
    flattened_points: Float32[ndarray, "num_points 3"] = rearrange(
        world_points,
        "num_cams resized_h resized_w C -> (num_cams resized_h resized_w) C",
    )
    flattened_colors: Float32[ndarray, "num_points 3"] = rearrange(
        processed_imgs,
        "num_cams resized_h resized_w C -> (num_cams resized_h resized_w) C",
    )

    conf: Float32[ndarray, "num_points"] = depth_confs.reshape(-1)  # noqa UP037

    # Convert percentage threshold to actual confidence value
    conf_threshold = 0.0 if confidence_threshold == 0.0 else np.percentile(conf, confidence_threshold)
    pc_conf_mask = (conf >= conf_threshold) & (conf > 1e-5)

    vertices_3d: Float32[ndarray, "num_points 3"] = flattened_points
    colors_rgb: Float32[ndarray, "num_points 3"] = flattened_colors

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
        # depth_map[~conf_mask] = 0.0

        depth_map = depth_map.squeeze()
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
                pointcloud_conf=pc_conf_mask,
                pinhole_param=pinhole_param,
            )
        )

    return mv_pred_list


class VGGTPredictor:
    def __init__(
        self,
        device: Literal["cpu", "cuda"],
        confidence_threshold: int | float = 50.0,
        preprocessing_mode: Literal["crop", "pad"] = "crop",
    ) -> None:
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.preprocessing_mode = preprocessing_mode
        load_start: float = timer()
        print("Loading model...")
        self.model = VGGT.from_pretrained("facebook/VGGT-1B").to(self.device)
        print("Model loaded in", timer() - load_start, "seconds")
        self.dtype: torch.dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    def __call__(self, rgb_list: list[UInt8[ndarray, "H W 3"]]) -> list[MultiviewPred]:
        img_tensors, metadata_list = preprocess_images(rgb_list, mode=self.preprocessing_mode)
        img_tensors = img_tensors.to(self.device)

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
            metadata_list=metadata_list if self.preprocessing_mode == "pad" else None,
        )
        return calibration_data
