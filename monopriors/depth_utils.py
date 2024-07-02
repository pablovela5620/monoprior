from jaxtyping import Float32
import numpy as np


def estimate_intrinsics(
    H: int, W: int, fov: float = 55.0
) -> Float32[np.ndarray, "3 3"]:
    """
    Intrinsics for a pinhole camera model from image dimensions.
    Assume fov of 55 degrees and central principal point.
    """
    f = 0.5 * W / np.tan(0.5 * fov * np.pi / 180.0)
    cx = 0.5 * W
    cy = 0.5 * H
    K_33: Float32[np.ndarray, "3 3"] = np.array(
        [[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float32
    )
    return K_33


def disparity_to_depth(
    disparity: Float32[np.ndarray, "h w"], focal_length: int, baseline: float = 1.0
) -> Float32[np.ndarray, "h w"]:
    range: float = float(np.minimum(disparity.max() / (disparity.min() + 1e-6), 100.0))
    disparity_max: float = float(disparity.max())
    min_disparity_range: float = disparity_max / range

    depth: Float32[np.ndarray, "h w"] = (focal_length * baseline) / np.maximum(
        disparity, min_disparity_range
    )
    # gamma correction for better visualizationg
    depth: Float32[np.ndarray, "h w"] = np.power(depth, 1.0 / 2.2)
    return depth


def depth_to_disparity(
    depth: Float32[np.ndarray, "h w"], focal_length: int, baseline: float = 1.0
) -> Float32[np.ndarray, "h w"]:
    disparity = (focal_length * baseline) / (depth + 0.01)
    return disparity


def depth_to_points(
    depth_1hw: Float32[np.ndarray, "1 h w"],
    K_33: Float32[np.ndarray, "3 3"],
    R=None,
    t=None,
):
    K_33_inv: Float32[np.ndarray, "3 3"] = np.linalg.inv(K_33)
    if R is None:
        R: Float32[np.ndarray, "3 3"] = np.eye(3, dtype=np.float32)
    if t is None:
        t = np.zeros(3, dtype=np.float32)

    M = np.eye(3, dtype=np.float32)

    _, height, width = depth_1hw.shape

    x = np.arange(width)
    y = np.arange(height)
    coord = np.stack(np.meshgrid(x, y), -1)
    coord = np.concatenate((coord, np.ones_like(coord)[:, :, [0]]), -1)  # z=1
    coord = coord.astype(np.float32)
    # coord = torch.as_tensor(coord, dtype=torch.float32, device=device)
    coord = coord[None]  # bs, h, w, 3

    D = depth_1hw[:, :, :, None, None]
    # print(D.shape, Kinv[None, None, None, ...].shape, coord[:, :, :, :, None].shape )
    pts3D_1 = D * K_33_inv[None, None, None, ...] @ coord[:, :, :, :, None]
    # pts3D_1 live in your coordinate system. Convert them to Py3D's
    pts3D_1 = M[None, None, None, ...] @ pts3D_1
    # from reference to targe tviewpoint
    pts3D_2 = R[None, None, None, ...] @ pts3D_1 + t[None, None, None, :, None]
    # pts3D_2 = pts3D_1
    # depth_2 = pts3D_2[:, :, :, 2, :]  # b,1,h,w
    print(pts3D_2.shape, pts3D_2.shape)
    return pts3D_2[:, :, :, :3, 0][0]
