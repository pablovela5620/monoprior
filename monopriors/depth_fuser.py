import numpy as np
import open3d as o3d
import trimesh
from jaxtyping import Float32, UInt16, UInt8


class DepthFuser:
    def __init__(self, gt_path="", fusion_resolution=0.04, max_fusion_depth=3.0):
        self.fusion_resolution = fusion_resolution
        self.max_fusion_depth = max_fusion_depth


class Open3DFuser(DepthFuser):
    """
    Wrapper class for the open3d fuser.

    This wrapper does not support fusion of tensors with higher than batch 1.
    """

    def __init__(
        self,
        gt_path="",
        fusion_resolution=0.04,
        max_fusion_depth=3,
    ):
        super().__init__(
            gt_path,
            fusion_resolution,
            max_fusion_depth,
        )

        self.fusion_max_depth = max_fusion_depth

        voxel_size = fusion_resolution * 100
        self.volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=float(voxel_size) / 100,
            sdf_trunc=3 * float(voxel_size) / 100,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
        )

    def fuse_frames(
        self,
        depth_hw: UInt16[np.ndarray, "h w"],
        K_33: Float32[np.ndarray, "3 3"],
        cam_T_world_44: Float32[np.ndarray, "4 4"],
        rgb_hw3: UInt8[np.ndarray, "h w 3"],
    ) -> None:
        height = depth_hw.shape[0]
        width = depth_hw.shape[1]

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(rgb_hw3),
            o3d.geometry.Image(depth_hw),
            depth_scale=1000.0,
            depth_trunc=self.fusion_max_depth,
            convert_rgb_to_intensity=False,
        )

        self.volume.integrate(
            rgbd,
            o3d.camera.PinholeCameraIntrinsic(
                width=width,
                height=height,
                fx=K_33[0, 0],
                fy=K_33[1, 1],
                cx=K_33[0, 2],
                cy=K_33[1, 2],
            ),
            cam_T_world_44,
        )

    def export_mesh(self, path, use_marching_cubes_mask=None):
        o3d.io.write_triangle_mesh(path, self.volume.extract_triangle_mesh())

    def get_mesh(self, export_single_mesh=None, convert_to_trimesh=False):
        mesh = self.volume.extract_triangle_mesh()

        if convert_to_trimesh:
            mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.triangles)

        return mesh
