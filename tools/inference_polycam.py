from argparse import ArgumentParser
import rerun as rr
from pathlib import Path
from PIL import Image
from monopriors.monoprior_models import DsineAndUnidepth, MonoPriorPrediction
import numpy as np
from monopriors.data.polycam_data import load_raw_polycam_data, PolycamCameraData
from monopriors.depth_fuser import Open3DFuser
import cv2
import zipfile
from tqdm import tqdm
from jaxtyping import UInt8, UInt16


def log_mono_pred(
    parent_path: Path,
    rgb: UInt8[np.ndarray, "h w 3"],
    mono_pred: MonoPriorPrediction,
    cam_data: PolycamCameraData,
    gt_depth: UInt16[np.ndarray, "h w 1"] | None = None,
):
    cam_path = parent_path / "cam"
    pinhole_path = cam_path / "pinhole"
    h, w, _ = rgb.shape

    rr.log(
        f"{cam_path}",
        rr.Transform3D(
            translation=cam_data.cam_T_world_44[:3, 3],
            mat3x3=cam_data.cam_T_world_44[:3, :3],
            from_parent=True,
        ),
    )
    rr.log(
        f"{pinhole_path}",
        rr.Pinhole(
            image_from_camera=cam_data.K_33,
            width=w,
            height=h,
            camera_xyz=rr.ViewCoordinates.RDF,
        ),
    )

    rr.log(f"{pinhole_path}/image", rr.Image(rgb))
    rr.log(
        f"{pinhole_path}/depth",
        rr.DepthImage(mono_pred.metric_pred.depth_meters, meter=1.0),
    )
    rr.log(f"{pinhole_path}/normal", rr.Image(mono_pred.normal_pred.normal_hw3))
    # rr.log(
    #     f"{pinhole_path}/normal_conf",
    #     rr.DepthImage(mono_pred.normal_pred.confidence_hw1),
    # )
    if gt_depth is not None:
        rr.log(f"{pinhole_path}/gt_depth", rr.DepthImage(gt_depth, meter=1000))
        # diff_depth_l1 = np.abs((depth_np.squeeze() - gt_depth))
        # normalize to 0-255
        # diff_depth_l1 = (diff_depth_l1 / diff_depth_l1.max() * 255).astype(np.uint8)
        # rr.log(f"{pinhole_path}/depth_error_l1", rr.Image(diff_depth_l1))


def extract_zip(zip_path: Path, extract_dir: Path) -> None:
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_dir)


def main(zip_path: Path) -> None:
    print(zip_path)
    extract_dir: Path = zip_path.parent / zip_path.stem
    print(extract_dir)

    if zip_path.is_file() and zip_path.exists() and (not extract_dir.exists()):
        extract_zip(zip_path, extract_dir=extract_dir)
        final_extract_dir: Path = extract_dir / extract_dir.name
    else:
        final_extract_dir: Path = extract_dir

    print(final_extract_dir)

    image_dir: Path = final_extract_dir / "corrected_images"
    camera_dir: Path = final_extract_dir / "corrected_cameras"
    depth_dir: Path = (
        final_extract_dir / "depth" if (final_extract_dir / "depth").exists() else None
    )

    assert (
        image_dir.exists() and image_dir.is_dir()
    ), f"Image directory not found: {image_dir}"
    assert (
        camera_dir.exists() and camera_dir.is_dir()
    ), f"Camera directory not found: {camera_dir}"

    image_paths: list[Path] = sorted(image_dir.glob("*.jpg"))
    camera_paths: list[Path] = sorted(camera_dir.glob("*.json"))
    if depth_dir is not None:
        assert depth_dir.exists() and depth_dir.is_dir(), "Depth directory not found"
        depth_paths: list[Path] = sorted(depth_dir.glob("*.png"))
        assert len(image_paths) == len(depth_paths), "Image and depth mismatch"

    model = DsineAndUnidepth()
    gt_fuser = Open3DFuser()
    pred_fuser = Open3DFuser()

    parent_path: Path = Path("world")
    rr.log(f"{parent_path}", rr.ViewCoordinates.RUB, timeless=True)

    height, width, _ = np.array(Image.open(image_paths[0])).shape

    pbar = tqdm(zip(image_paths, camera_paths), total=len(image_paths))
    for idx, (image_path, camera_path) in enumerate(pbar):
        rr.set_time_sequence("timestep", idx)
        assert image_path.stem == camera_path.stem, "Image and camera mismatch"
        rgb_hw3: UInt8[np.ndarray, "h w 3"] = np.array(Image.open(image_path))
        # load depth from polycam if available
        if depth_dir is not None:
            # depth of shape 256x192
            gt_depth_hw = cv2.imread(str(depth_paths[idx]), cv2.IMREAD_ANYDEPTH)
            # upscale to image size
            gt_depth_hw: UInt16[np.ndarray, "h w"] = cv2.resize(
                gt_depth_hw,
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            )

        cam_data: PolycamCameraData = load_raw_polycam_data(camera_path)

        pred: MonoPriorPrediction = model.__call__(rgb_hw3, cam_data.K_33)
        # convert to mm and Uint16
        depth_hw1: UInt16[np.ndarray, "h w 1"] = (
            pred.metric_pred.depth_meters * 1000
        ).astype(np.uint16)

        gt_fuser.fuse_frames(
            gt_depth_hw, cam_data.K_33, cam_data.cam_T_world_44, rgb_hw3
        )
        pred_fuser.fuse_frames(
            depth_hw1.squeeze(), cam_data.K_33, cam_data.cam_T_world_44, rgb_hw3
        )

        log_mono_pred(
            parent_path=parent_path,
            rgb=rgb_hw3,
            mono_pred=pred,
            cam_data=cam_data,
            gt_depth=gt_depth_hw if depth_dir is not None else None,
        )
    gt_mesh = gt_fuser.get_mesh()
    gt_mesh.compute_vertex_normals()

    pred_mesh = pred_fuser.get_mesh()
    pred_mesh.compute_vertex_normals()

    rr.log(
        f"{parent_path}/gt_mesh",
        rr.Mesh3D(
            vertex_positions=gt_mesh.vertices,
            triangle_indices=gt_mesh.triangles,
            vertex_normals=gt_mesh.vertex_normals,
            vertex_colors=gt_mesh.vertex_colors,
        ),
    )

    rr.log(
        f"{parent_path}/pred_mesh",
        rr.Mesh3D(
            vertex_positions=pred_mesh.vertices,
            triangle_indices=pred_mesh.triangles,
            vertex_normals=pred_mesh.vertex_normals,
            vertex_colors=pred_mesh.vertex_colors,
        ),
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--zip-path",
        type=Path,
        required=True,
        help="Path to the image for inference",
    )
    rr.script_add_args(parser)

    args = parser.parse_args()
    rr.script_setup(args, "monopriors inference")
    main(args.zip_path)
    rr.script_teardown(args)
