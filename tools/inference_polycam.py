from argparse import ArgumentParser
import rerun as rr
from pathlib import Path
from PIL import Image
from monopriors.monoprior import DsineAndUnidepth, MonoPriorPrediction
import numpy as np
from monopriors.data import load_raw_polycam_data
import cv2
import zipfile


def log_depth_pred(
    parent_path,
    rgb,
    depth_np,
    normal_np,
    rotation,
    translation,
    K_33,
    gt_depth=None,
):
    cam_path = parent_path / "cam"
    pinhole_path = cam_path / "pinhole"
    h, w, _ = rgb.shape

    rr.log(
        f"{cam_path}",
        rr.Transform3D(translation=translation, mat3x3=rotation, from_parent=False),
    )
    rr.log(
        f"{pinhole_path}",
        rr.Pinhole(
            image_from_camera=K_33,
            width=w,
            height=h,
            camera_xyz=rr.ViewCoordinates.RUB,
        ),
    )
    rr.log(f"{pinhole_path}/image", rr.Image(rgb))
    rr.log(f"{pinhole_path}/depth", rr.DepthImage(depth_np, meter=1))
    rr.log(f"{pinhole_path}/normal", rr.Image(normal_np))
    if gt_depth is not None:
        rr.log(f"{pinhole_path}/gt_depth", rr.DepthImage(gt_depth, meter=1000))
        diff_depth_l1 = np.abs((depth_np[:, :, 0] - gt_depth / 1000))
        # normalize to 0-255
        diff_depth_l1 = (diff_depth_l1 / diff_depth_l1.max() * 255).astype(np.uint8)
        rr.log(f"{pinhole_path}/depth_error_l1", rr.Image(diff_depth_l1))


def extract_zip(zip_path: Path, extract_dir: Path):
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_dir)


def main(zip_path: Path):
    extract_dir = zip_path.parent / zip_path.stem

    if zip_path.is_file() and zip_path.exists() and (not extract_dir.exists()):
        extract_zip(zip_path, extract_dir=extract_dir)
        final_extract_dir = extract_dir / extract_dir.name
    else:
        final_extract_dir = extract_dir / extract_dir.name

    image_dir = final_extract_dir / "corrected_images"
    camera_dir = final_extract_dir / "corrected_cameras"
    depth_dir = (
        final_extract_dir / "depth" if (final_extract_dir / "depth").exists() else None
    )

    assert (
        image_dir.exists() and image_dir.is_dir()
    ), f"Image directory not found: {image_dir}"
    assert (
        camera_dir.exists() and camera_dir.is_dir()
    ), f"Camera directory not found: {camera_dir}"

    image_paths = sorted(image_dir.glob("*.jpg"))
    camera_paths = sorted(camera_dir.glob("*.json"))
    if depth_dir is not None:
        assert depth_dir.exists() and depth_dir.is_dir(), "Depth directory not found"
        depth_paths = sorted(depth_dir.glob("*.png"))
        assert len(image_paths) == len(depth_paths), "Image and depth mismatch"

    model = DsineAndUnidepth()

    parent_path = Path("world")
    rr.log(f"{parent_path}", rr.ViewCoordinates.RUB, timeless=True)

    timestep = 0
    for idx, (image_path, camera_path) in enumerate(zip(image_paths, camera_paths)):
        rr.set_time_sequence("timestep", timestep)
        assert image_path.stem == camera_path.stem, "Image and camera mismatch"
        rgb = np.array(Image.open(image_path))
        # load depth from polycam if available
        if depth_dir is not None:
            # depth of shape 256x192
            gt_depth = cv2.imread(str(depth_paths[idx]), cv2.IMREAD_ANYDEPTH)
            # upscale to image size
            gt_depth = cv2.resize(
                gt_depth, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST
            )
        cam_data = load_raw_polycam_data(camera_path)
        K_33 = np.array(
            [
                [cam_data.fx, 0, cam_data.cx],
                [0, cam_data.fy, cam_data.cy],
                [0, 0, 1],
            ],
            dtype=np.float32,
        )
        cam_T_world = np.array(
            [
                [cam_data.t_00, cam_data.t_01, cam_data.t_02, cam_data.t_03],
                [cam_data.t_10, cam_data.t_11, cam_data.t_12, cam_data.t_13],
                [cam_data.t_20, cam_data.t_21, cam_data.t_22, cam_data.t_23],
                [0, 0, 0, 1],
            ]
        )

        pred: MonoPriorPrediction = model(rgb)
        depth_np_bhw1, normal_np_bhw3, _ = pred.to_numpy()

        parent_path = Path("world")
        log_depth_pred(
            parent_path=parent_path,
            rgb=rgb,
            depth_np=depth_np_bhw1[0],
            normal_np=normal_np_bhw3[0],
            rotation=cam_T_world[:3, :3],
            translation=cam_T_world[:3, 3],
            K_33=K_33,
            gt_depth=gt_depth if depth_dir is not None else None,
        )
        timestep += 1


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
