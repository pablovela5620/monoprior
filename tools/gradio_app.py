import gradio as gr
import numpy as np

try:
    import spaces  # type: ignore

    IN_SPACES = True
except ImportError:
    print("Not running on Zero")
    IN_SPACES = False
import torch

from monopriors.relative_depth_models import (
    DepthAnythingV2Predictor,
    RelativeDepthPrediction,
    UniDepthPredictor,
)
from monopriors.relative_depth_models.base_relative_depth import BaseRelativePredictor
import rerun as rr
import rerun.blueprint as rrb
from gradio_rerun import Rerun
from pathlib import Path
from typing import Literal, get_args
import gc

from jaxtyping import UInt8

title = "# Depth Comparison"
description1 = """Demo to help compare different depth models. Including both Scale | Shift Invariant and Metric Depth types."""
description2 = """Invariant models mean they have no true scale and are only relative, where as Metric models have a true scale and are absolute (meters)."""

relative_depth_models = Literal["Depth Anything V2", "UniDepth"]
DEVICE: Literal["cuda"] | Literal["cpu"] = (
    "cuda" if torch.cuda.is_available() else "cpu"
)
if gr.NO_RELOAD:
    MODEL_1 = DepthAnythingV2Predictor(device=DEVICE)
    MODEL_2 = UniDepthPredictor(device=DEVICE)


def create_blueprint(models: list[str]) -> rrb.Blueprint:
    model_names: list[str] = [model.__class__.__name__ for model in models]
    blueprint = rrb.Blueprint(
        rrb.Horizontal(
            contents=[
                rrb.Spatial3DView(origin=f"{model_names[0]}"),
                rrb.Vertical(
                    rrb.Spatial2DView(
                        origin=f"{model_names[0]}/camera/pinhole/image",
                    ),
                    rrb.Spatial2DView(
                        origin=f"{model_names[0]}/camera/pinhole/depth",
                    ),
                    rrb.Spatial2DView(
                        origin=f"{model_names[0]}/camera/disparity",
                    ),
                ),
                rrb.Spatial3DView(origin=f"{model_names[1]}"),
                rrb.Vertical(
                    rrb.Spatial2DView(
                        origin=f"{model_names[1]}/camera/pinhole/image",
                    ),
                    rrb.Spatial2DView(
                        origin=f"{model_names[1]}/camera/pinhole/depth",
                    ),
                    rrb.Spatial2DView(
                        origin=f"{model_names[1]}/camera/disparity",
                    ),
                ),
            ],
            column_shares=(3, 1, 3, 1),
        ),
        collapse_panels=True,
    )
    return blueprint


def log_relative_pred(
    parent_log_path: Path,
    relative_pred: RelativeDepthPrediction,
    rgb: UInt8[np.ndarray, "h w 3"],
) -> None:
    cam_log_path = parent_log_path / "camera"
    pinhole_path = cam_log_path / "pinhole"

    # assume camera is at the origin
    cam_T_world_44 = np.eye(4)

    rr.log(
        f"{cam_log_path}",
        rr.Transform3D(
            translation=cam_T_world_44[:3, 3],
            mat3x3=cam_T_world_44[:3, :3],
            from_parent=True,
        ),
    )
    rr.log(
        f"{pinhole_path}",
        rr.Pinhole(
            image_from_camera=relative_pred.K_33,
            width=rgb.shape[1],
            height=rgb.shape[0],
            camera_xyz=rr.ViewCoordinates.RDF,
        ),
    )
    rr.log(f"{pinhole_path}/image", rr.Image(rgb))

    rr.log(f"{pinhole_path}/depth", rr.DepthImage(relative_pred.depth))
    # log to cam_log_path to avoid backprojecting disparity

    # Get the 95th percentile of the disparity values
    p95 = np.percentile(relative_pred.disparity, 99)
    # Clip the disparity values at 105% of the 95th percentile
    clipped_disparity = np.clip(relative_pred.disparity, a_min=None, a_max=1.01 * p95)
    # normalize the disparity to 0-1
    clipped_disparity = (clipped_disparity - clipped_disparity.min()) / (
        clipped_disparity.max() - clipped_disparity.min()
    )
    clipped_disparity = (clipped_disparity * 255).astype(np.uint8)
    rr.log(f"{cam_log_path}/disparity", rr.DepthImage(clipped_disparity))


def predict_depth(
    model: BaseRelativePredictor, rgb: UInt8[np.ndarray, "h w 3"]
) -> RelativeDepthPrediction:
    model.set_model_device(device=DEVICE)
    relative_pred: RelativeDepthPrediction = model(rgb, None)
    return relative_pred


if IN_SPACES:
    predict_depth = spaces.GPU(predict_depth)


def load_models(
    model_1: relative_depth_models,
    model_2: relative_depth_models,
    progress=gr.Progress(),
):
    global MODEL_1, MODEL_2
    # delete the previous models and clear gpu memory
    if "MODEL_1" in globals():
        del MODEL_1
    if "MODEL_2" in globals():
        del MODEL_2
    torch.cuda.empty_cache()
    gc.collect()

    progress(0, desc="Loading Models please wait...")

    models: list[int] = [model_1, model_2]
    loaded_models = []

    for model in models:
        match model:
            case "Depth Anything V2":
                loaded_models.append(DepthAnythingV2Predictor(device=DEVICE))
            case "UniDepth":
                loaded_models.append(UniDepthPredictor(device=DEVICE))
            case _:
                raise ValueError(f"Unknown model: {model}")

        progress(0.5, desc=f"Loaded {model}")

    progress(1, desc="Models Loaded")
    MODEL_1, MODEL_2 = loaded_models

    return "Models loaded"


@rr.thread_local_stream("depth")
def on_submit(rgb: UInt8[np.ndarray, "h w 3"]):
    stream = rr.binary_stream()
    models_list = [MODEL_1, MODEL_2]
    blueprint = create_blueprint(models_list)
    rr.send_blueprint(blueprint)
    for model in models_list:
        # get the name of the model
        parent_log_path = Path(f"{model.__class__.__name__}")
        rr.log(f"{parent_log_path}", rr.ViewCoordinates.RDF, timeless=True)

        relative_pred: RelativeDepthPrediction = predict_depth(model, rgb)

        log_relative_pred(
            parent_log_path=parent_log_path, relative_pred=relative_pred, rgb=rgb
        )

        yield stream.read()


with gr.Blocks() as demo:
    gr.Markdown(title)
    gr.Markdown(description1)
    gr.Markdown(description2)
    gr.Markdown("### Depth Prediction demo")

    with gr.Row():
        input_image = gr.Image(
            label="Input Image",
            type="numpy",
            height=300,
        )
        with gr.Column():
            gr.Radio(
                choices=["Scale | Shift Invariant", "Metric (TODO)"],
                label="Depth Type",
                value="Scale | Shift Invariant",
                interactive=True,
            )
            with gr.Row():
                model_1_dropdown = gr.Dropdown(
                    choices=list(get_args(relative_depth_models)),
                    label="Model1",
                    value="Depth Anything V2",
                    interactive=True,
                )
                model_2_dropdown = gr.Dropdown(
                    choices=list(get_args(relative_depth_models)),
                    label="Model2",
                    value="UniDepth",
                    interactive=True,
                )
            model_status = gr.Textbox(
                label="Model Status",
                value="Models loaded",
                interactive=False,
            )

    with gr.Row():
        submit = gr.Button(value="Compute Depth")
        load_models_btn = gr.Button(value="Load Models")
    rr_viewer = Rerun(streaming=True, height=800)

    submit.click(
        on_submit,
        inputs=[input_image],
        outputs=[rr_viewer],
    )

    load_models_btn.click(
        load_models,
        inputs=[model_1_dropdown, model_2_dropdown],
        outputs=[model_status],
    )

    examples_paths = Path("examples").glob("*.jpeg")
    examples_list = sorted([str(path) for path in examples_paths])
    examples = gr.Examples(
        examples=examples_list,
        inputs=[input_image],
        outputs=[rr_viewer],
        fn=on_submit,
        cache_examples=False,
    )

if __name__ == "__main__":
    demo.launch()
