import gradio as gr
import numpy as np
import torch
from monopriors.relative_depth_models import (
    RelativeDepthPrediction,
    get_relative_predictor,
    RELATIVE_PREDICTORS,
)
from monopriors.relative_depth_models.base_relative_depth import BaseRelativePredictor
from monopriors.rr_logging_utils import (
    log_relative_pred,
    create_relative_depth_blueprint,
)
import rerun as rr
from gradio_rerun import Rerun
from pathlib import Path
from typing import Literal, get_args
import gc

from jaxtyping import UInt8
import mmcv

try:
    import spaces  # type: ignore

    IN_SPACES = True
except ImportError:
    print("Not running on Zero")
    IN_SPACES = False

title = "# Depth Comparison"
description1 = """Demo to help compare different depth models. Including both Scale | Shift Invariant and Metric Depth types."""
description2 = """Invariant models mean they have no true scale and are only relative, where as Metric models have a true scale and are absolute (meters)."""
description3 = """Checkout the [Github Repo](https://github.com/pablovela5620/monoprior) [![GitHub Repo stars](https://img.shields.io/github/stars/pablovela5620/monoprior)](https://github.com/pablovela5620/monoprior)"""
model_load_status: str = "Models loaded and ready to use!"
DEVICE: Literal["cuda"] | Literal["cpu"] = (
    "cuda" if torch.cuda.is_available() else "cpu"
)
MODELS_TO_SKIP: list[str] = []
if gr.NO_RELOAD:
    MODEL_1 = get_relative_predictor("DepthAnythingV2Predictor")(device=DEVICE)
    MODEL_2 = get_relative_predictor("UniDepthRelativePredictor")(device=DEVICE)


def predict_depth(
    model: BaseRelativePredictor, rgb: UInt8[np.ndarray, "h w 3"]
) -> RelativeDepthPrediction:
    model.set_model_device(device=DEVICE)
    relative_pred: RelativeDepthPrediction = model(rgb, None)
    return relative_pred


if IN_SPACES:
    predict_depth = spaces.GPU(predict_depth)
    # remove any model that fails on zerogpu spaces
    MODELS_TO_SKIP.extend(["Metric3DRelativePredictor"])


def load_models(
    model_1: RELATIVE_PREDICTORS,
    model_2: RELATIVE_PREDICTORS,
    progress=gr.Progress(),
) -> str:
    models: list[int] = [model_1, model_2]
    # check if the models are in the list of models to skip
    if any(model in MODELS_TO_SKIP for model in models):
        raise gr.Error(
            f"Model not supported on ZeroGPU, please try another model: {MODELS_TO_SKIP}"
        )

    global MODEL_1, MODEL_2
    # delete the previous models and clear gpu memory
    if "MODEL_1" in globals():
        del MODEL_1
    if "MODEL_2" in globals():
        del MODEL_2
    torch.cuda.empty_cache()
    gc.collect()

    progress(0, desc="Loading Models please wait...")

    loaded_models = []

    for model in models:
        loaded_models.append(get_relative_predictor(model)(device=DEVICE))
        progress(0.5, desc=f"Loaded {model}")

    progress(1, desc="Models Loaded")
    MODEL_1, MODEL_2 = loaded_models

    return model_load_status


@rr.thread_local_stream("depth")
def on_submit(
    rgb: UInt8[np.ndarray, "h w 3"],
    remove_flying_pixels: bool,
    depth_map_threshold: float,
):
    stream: rr.BinaryStream = rr.binary_stream()
    models_list = [MODEL_1, MODEL_2]
    blueprint = create_relative_depth_blueprint(models_list)
    rr.send_blueprint(blueprint)

    # resize the image to have a max dim of 1024
    max_dim = 1024
    current_dim = max(rgb.shape[0], rgb.shape[1])
    if current_dim > max_dim:
        scale_factor = max_dim / current_dim
        rgb = mmcv.imrescale(img=rgb, scale=scale_factor)

    try:
        for model in models_list:
            # get the name of the model
            parent_log_path = Path(f"{model.__class__.__name__}")
            rr.log(f"{parent_log_path}", rr.ViewCoordinates.RDF, timeless=True)

            relative_pred: RelativeDepthPrediction = predict_depth(model, rgb)

            log_relative_pred(
                parent_log_path=parent_log_path,
                relative_pred=relative_pred,
                rgb_hw3=rgb,
                remove_flying_pixels=remove_flying_pixels,
                depth_edge_threshold=depth_map_threshold,
            )

            yield stream.read()
    except Exception as e:
        raise gr.Error(f"Error with model {model.__class__.__name__}: {e}")


with gr.Blocks() as demo:
    gr.Markdown(title)
    gr.Markdown(description1)
    gr.Markdown(description2)
    gr.Markdown(description3)
    gr.Markdown("### Depth Prediction demo")

    with gr.Tab(label="Scale and Shift Invariant"):
        with gr.Row():
            input_image = gr.Image(
                label="Input Image",
                type="numpy",
                height=300,
            )
            with gr.Column():
                with gr.Row():
                    remove_flying_pixels = gr.Checkbox(
                        label="Remove Flying Pixels",
                        value=True,
                        interactive=True,
                    )
                    depth_map_threshold = gr.Slider(
                        label="⬇️ number == more pruning ⬆️ less pruning",
                        minimum=0.05,
                        maximum=0.95,
                        step=0.05,
                        value=0.1,
                    )
                with gr.Row():
                    model_1_dropdown = gr.Dropdown(
                        choices=list(get_args(RELATIVE_PREDICTORS)),
                        label="Model1",
                        value="DepthAnythingV2Predictor",
                    )
                    model_2_dropdown = gr.Dropdown(
                        choices=list(get_args(RELATIVE_PREDICTORS)),
                        label="Model2",
                        value="UniDepthRelativePredictor",
                    )
                model_status = gr.Textbox(
                    label="Model Status",
                    value=model_load_status,
                    interactive=False,
                )

        with gr.Row():
            submit = gr.Button(value="Compute Depth")
            load_models_btn = gr.Button(value="Load Models")
        rr_viewer = Rerun(streaming=True, height=800)

        submit.click(
            on_submit,
            inputs=[input_image, remove_flying_pixels, depth_map_threshold],
            outputs=[rr_viewer],
        )

        load_models_btn.click(
            load_models,
            inputs=[model_1_dropdown, model_2_dropdown],
            outputs=[model_status],
        )

        # get all jpegs in examples path
        examples_paths = Path("examples").glob("*.jpeg")
        # set the examples to be the sorted list of input parameterss (path, remove_flying_pixels, depth_map_threshold)
        examples_list = sorted([[str(path), True, 0.1] for path in examples_paths])
        examples = gr.Examples(
            examples=examples_list,
            inputs=[input_image, remove_flying_pixels, depth_map_threshold],
            outputs=[rr_viewer],
            fn=on_submit,
            cache_examples=False,
        )

if __name__ == "__main__":
    demo.launch()
