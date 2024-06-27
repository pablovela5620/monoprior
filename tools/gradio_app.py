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
    UniDepthRelativePredictor,
    get_relative_predictor,
    RELATIVE_PREDICTORS,
)
from monopriors.relative_depth_models.base_relative_depth import BaseRelativePredictor
from monopriors.rr_logging_utils import (
    log_relative_pred,
    create_depth_comparison_blueprint,
)
import rerun as rr
from gradio_rerun import Rerun
from pathlib import Path
from typing import Literal, get_args
import gc

from jaxtyping import UInt8

title = "# Depth Comparison"
description1 = """Demo to help compare different depth models. Including both Scale | Shift Invariant and Metric Depth types."""
description2 = """Invariant models mean they have no true scale and are only relative, where as Metric models have a true scale and are absolute (meters)."""

DEVICE: Literal["cuda"] | Literal["cpu"] = (
    "cuda" if torch.cuda.is_available() else "cpu"
)
if gr.NO_RELOAD:
    MODEL_1 = DepthAnythingV2Predictor(device=DEVICE)
    MODEL_2 = UniDepthRelativePredictor(device=DEVICE)


def predict_depth(
    model: BaseRelativePredictor, rgb: UInt8[np.ndarray, "h w 3"]
) -> RelativeDepthPrediction:
    try:
        model.set_model_device(device=DEVICE)
        relative_pred: RelativeDepthPrediction = model(rgb, None)
        return relative_pred
    except Exception as e:
        raise gr.Error(f"Error with model {model.__class__.__name__}: {e}")


if IN_SPACES:
    predict_depth = spaces.GPU(predict_depth)


def load_models(
    model_1: RELATIVE_PREDICTORS,
    model_2: RELATIVE_PREDICTORS,
    progress=gr.Progress(),
) -> Literal["Models loaded"]:
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
        loaded_models.append(get_relative_predictor(model)(device=DEVICE))

        progress(0.5, desc=f"Loaded {model}")

    progress(1, desc="Models Loaded")
    MODEL_1, MODEL_2 = loaded_models

    return "Models loaded"


@rr.thread_local_stream("depth")
def on_submit(rgb: UInt8[np.ndarray, "h w 3"]):
    stream: rr.BinaryStream = rr.binary_stream()
    models_list = [MODEL_1, MODEL_2]
    blueprint = create_depth_comparison_blueprint(models_list)
    rr.send_blueprint(blueprint)
    for model in models_list:
        # get the name of the model
        parent_log_path = Path(f"{model.__class__.__name__}")
        rr.log(f"{parent_log_path}", rr.ViewCoordinates.RDF, timeless=True)

        relative_pred: RelativeDepthPrediction = predict_depth(model, rgb)

        log_relative_pred(
            parent_log_path=parent_log_path, relative_pred=relative_pred, rgb_hw3=rgb
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
                    choices=list(get_args(RELATIVE_PREDICTORS)),
                    label="Model1",
                    value="DepthAnythingV2Predictor",
                )
                model_2_dropdown = gr.Dropdown(
                    choices=list(get_args(RELATIVE_PREDICTORS)),
                    label="Model2",
                    value="UniDepthPredictor",
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
