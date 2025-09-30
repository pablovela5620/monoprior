import uuid
from pathlib import Path

import gradio as gr
import numpy as np
import rerun as rr
from gradio_rerun import Rerun
from jaxtyping import UInt8
from numpy import ndarray
from PIL import Image


# Whenever we need a recording, we construct a new recording stream.
# As long as the app and recording IDs remain the same, the data
# will be merged by the Viewer.
def get_recording(recording_id: uuid.UUID) -> rr.RecordingStream:
    return rr.RecordingStream(application_id="rerun_example_gradio", recording_id=recording_id)


def streaming_repeated_blur(recording_id, imgs):
    yield from _streaming_repeated_blur(recording_id, imgs)


def _streaming_repeated_blur(recording_id: uuid.UUID, img_files: str | list[str]):
    # Here we get a recording using the provided recording id.
    recording: rr.RecordingStream = get_recording(recording_id)
    stream = recording.binary_stream()
    if isinstance(img_files, str):
        img_files = [img_files]
    img_paths: list[Path] = [Path(p) for p in img_files]

    if len(img_paths) < 1:
        raise gr.Error("Must provide an image to blur.")

    rr.set_time("iteration", sequence=0, recording=recording)
    for idx, img_path in enumerate(img_paths):
        pil_img: Image.Image = Image.open(img_path).convert("RGB")
        rgb: UInt8[ndarray, "height width 3"] = np.array(pil_img)
        rr.log(f"image_{idx}", rr.Image(rgb, color_model=rr.ColorModel.RGB), recording=recording)

        yield stream.read()


with gr.Blocks() as mv_calibration_block:
    # We make a new recording id, and store it in a Gradio's session state.
    recording_id = gr.State(uuid.uuid4())
    with gr.Row():
        input_image = gr.File(
            label="Input Images",
            file_count="multiple",
            file_types=[".png", ".jpg", ".jpeg"],
        )
        with gr.Column():
            stream_blur_btn = gr.Button("Stream Repeated Blur")

    with gr.Row():
        rr_viewer = Rerun(
            streaming=True,
            panel_states={
                "time": "collapsed",
                "blueprint": "collapsed",
                "selection": "collapsed",
            },
        )

    stream_blur_btn.click(
        # Using the `viewer` as an output allows us to stream data to it by yielding bytes from the callback.
        streaming_repeated_blur,
        inputs=[recording_id, input_image],
        outputs=[rr_viewer],
    )
