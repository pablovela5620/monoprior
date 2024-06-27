from typing import TypedDict
import os
from huggingface_hub import upload_file


class FileUpload(TypedDict):
    local_path: str
    repo_path: str


space_id: str | None = os.environ.get("SPACE_ID")
whl_path: str | None = os.environ.get("WHL_PATH")

assert space_id is not None, "Please set the SPACE_ID environment variable"
assert whl_path is not None, "Please set the WHL_PATH environment variable"

files_to_upload: list[FileUpload] = [
    {"local_path": "./tools/app.py", "repo_path": "app.py"},
    {"local_path": "./tools/gradio_app.py", "repo_path": "gradio_app.py"},
    {
        "local_path": "./dist/mini_monopriors-0.1.0-py3-none-any.whl",
        "repo_path": "mini_monopriors-0.1.0-py3-none-any.whl",
    },
    {"local_path": "./pyproject.toml", "repo_path": "pyproject.toml"},
]

for file in files_to_upload:
    with open(file["local_path"], "rb") as fobj:
        upload_file(
            path_or_fileobj=fobj,
            path_in_repo=file["repo_path"],
            repo_id=space_id,
            repo_type="space",
        )
