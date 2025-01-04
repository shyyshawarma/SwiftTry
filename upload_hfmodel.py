from huggingface_hub import HfApi

api = HfApi()

api.upload_folder(
    folder_path="/root/Projects/Moore-AnimateAnyone/checkpoints",
    repo_id="NMHung/video-virtual-tryon",
    repo_type="model",
)