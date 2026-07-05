from huggingface_hub import hf_hub_download

print("Downloading model...")

model_path = hf_hub_download(
    repo_id="your-model-repo",
    filename="model.pth",
    local_dir="models"
)

print("Model saved to:", model_path)