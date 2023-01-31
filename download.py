import torch
from diffusers import StableDiffusionPipeline
import os

def download_model():
    HF_AUTH_TOKEN = os.getenv("HF_AUTH_TOKEN")
    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16,use_auth_token=HF_AUTH_TOKEN )
if __name__ == "__main__":
    download_model()
