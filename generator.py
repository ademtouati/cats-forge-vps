import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import random
import gc

class NakedMaleGenerator:
    def __init__(self):
        self.device = "cpu"
        self.pipe = None
        self._load_pipe()

    def _load_pipe(self):
        self.pipe = StableDiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float32,
            safety_checker=None,
            requires_safety_checker=False
        )
        self.pipe.to(self.device)

    def generate(self, prompt, negative_prompt="", steps=50):
        result = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=steps
        )
        return result.images[0]
