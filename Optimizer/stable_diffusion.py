from typing import Optional
from torch import autocast
from diffusers import StableDiffusionPipeline, LMSDiscreteScheduler
import torch

class StableDiffusion():
    def __init__(self, precision_high):
        with open('token.txt') as f:
            token = f.readline()
        if precision_high:
            self.pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", use_auth_token=token)
        else:
            self.pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", revision="fp16", torch_dtype=torch.float16, use_auth_token=token)


    def generate_image(self,prompt, num_inference_steps=50, guidance_scale=7.5, seedGen=1023):
        #generator = torch.Generator("cuda").manual_seed(seedGen)
        self.image = self.pipe(prompt, height=512, width=512, 
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale)["sample"]
        return self.image
    