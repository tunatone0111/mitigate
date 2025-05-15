import json
import os

import torch
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline

from ..mitigation_method import MitigationMethod


class ICLR25(MitigationMethod):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_id = "CompVis/stable-diffusion-v1-4"
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id, safety_checker=None
        ).to(self.device)
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_pretrained(
            model_id, subfolder="scheduler"
        )
        dirname = os.path.dirname(__file__)
        json_path = os.path.join(dirname, "flipd_perturbations.json")
        self.prompt_map = json.load(open(json_path))

    def mitigate(self, prompt, num_inference_steps, seed):
        gen = torch.Generator(device=self.device).manual_seed(seed)
        perturbed_prompt = self.prompt_map[prompt]["1"][0]["perturbed_prompt"]
        image = self.pipe(
            perturbed_prompt,
            num_inference_steps=num_inference_steps,
            generator=gen,
        ).images[0]
        return image
