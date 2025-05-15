import torch
from diffusers import DPMSolverMultistepScheduler, UNet2DConditionModel

from ..mitigation_method import MitigationMethod
from .local_sd_pipeline import LocalStableDiffusionPipeline
from .optim_utils import prompt_augmentation, set_random_seed


class ICLR24(MitigationMethod):
    def __init__(self, random_mode=False, pretrained=False):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if pretrained:
            unet = UNet2DConditionModel.from_pretrained(
                "finetuned_cehckpoints/checkpoint-20000/unet",
                torch_dtype=torch.bfloat16,
            )
        else:
            unet = UNet2DConditionModel.from_pretrained(
                "CompVis/stable-diffusion-v1-4",
                subfolder="unet",
                torch_dtype=torch.bfloat16,
            )

        unet.to(self.device)

        self.pipe = LocalStableDiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            unet=unet,
            torch_dtype=torch.bfloat16,
            safety_checker=None,
            requires_safety_checker=False,
        )
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_pretrained(
            "CompVis/stable-diffusion-v1-4", subfolder="scheduler"
        )
        self.pipe = self.pipe.to(self.device)
        self.random_mode = random_mode
        self.target_loss = 3 if not random_mode else None

    def mitigate(
        self,
        prompt,
        num_inference_steps,
        seed,
    ):
        gt_prompt = prompt

        if not self.random_mode:
            prompt = prompt_augmentation(
                gt_prompt,
                "rand_word_add",
                tokenizer=self.pipe.tokenizer,
                repeat_num=4,
            )

        if self.target_loss is not None:
            set_random_seed(seed)
            auged_prompt_embeds = self.pipe.aug_prompt(
                prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=7.5,
                num_images_per_prompt=1,
                target_steps=[0],
                lr=0.05,
                optim_iters=10,
                target_loss=self.target_loss,
            )

            ### generation
            set_random_seed(seed)
            outputs = self.pipe(
                prompt_embeds=auged_prompt_embeds,
                num_inference_steps=num_inference_steps,
                guidance_scale=7.5,
                num_images_per_prompt=1,
            )

        else:
            outputs = self.pipe(
                prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=7.5,
                num_images_per_prompt=1,
            )

        return outputs.images[0]
