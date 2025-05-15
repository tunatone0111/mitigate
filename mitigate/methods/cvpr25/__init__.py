import numpy as np
import torch
import tqdm
from diffusers import (
    AutoencoderKL,
    DPMSolverMultistepScheduler,
    UNet2DConditionModel,
)  # FlaxKarrasVeOutput
from PIL import Image
from transformers import AutoTokenizer, CLIPTextModel

from ..mitigation_method import MitigationMethod


def insert_rand_word(sentence, word):
    import random

    sent_list = sentence.split(" ")
    sent_list.insert(random.randint(0, len(sent_list)), word)
    new_sent = " ".join(sent_list)
    return new_sent


def prompt_augmentation(prompt, aug_style, tokenizer=None, repeat_num=2):
    if aug_style == "rand_numb_add":
        for i in range(repeat_num):
            randnum = np.random.choice(100000)
            prompt = insert_rand_word(prompt, str(randnum))
    elif aug_style == "rand_word_add":
        for i in range(repeat_num):
            randword = tokenizer.decode(list(np.random.randint(49400, size=1)))
            prompt = insert_rand_word(prompt, randword)
    elif aug_style == "rand_word_repeat":
        wordlist = prompt.split(" ")
        for i in range(repeat_num):
            randword = np.random.choice(wordlist)
            prompt = insert_rand_word(prompt, randword)
    else:
        raise Exception("This style of prompt augmnentation is not written")
    return prompt


class CVPR25(MitigationMethod):
    def __init__(self):
        super().__init__()

        vae = AutoencoderKL.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            subfolder="vae",  # , use_safetensors=True
        )  # stabilityai/stable-diffusion-2-1
        tokenizer = AutoTokenizer.from_pretrained(
            "CompVis/stable-diffusion-v1-4", subfolder="tokenizer", use_fast=False
        )  # CLIPTokenizer.from_pretrained(pretrained_path, subfolder="tokenizer")
        text_encoder = CLIPTextModel.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            subfolder="text_encoder",  # , use_safetensors=True
        )
        unet = UNet2DConditionModel.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            subfolder="unet",  # , use_safetensors=True
        )
        scheduler = DPMSolverMultistepScheduler.from_pretrained(
            "CompVis/stable-diffusion-v1-4", subfolder="scheduler"
        )  # KarrasVeScheduler.from_pretrained("CompVis/stable-diffusion-v1-4")
        # scheduler.set_timesteps(50)
        # scheduler.use_karras_sigmas = True

        torch_device = "cuda"
        vae.to(torch_device)
        text_encoder.to(torch_device)
        unet.to(torch_device)

        self.vae = vae
        self.text_encoder = text_encoder
        self.unet = unet
        self.scheduler = scheduler
        self.tokenizer = tokenizer
        self.torch_device = torch_device

    def mitigate(self, prompt, num_inference_steps, seed):
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )

        with torch.no_grad():
            text_embeddings = self.text_encoder(
                text_input.input_ids.to(self.torch_device)
            )[0]

        max_length = text_input.input_ids.shape[-1]
        uncond_input = self.tokenizer(
            [""] * 1,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        uncond_embeddings = self.text_encoder(
            uncond_input.input_ids.to(self.torch_device)
        )[0]

        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        # scale and decode the image latents with vae
        latents_init = torch.randn(
            (1, self.unet.config.in_channels, 512 // 8, 512 // 8),
            generator=torch.Generator(device=self.torch_device).manual_seed(seed),
            device=self.torch_device,
        )

        self.scheduler.set_timesteps(num_inference_steps)

        # for k in scheduler.timesteps:

        latents = latents_init * self.scheduler.init_noise_sigma

        # scheduler.set_timesteps(num_inference_steps)

        transition_point = -1
        diff_value_prev = -1
        diff_value_prev_prev = -1

        diffs = []

        for t in tqdm.tqdm(self.scheduler.timesteps):
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = torch.cat([latents] * 2)

            latent_model_input = self.scheduler.scale_model_input(
                latent_model_input, timestep=t
            )

            if True:
                s = 0.1
                phi = 1
                tau1 = 500  # 600
                tau2 = 800

                gamma = 0
                if t <= tau1:
                    gamma = 1
                elif t > tau1 and t <= tau2:
                    gamma = (tau2 - t) / float(tau2 - tau1)

                text_embeddings_t = (gamma**0.5) * text_embeddings + s * (
                    (1 - gamma) ** 0.5
                ) * torch.rand(text_embeddings.shape, device=text_embeddings.device)
                text_embeddings_t_rescaled = (
                    text_embeddings_t - torch.mean(text_embeddings_t)
                ) * torch.std(text_embeddings) / torch.std(
                    text_embeddings_t
                ) + torch.mean(text_embeddings)
                text_embeddings_final = (
                    phi * text_embeddings_t_rescaled + (1 - phi) * text_embeddings_t
                )

            else:
                text_embeddings_final = text_embeddings

            # predict the noise residual
            with torch.no_grad():
                noise_pred = self.unet(
                    latent_model_input, t, encoder_hidden_states=text_embeddings_final
                ).sample

            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)

            diff_current = torch.norm(
                noise_pred_uncond - noise_pred_text
            )  # 1- torch.nn.functional.cosine_similarity(torch.flatten(noise_pred_text), torch.flatten(noise_pred_uncond), dim=0) #
            diffs.append(diff_current.item())

            if (
                transition_point == -1
                and diff_current > diff_value_prev
                and diff_value_prev_prev > diff_value_prev
            ):  # len(min_index[0])!=0: # #(diff_value_prev - diff_current) < (diff_value_prev_prev - diff_value_prev):
                transition_point = t
                guidance_scale = 7.5
                CFG_scheduling = "static"

            elif transition_point != -1:
                guidance_scale = 7.5
                CFG_scheduling = "static"
            else:
                guidance_scale = 1.0
                CFG_scheduling = "static"

            diff_value_prev_prev = diff_value_prev
            diff_value_prev = diff_current

            if CFG_scheduling == "invlinear":
                guidance_scale_new = guidance_scale * (t / 1000)
            elif CFG_scheduling == "linear":
                guidance_scale_new = guidance_scale * (1 - (t / 1000))
            elif CFG_scheduling == "cosine":
                pi = (
                    torch.acos(torch.zeros(1)).item() * 2
                )  # which is 3.1415927410125732
                guidance_scale_new = guidance_scale * (
                    torch.cos(pi * t / 1000).item() + 1
                )
            elif CFG_scheduling == "sine":
                pi = (
                    torch.acos(torch.zeros(1)).item() * 2
                )  # which is 3.1415927410125732
                guidance_scale_new = guidance_scale * (
                    torch.sin(pi * (t) / 1000 - pi / 2).item() + 1
                )
            elif CFG_scheduling == "v_shape":
                if t > 500:
                    guidance_scale_new = guidance_scale * (1 - (t / 1000))
                else:
                    guidance_scale_new = guidance_scale * (t / 1000)
            elif CFG_scheduling == "a_shape":
                if t < 500:
                    guidance_scale_new = guidance_scale * (1 - (t / 1000))
                else:
                    guidance_scale_new = guidance_scale * (t / 1000)

            elif CFG_scheduling == "static":
                guidance_scale_new = guidance_scale

            noise_pred = noise_pred_uncond + guidance_scale_new * (
                noise_pred_text - noise_pred_uncond
            )

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        latents = 1.0 / self.vae.config.scaling_factor * latents

        with torch.no_grad():
            image = self.vae.decode(latents).sample[0]

        image = (image / 2 + 0.5).clamp(0, 1).squeeze()
        image = (image.permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()
        image = Image.fromarray(image)

        return image
