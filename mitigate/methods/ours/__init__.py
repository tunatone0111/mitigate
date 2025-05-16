import hashlib
import json
import os
import re
import time

import requests
import torch
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline
from PIL import Image

from ..mitigation_method import MitigationMethod


def GPT4_Rare2Frequent(prompt, cache_idx, cache_dir, seed):
    if os.path.exists(f"{cache_dir}/{cache_idx}_{seed}.txt"):
        with open(f"{cache_dir}/{cache_idx}_{seed}.txt", "r") as f:
            result = f.read()
        try:
            parse_weighted_prompt_format(result)
            return result
        except Exception:
            pass

    url = "https://api.openai.com/v1/chat/completions"

    dirname = os.path.dirname(__file__)
    with open(os.path.join(dirname, "gpt/template_mem_system.txt"), "r") as f:
        template_system = f.readlines()
        prompt_system = " ".join(template_system)

    with open(os.path.join(dirname, "gpt/template_mem_user.txt"), "r") as f:
        template_user = f.readlines()
        template_user = " ".join(template_user)

    prompt_user = f"### Input: {prompt}\n### Output: \n"
    prompt_user = f"{template_user}\n\n{prompt_user}"

    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
        "User-Agent": "Apifox/1.0.0 (https://apifox.com)",
        "Content-Type": "application/json",
    }

    success = False
    trial = 0

    while not success and trial < 5:
        try:
            payload = json.dumps(
                {
                    "model": "gpt-4o",
                    "messages": [
                        {"role": "system", "content": prompt_system},
                        {"role": "user", "content": prompt_user},
                    ],
                    "seed": seed + 10 * trial,
                }
            )
            response = requests.request("POST", url, headers=headers, data=payload)
            obj = response.json()
            text = obj["choices"][0]["message"]["content"]
            prompts, step_ranges, weights = parse_weighted_prompt_format(text)
            success = True
        except Exception as e:
            print(e)
            trial += 1
            time.sleep(10)
    time.sleep(1)

    with open(f"{cache_dir}/{cache_idx}_{seed}.txt", "w") as f:
        f.write(text)

    return text


def parse_weighted_prompt_format(r2f_prompt: str):
    """GPT 출력 문자열에서 프롬프트, 스텝 범위, 가중치 파싱"""
    lines = r2f_prompt.strip().split("\n")
    prompts, step_ranges, weights = [], [], []

    for line in lines:
        line = line.strip()
        # '### Output:' 또는 공백 줄은 무시
        if not line or line.startswith("###"):
            continue
        match = re.match(
            r"\d+\.\s+(.*?)(?:,\s*(\d+)[–-](\d+))?(?:,\s*weight\s*=\s*(\d\.\d+))?$",
            line.strip(),
        )
        if match:
            text, start, end, weight = match.groups()
            prompts.append(text.strip())
            if start and end:
                step_ranges.append(list(range(int(start), int(end))))
            else:
                step_ranges.append(None)
            weights.append(float(weight) if weight else None)
        else:
            raise ValueError(f"Line could not be parsed: {line}")

    return prompts, step_ranges, weights


def weighted_average_embeddings(
    pipe, prompts, weights, device, do_cfg, negative_prompt, num_images_per_prompt
):
    """weight 기반으로 여러 프롬프트 임베딩 평균"""
    total_weight = sum(w for w in weights if w is not None)
    if total_weight == 0:
        raise ValueError("Total weight is zero.")

    weighted_embed = None
    for prompt, weight in zip(prompts, weights):
        if weight is None:
            continue
        embed = pipe._encode_prompt(
            prompt, device, num_images_per_prompt, do_cfg, negative_prompt
        )
        if weighted_embed is None:
            weighted_embed = embed * weight
        else:
            weighted_embed += embed * weight

    return weighted_embed / total_weight


@torch.no_grad()
def sample_weighted_prompt(
    pipe,
    prompts,
    step_ranges,
    weights,
    num_inference_steps=50,
    guidance_scale=7.5,
    device="cuda",
    generator=None,
    negative_prompt="Poor anatomy, Unclear, Cut-off, Distorted, Copy, Mistake, Additional arms, Extra legs, Unpleasant proportions, Long neck, Low-grade, Low resolution, Lack of arms legs, Unhealthy, Genetic variation, Beyond the frame, Inserted text, Unattractive, Lowest quality",
    num_images_per_prompt=1,
    do_cfg=True,
) -> Image.Image:
    # weighted average embedding (step_range=None)
    weighted_embed = weighted_average_embeddings(
        pipe,
        [p for p, s in zip(prompts, step_ranges) if s is None],
        [w for w, s in zip(weights, step_ranges) if s is None],
        device,
        do_cfg,
        negative_prompt,
        num_images_per_prompt,
    )

    # 마지막 prompt embedding (step_range != None인 마지막 것 하나만 있다고 가정)
    final_idx = next(
        i for i, s in reversed(list(enumerate(step_ranges))) if s is not None
    )
    final_embed = pipe._encode_prompt(
        prompts[final_idx], device, num_images_per_prompt, do_cfg, negative_prompt
    )
    final_steps = step_ranges[final_idx]

    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    latents = torch.randn((1, 4, 64, 64), device=device, generator=generator)
    latents *= pipe.scheduler.init_noise_sigma

    for i, t in enumerate(pipe.scheduler.timesteps):
        if i in final_steps:
            cond = final_embed
        else:
            cond = weighted_embed

        latent_input = torch.cat([latents] * 2) if do_cfg else latents
        latent_input = pipe.scheduler.scale_model_input(latent_input, t)

        noise_pred = pipe.unet(latent_input, t, encoder_hidden_states=cond).sample

        if do_cfg:
            noise_uncond, noise_text = noise_pred.chunk(2)
            noise_pred = noise_uncond + guidance_scale * (noise_text - noise_uncond)

        latents = pipe.scheduler.step(noise_pred, t, latents)[0]

    image = pipe.decode_latents(latents)
    return pipe.numpy_to_pil(image)


class Ours(MitigationMethod):
    def __init__(self, gpt_only=False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model_id = "CompVis/stable-diffusion-v1-4"
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id, safety_checker=None
        ).to(self.device)
        pipe.scheduler = DPMSolverMultistepScheduler.from_pretrained(
            model_id, subfolder="scheduler"
        )

        self.pipe = pipe
        self.gpt_only = gpt_only

    def mitigate(self, prompt, num_inference_steps, seed):
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
        output_dir = os.path.join(os.path.dirname(__file__), "gpt_cache")

        os.makedirs(output_dir, exist_ok=True)

        gpt_output = GPT4_Rare2Frequent(
            prompt,
            cache_idx=prompt_hash,
            cache_dir=output_dir,
            seed=seed,
        )

        if self.gpt_only:
            return

        try:
            prompts, step_ranges, weights = parse_weighted_prompt_format(gpt_output)
        except Exception as e:
            print(e)
            return

        image = sample_weighted_prompt(
            pipe=self.pipe,
            prompts=prompts,
            step_ranges=step_ranges,
            weights=weights,
            num_inference_steps=num_inference_steps,
            guidance_scale=7.5,
            generator=torch.Generator(device=self.device).manual_seed(seed),
            device=self.device,
        )[0]

        return image
