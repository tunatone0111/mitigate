import datetime
import glob
import hashlib
import json
import os
from typing import Literal

import fire
import open_clip
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import tqdm
from PIL import Image
from prompts import load_prompts


def measure_CLIP_similarity(images, prompt, model, clip_preprocess, tokenizer, device):
    with torch.no_grad():
        img_batch = [clip_preprocess(i).unsqueeze(0) for i in images]
        img_batch = torch.concatenate(img_batch).to(device)
        image_features: torch.Tensor = model.encode_image(img_batch)
        text = tokenizer([prompt]).to(device)
        text_features: torch.Tensor = model.encode_text(text)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        return (image_features @ text_features.T).mean(-1)


### credit: https://github.com/somepago/DCR
def measure_SSCD_similarity(gt_images, images, model, device):
    ret_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    gt_images = torch.stack([ret_transform(x.convert("RGB")) for x in gt_images]).to(
        device
    )
    images = torch.stack([ret_transform(x.convert("RGB")) for x in images]).to(device)
    with torch.no_grad():
        feat_1 = model(gt_images).clone()
        feat_1 = nn.functional.normalize(feat_1, dim=1, p=2)
        feat_2 = model(images).clone()
        feat_2 = nn.functional.normalize(feat_2, dim=1, p=2)
        return torch.mm(feat_1, feat_2.T)


def main(
    method: Literal["ours", "cvpr25", "iclr25", "iclr24", "random"],
    num_inference_steps=100,
    dataset: Literal["match_verbatim", "memorized_500", "normal"] = "memorized_500",
    image_dir: str = "output",
    output_file: str = "results.jsonl",
    cache_dir: str = "cache",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sim_model = torch.jit.load(
        os.path.join(cache_dir, "sscd_disc_large.torchscript.pt")
    ).to(device)

    ref_model, _, ref_clip_preprocess = open_clip.create_model_and_transforms(
        "ViT-g-14",
        pretrained="laion2b_s12b_b42k",
        device=device,
        cache_dir=cache_dir,
    )
    ref_tokenizer = open_clip.get_tokenizer("ViT-g-14")

    memorized_prompts = load_prompts(dataset, with_index=True)

    ###

    with tqdm.tqdm(total=len(memorized_prompts)) as pbar:
        for i, (memorized_prompt_idx, memorized_prompt) in enumerate(memorized_prompts):
            pbar.update(1)
            pbar.set_description(f"Processing {i:03d}")

            if dataset == "normal":
                for filename in glob.glob(
                    f"{image_dir}/{dataset}/steps{num_inference_steps}/method{method}/{i:03d}_seed*.png"
                ):
                    seed = filename.split("_seed")[1].split(".")[0]
                    gen_image = Image.open(filename)

                    clip = measure_CLIP_similarity(
                        [gen_image],
                        memorized_prompt,
                        ref_model,
                        ref_clip_preprocess,
                        ref_tokenizer,
                        device,
                    )

                    timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

                    with open(output_file, "a") as f:
                        f.write(
                            json.dumps(
                                dict(
                                    prompt_idx=i,
                                    method=method,
                                    seed=seed,
                                    clip_gen=clip[0].tolist(),
                                    clip_gt=0,
                                    sscd_max=0,
                                    sscd_min=0,
                                    sscd_mean=0,
                                    timestamp=timestamp,
                                    prompt_hash=hashlib.sha256(
                                        memorized_prompt.encode("utf-8")
                                    ).hexdigest(),
                                )
                            )
                            + "\n"
                        )
                continue

            gt_images = []
            for filename in glob.glob(
                f"{cache_dir}/gt/gt_images/{memorized_prompt_idx:03d}/*.png"
            ):
                im = Image.open(filename)
                gt_images.append(im)

            assert len(gt_images) > 0, f"No gt images found for {i:03d}"

            for filename in glob.glob(
                f"{image_dir}/{dataset}/steps{num_inference_steps}/method{method}/{i:03d}_seed*.png"
            ):
                seed = filename.split("_seed")[1].split(".")[0]
                gen_image = Image.open(filename)

                sscd = measure_SSCD_similarity(
                    gt_images, [gen_image], sim_model, device
                )
                gt_image = gt_images[sscd.argmax(dim=0)[0].item()]

                clip = measure_CLIP_similarity(
                    [gt_image, gen_image],
                    memorized_prompt,
                    ref_model,
                    ref_clip_preprocess,
                    ref_tokenizer,
                    device,
                )

                timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

                with open(output_file, "a") as f:
                    f.write(
                        json.dumps(
                            dict(
                                prompt_idx=i,
                                method=method,
                                seed=seed,
                                clip_gen=clip[1].tolist(),
                                clip_gt=clip[0].tolist(),
                                sscd_max=sscd.max().item(),
                                sscd_min=sscd.min().item(),
                                sscd_mean=sscd.mean().item(),
                                timestamp=timestamp,
                                prompt_hash=hashlib.sha256(
                                    memorized_prompt.encode("utf-8")
                                ).hexdigest(),
                            )
                        )
                        + "\n"
                    )


if __name__ == "__main__":
    fire.Fire(main)
