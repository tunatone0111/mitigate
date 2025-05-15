import os
from typing import Literal

import fire
from dotenv import load_dotenv

from .methods import CVPR25, ICLR24, ICLR25, Ours
from .prompts import load_prompts


def main(
    method: Literal["ours", "cvpr25", "iclr25", "iclr24", "random"],
    num_inference_steps=100,
    seed=42,
    dataset: Literal["match_verbatim", "memorized_500"] = "match_verbatim",
    offset: int = 0,
    limit: int = 0,
    output_dir: str = "output",
):
    load_dotenv()

    assert os.path.exists(output_dir), f"Output directory {output_dir} does not exist"
    output_dir = os.path.join(
        output_dir,
        f"{dataset}",
        f"steps{num_inference_steps}",
        f"method{method}",
    )

    os.makedirs(output_dir, exist_ok=True)

    if method == "ours":
        mitigation = Ours()
    elif method == "cvpr25":
        mitigation = CVPR25()
    elif method == "iclr25":
        mitigation = ICLR25()
    elif method == "iclr24":
        mitigation = ICLR24()
    elif method == "random":
        mitigation = ICLR24(random_mode=True)
    else:
        raise ValueError(f"Invalid method: {method}")

    prompts = load_prompts(dataset, offset, limit)

    for i, prompt in enumerate(prompts):
        image = mitigation.mitigate(prompt, num_inference_steps, seed)
        image.save(os.path.join(output_dir, f"{offset + i:03d}_seed{seed}.png"))


if __name__ == "__main__":
    fire.Fire(main)
