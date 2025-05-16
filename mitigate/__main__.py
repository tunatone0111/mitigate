import os
from typing import Literal

import fire
import tqdm
from dotenv import load_dotenv

from .methods import CVPR25, ICLR24, ICLR25, Ours
from .prompts import load_prompts


def main(
    method: Literal[
        "ours",
        "ours-gpt-only",
        "cvpr25",
        "iclr25",
        "iclr25-2",
        "iclr25-3",
        "iclr25-4",
        "iclr25-6",
        "iclr25-8",
        "iclr24",
        "random",
    ],
    num_inference_steps=100,
    seed=42,
    dataset: Literal["match_verbatim", "memorized_500"] = "memorized_500",
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
    elif method == "ours-gpt-only":
        mitigation = Ours(gpt_only=True)
    elif method == "cvpr25":
        mitigation = CVPR25()
    elif method == "iclr25":
        mitigation = ICLR25()
    elif method == "iclr25-2":
        mitigation = ICLR25(num_words=2)
    elif method == "iclr25-3":
        mitigation = ICLR25(num_words=3)
    elif method == "iclr25-4":
        mitigation = ICLR25(num_words=4)
    elif method == "iclr25-6":
        mitigation = ICLR25(num_words=6)
    elif method == "iclr25-8":
        mitigation = ICLR25(num_words=8)
    elif method == "iclr24":
        mitigation = ICLR24()
    elif method == "random":
        mitigation = ICLR24(random_mode=True)
    else:
        raise ValueError(f"Invalid method: {method}")

    prompts = load_prompts(dataset, offset, limit)

    for i, prompt in tqdm.tqdm(enumerate(prompts), total=len(prompts)):
        path_name = os.path.join(output_dir, f"{offset + i:03d}_seed{seed}.png")

        if os.path.exists(path_name):
            continue

        if method == "ours-gpt-only":
            image = mitigation.mitigate(prompt, num_inference_steps, seed)
        else:
            image = mitigation.mitigate(prompt, num_inference_steps, seed)
            image.save(path_name)


if __name__ == "__main__":
    fire.Fire(main)
