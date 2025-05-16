import json
import os
from typing import Literal

import pandas as pd


def load_prompts(
    dataset: Literal["match_verbatim", "memorized_500", "normal"],
    offset: int = 0,
    limit: int = 0,
    with_index: bool = False,
) -> list[str]:
    dirname = os.path.dirname(__file__)
    if dataset == "match_verbatim":
        with open(os.path.join(dirname, "match_verbatim.json"), "r") as f:
            prompts = json.load(f)
    elif dataset == "memorized_500":
        prompts = pd.read_json(
            os.path.join(dirname, "memorized_500.jsonl"), lines=True
        )["caption"].tolist()
        indexes = pd.read_json(
            os.path.join(dirname, "memorized_500.jsonl"), lines=True
        )["index"].tolist()
    elif dataset == "normal":
        prompts = pd.read_csv(
            os.path.join(dirname, "unmemorized_laion_prompts.csv"), delimiter=";"
        )
        prompts = prompts["Caption"].tolist()
        indexes = list(range(len(prompts)))

    if limit > 0:
        prompts = prompts[offset : offset + limit]

    if with_index:
        return list(zip(indexes, prompts))
    else:
        return prompts
