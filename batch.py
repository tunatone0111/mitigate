import hashlib
import json

from mitigate.prompts import load_prompts


def make_batch_jsonl(
    prompts,
    system_prompt,
    user_template,
    model="gpt-4o",
    seed=42,
    output_path="openai_batch.jsonl",
):
    """
    Write a JSONL file for OpenAI batch completion API.
    Each line is a JSON object with 'messages', 'model', and 'seed'.
    """
    with open(output_path, "w") as f:
        for prompt in prompts:
            h = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            prompt_user = f"{user_template}\n\n### Input: {prompt}\n### Output: \n"
            obj = {
                "custom_id": h,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt_user},
                    ],
                    "seed": seed,
                },
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


# Example usage:
if __name__ == "__main__":
    with open("mitigate/methods/ours/gpt/template_mem_system.txt", "r") as f:
        system_prompt = "\n".join(f.readlines())
    with open("mitigate/methods/ours/gpt/template_mem_user.txt", "r") as f:
        user_template = "\n".join(f.readlines())

    prompts = load_prompts("memorized_500")

    # Chunk prompts into batches of 100
    chunk_size = 100
    for i in range(0, len(prompts), chunk_size):
        chunk_prompts = prompts[i : i + chunk_size]
        chunk_output_path = f"openai_batch_memorized_42_chunk{i // chunk_size}.jsonl"

        make_batch_jsonl(
            prompts=chunk_prompts,
            system_prompt=system_prompt,
            user_template=user_template,
            model="gpt-4o",
            seed=42,
            output_path=chunk_output_path,
        )
        print(f"Created batch {i // chunk_size} with {len(chunk_prompts)} prompts")
