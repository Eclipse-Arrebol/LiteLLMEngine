#!/usr/bin/env python3

import argparse
import subprocess
from collections.abc import Mapping
from transformers import AutoTokenizer


def parse_new_token_ids(output: str):
    lines = output.splitlines()

    for i, line in enumerate(lines):
        if line.strip() == "New token ids:":
            if i + 1 >= len(lines):
                raise RuntimeError("Found 'New token ids:' but no ids line")

            ids_line = lines[i + 1].strip()

            if ids_line == "<none>":
                return []

            if not ids_line:
                raise RuntimeError("Empty new token ids line")

            return [int(x.strip()) for x in ids_line.split(",") if x.strip()]

    raise RuntimeError("Could not find 'New token ids:' in LiteLLMEngine output")


def extract_input_ids(encoded):
    """
    兼容这些返回类型：
    1. list[int]
    2. list[list[int]]
    3. torch.Tensor / numpy array
    4. dict / BatchEncoding / BatchFeature-like object:
       {"input_ids": ...}
    """
    if isinstance(encoded, Mapping):
        if "input_ids" not in encoded:
            raise RuntimeError(
                f"encoded object has no input_ids key, keys={list(encoded.keys())}"
            )
        encoded = encoded["input_ids"]

    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()

    if (
        isinstance(encoded, list)
        and len(encoded) > 0
        and isinstance(encoded[0], list)
    ):
        encoded = encoded[0]

    if not isinstance(encoded, list):
        raise RuntimeError(f"Unsupported input_ids type: {type(encoded)}")

    return [int(x) for x in encoded]


def encode_prompt(tokenizer, prompt: str, use_chat_template: bool):
    if not use_chat_template:
        encoded = tokenizer.encode(
            prompt,
            add_special_tokens=False,
        )
        return extract_input_ids(encoded)

    messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]

    try:
        encoded = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_dict=True,
        )
        return extract_input_ids(encoded)

    except TypeError:
        encoded = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
        return extract_input_ids(encoded)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--engine", default="./build/LiteLLMEngine")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-chat-template", action="store_true")
    parser.add_argument("--show-full-text", action="store_true")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
    )

    input_ids = encode_prompt(
        tokenizer=tokenizer,
        prompt=args.prompt,
        use_chat_template=not args.no_chat_template,
    )

    input_ids_text = ",".join(map(str, input_ids))

    eos_token_id = tokenizer.eos_token_id

    cmd = [
        args.engine,
        "--model", args.model,
        "--input-ids", input_ids_text,
        "--max-tokens", str(args.max_tokens),
        "--device", args.device,
        "--temperature", "0",
    ]

    if eos_token_id is not None:
        cmd.extend(["--eos-token-id", str(eos_token_id)])

    print("[run_prompt] input ids:")
    print(input_ids_text)
    print("len =", len(input_ids))
    print()

    result = subprocess.run(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    print("[LiteLLMEngine output]")
    print(result.stdout)

    if result.returncode != 0:
        raise SystemExit(result.returncode)

    new_ids = parse_new_token_ids(result.stdout)

    print("[decoded new text]")
    print(tokenizer.decode(new_ids, skip_special_tokens=True))

    if args.show_full_text:
        full_ids = input_ids + new_ids
        print()
        print("[decoded full text]")
        print(tokenizer.decode(full_ids, skip_special_tokens=True))


if __name__ == "__main__":
    main()