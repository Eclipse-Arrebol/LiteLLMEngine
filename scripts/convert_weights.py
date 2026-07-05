#!/usr/bin/env python3

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from safetensors.torch import safe_open


FORMAT_VERSION = "litellm_engine_weights_v1"

"""
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1

python scripts/convert_weights.py \
  --model Qwen/Qwen3-0.6B \
  --output /root/rivermind-data/Qwen_Qwen3-0.6B/converted_weights \
  --clean
"""
def is_local_path(model: str) -> bool:
    return Path(model).exists()


def resolve_model_dir(model: str, local_files_only: bool) -> Path:
    path = Path(model)

    if path.exists():
        return path.resolve()

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required when --model is a HuggingFace repo id. "
            "Install it with: pip install huggingface_hub"
        ) from exc

    print(f"Downloading safetensors from HuggingFace repo: {model}")

    downloaded = snapshot_download(
        repo_id=model,
        allow_patterns=[
            "*.safetensors",
            "*.safetensors.index.json",
            "model.safetensors.index.json",
            "config.json",
        ],
        local_files_only=local_files_only,
    )

    return Path(downloaded).resolve()


def find_safetensors_files(model_dir: Path) -> List[Path]:
    files = sorted(model_dir.glob("*.safetensors"))

    if not files:
        raise RuntimeError(f"No .safetensors files found in: {model_dir}")

    return files


def sanitize_filename(tensor_name: str) -> str:
    return tensor_name.replace("/", "__") + ".bin"


def torch_dtype_to_output_dtype(dtype: torch.dtype) -> str:
    # 当前 C++ 侧 Linear / Embedding / RMSNorm 都只支持 FP32。
    # 所以这里统一把浮点权重转成 fp32。
    if dtype.is_floating_point:
        return "fp32"

    if dtype == torch.int32:
        return "int32"

    raise RuntimeError(f"Unsupported tensor dtype: {dtype}")


def tensor_to_output_tensor(tensor: torch.Tensor) -> Tuple[torch.Tensor, str]:
    output_dtype = torch_dtype_to_output_dtype(tensor.dtype)

    if output_dtype == "fp32":
        return tensor.detach().cpu().to(torch.float32).contiguous(), "fp32"

    if output_dtype == "int32":
        return tensor.detach().cpu().to(torch.int32).contiguous(), "int32"

    raise RuntimeError(f"Unsupported output dtype: {output_dtype}")


def write_tensor_bin(path: Path, tensor: torch.Tensor) -> None:
    array = tensor.numpy()

    with open(path, "wb") as f:
        f.write(array.tobytes(order="C"))


def collect_tensor_sources(safetensors_files: List[Path]) -> Dict[str, Path]:
    tensor_sources: Dict[str, Path] = {}

    for file_path in safetensors_files:
        with safe_open(str(file_path), framework="pt", device="cpu") as f:
            for key in f.keys():
                if key in tensor_sources:
                    raise RuntimeError(
                        "Duplicate tensor name found in safetensors files:\n"
                        f"  tensor: {key}\n"
                        f"  first file: {tensor_sources[key]}\n"
                        f"  second file: {file_path}"
                    )

                tensor_sources[key] = file_path

    return tensor_sources


def convert_weights(
    model_dir: Path,
    output_dir: Path,
    clean: bool,
) -> None:
    safetensors_files = find_safetensors_files(model_dir)

    print("Found safetensors files:")
    for path in safetensors_files:
        print(f"  {path}")

    if output_dir.exists() and clean:
        print(f"Removing existing output directory: {output_dir}")
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    tensor_sources = collect_tensor_sources(safetensors_files)

    print(f"Found tensors: {len(tensor_sources)}")

    index = {
        "format": FORMAT_VERSION,
        "source_model_dir": str(model_dir),
        "tensors": [],
    }

    converted_count = 0
    total_bytes = 0

    for tensor_name in sorted(tensor_sources.keys()):
        source_file = tensor_sources[tensor_name]

        with safe_open(str(source_file), framework="pt", device="cpu") as f:
            tensor = f.get_tensor(tensor_name)

        out_tensor, out_dtype = tensor_to_output_tensor(tensor)

        filename = sanitize_filename(tensor_name)
        bin_path = output_dir / filename

        write_tensor_bin(bin_path, out_tensor)

        nbytes = bin_path.stat().st_size
        total_bytes += nbytes

        index["tensors"].append(
            {
                "name": tensor_name,
                "filename": filename,
                "dtype": out_dtype,
                "shape": list(out_tensor.shape),
                "source_file": str(source_file),
                "nbytes": nbytes,
            }
        )

        converted_count += 1

        print(
            f"[{converted_count}/{len(tensor_sources)}] "
            f"{tensor_name} -> {filename}, "
            f"shape={list(out_tensor.shape)}, dtype={out_dtype}, bytes={nbytes}"
        )

    index_path = output_dir / "weights_index.json"

    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)

    print()
    print("Conversion finished")
    print(f"  output_dir: {output_dir}")
    print(f"  index: {index_path}")
    print(f"  tensors: {converted_count}")
    print(f"  total bytes: {total_bytes}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert HuggingFace safetensors weights to LiteLLMEngine raw bin format."
    )

    parser.add_argument(
        "--model",
        required=True,
        help="Local model directory or HuggingFace repo id, e.g. models/Qwen_Qwen3-0.6B or Qwen/Qwen3-0.6B",
    )

    parser.add_argument(
        "--output",
        default=None,
        help="Output directory. Default: <model_dir>/converted_weights",
    )

    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove output directory before conversion.",
    )

    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Do not download from HuggingFace; only use local cache.",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    model_dir = resolve_model_dir(
        model=args.model,
        local_files_only=args.local_files_only,
    )

    if args.output is None:
        output_dir = model_dir / "converted_weights"
    else:
        output_dir = Path(args.output).resolve()

    print(f"model_dir: {model_dir}")
    print(f"output_dir: {output_dir}")

    convert_weights(
        model_dir=model_dir,
        output_dir=output_dir,
        clean=args.clean,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())