"""QLoRA training entry point (``make train``). Config: configs/finetune.yaml.

Stack: Unsloth + Hugging Face TRL SFTTrainer, PEFT, bitsandbytes
(4-bit NF4 base + LoRA rank 16, alpha 16). Install the training stack with
``pip install -e ".[finetune]"`` plus Unsloth per https://docs.unsloth.ai.
"""

from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> int:
    """Train the LoRA adapter and save it to ``output_dir`` for publishing."""
    parser = argparse.ArgumentParser(description="QLoRA fine-tune Mistral-7B-Instruct")
    parser.add_argument("--config", default="configs/finetune.yaml")
    parser.parse_args(argv)
    raise NotImplementedError("Training not implemented yet")


if __name__ == "__main__":
    raise SystemExit(main())
