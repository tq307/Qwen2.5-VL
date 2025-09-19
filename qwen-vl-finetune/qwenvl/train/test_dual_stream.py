"""Evaluate Qwen2.5-VL with or without the dual-stream enhancements."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import torch
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
)

from qwenvl.models import Qwen2_5_VLDualStreamForConditionalGeneration


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--data_file", type=str, required=True, help="JSON/JSONL annotations")
    parser.add_argument(
        "--output_file", type=str, default=None, help="Optional path to dump per-sample predictions"
    )
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Use the original Qwen2.5-VL model without the dual-stream modules.",
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_samples", type=int, default=None)
    return parser.parse_args()


def load_dataset(path: Path) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fp:
        if path.suffix.lower() in {".jsonl", ".jsonl.gz"}:
            for line in fp:
                if not line.strip():
                    continue
                samples.append(json.loads(line))
        else:
            data = json.load(fp)
            if isinstance(data, list):
                samples = data
            else:
                raise ValueError("Expected a list of samples in the dataset file")
    return samples


def normalise_text(text: str) -> str:
    return " ".join(text.strip().lower().split())


def run_evaluation(args: argparse.Namespace) -> None:
    path = Path(args.data_file)
    data = load_dataset(path)
    if args.max_samples is not None:
        data = data[: args.max_samples]

    processor = AutoProcessor.from_pretrained(args.model_name_or_path)
    if args.baseline:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model_name_or_path)
    else:
        model = Qwen2_5_VLDualStreamForConditionalGeneration.from_pretrained(args.model_name_or_path)
    model.to(args.device)
    model.eval()

    results: List[Dict[str, Any]] = []
    correct = 0

    for sample in tqdm(data, desc="evaluating"):
        image = Image.open(sample["image"]).convert("RGB")
        question = sample["question"].strip()
        answer = normalise_text(sample["answer"])

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question},
                ],
            }
        ]
        prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(
            text=[prompt], images=[image], return_tensors="pt", padding=True
        ).to(args.device)

        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=processor.tokenizer.eos_token_id,
            )

        output_ids = generated[:, inputs["input_ids"].shape[-1] :]
        prediction = processor.tokenizer.batch_decode(
            output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )[0]
        prediction_norm = normalise_text(prediction)
        is_correct = prediction_norm == answer
        correct += int(is_correct)

        gate_stats = None
        if hasattr(model, "model") and getattr(model.model, "last_gating_metadata", None):
            metadata = model.model.last_gating_metadata or {}
            gate_values = metadata.get("gate")
            if gate_values is not None and gate_values.numel() > 0:
                gate_values = gate_values.squeeze(-1).cpu()
                gate_stats = {
                    "mean": gate_values.mean().item(),
                    "min": gate_values.min().item(),
                    "max": gate_values.max().item(),
                }

        results.append(
            {
                "question": question,
                "answer": sample["answer"],
                "prediction": prediction.strip(),
                "correct": bool(is_correct),
                "gate_stats": gate_stats,
            }
        )

    accuracy = correct / max(1, len(results))
    print(f"Accuracy: {accuracy * 100:.2f}% ({correct}/{len(results)})")

    if args.output_file:
        out_path = Path(args.output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as fp:
            for item in results:
                fp.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    run_evaluation(parse_args())
