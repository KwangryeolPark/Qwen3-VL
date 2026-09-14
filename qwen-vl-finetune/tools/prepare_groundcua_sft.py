#!/usr/bin/env python3
"""Convert the prepared GroundCUA EasyR1 parquet into Qwen3-VL SFT JSONL.

The converter intentionally reuses the same GUI-grounding prompt and tool-call
output protocol used by the GroundCUA EasyR1/GRPO pipeline.  This keeps the SFT
checkpoint directly compatible with the later EasyR1 RL stage.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


DEFAULT_PROJECT_ROOT = Path("/home/kwangryeol/workspace/Qwen3-8B-Instruct")
DEFAULT_INPUT = DEFAULT_PROJECT_ROOT / "datasets/groundcua/easyr1/train.parquet"
DEFAULT_OUTPUT = DEFAULT_PROJECT_ROOT / "datasets/groundcua/sft/qwen3vl_train.jsonl"

PROMPT_TEMPLATE = """<image>
You are a GUI grounding assistant.

Given the screenshot and the user's instruction, identify the UI element that should be clicked.

The screen coordinate system is normalized to 1000 x 1000:
- x ranges from 0 to 1000 from left to right.
- y ranges from 0 to 1000 from top to bottom.

Return exactly one tool call in the following format:

<tool_call>
{\"name\":\"computer_use\",\"arguments\":{\"action\":\"left_click\",\"coordinate\":[x,y]}}
</tool_call>

The coordinate should be near the center of the target UI element.

User instruction:
{problem}"""


def _parse_answer(value: Any) -> tuple[int, int]:
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, dict):
        raise ValueError("answer must be a JSON object")

    center = value.get("center")
    if center is None:
        bbox = value.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            raise ValueError("answer must contain center=[x,y] or bbox=[x1,y1,x2,y2]")
        center = [(float(bbox[0]) + float(bbox[2])) / 2.0, (float(bbox[1]) + float(bbox[3])) / 2.0]

    if not isinstance(center, (list, tuple)) or len(center) != 2:
        raise ValueError("center must contain exactly two coordinates")

    x, y = float(center[0]), float(center[1])
    if not (0.0 <= x <= 1000.0 and 0.0 <= y <= 1000.0):
        raise ValueError(f"center outside normalized coordinate range: {(x, y)}")

    return int(round(x)), int(round(y))


def _image_path(value: Any) -> str:
    if isinstance(value, str):
        path = value
    elif isinstance(value, (list, tuple)) and len(value) == 1:
        path = str(value[0])
    else:
        raise ValueError("images must be a path or a single-item path list")

    image = Path(path)
    if not image.is_absolute():
        raise ValueError(f"expected an absolute image path, got: {image}")
    if not image.is_file():
        raise FileNotFoundError(str(image))
    return str(image)


def _tool_call(x: int, y: int) -> str:
    payload = {
        "name": "computer_use",
        "arguments": {"action": "left_click", "coordinate": [x, y]},
    }
    return f"<tool_call>{json.dumps(payload, separators=(',', ':'))}</tool_call>"


def convert(input_path: Path, output_path: Path, limit: int | None, strict: bool) -> dict[str, Any]:
    parquet = pq.ParquetFile(input_path)
    available = set(parquet.schema.names)
    required = {"problem", "images", "answer"}
    missing = required - available
    if missing:
        raise ValueError(f"missing required parquet columns: {sorted(missing)}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped = 0
    errors: dict[str, int] = {}

    columns = ["problem", "images", "answer"]
    if "id" in available:
        columns.append("id")

    with output_path.open("w", encoding="utf-8") as out:
        stop = False
        for batch in parquet.iter_batches(batch_size=4096, columns=columns):
            data = batch.to_pydict()
            size = len(data["problem"])
            for i in range(size):
                if limit is not None and written >= limit:
                    stop = True
                    break
                try:
                    problem = str(data["problem"][i]).strip()
                    if not problem:
                        raise ValueError("empty problem")
                    image = _image_path(data["images"][i])
                    x, y = _parse_answer(data["answer"][i])
                    record = {
                        "image": image,
                        "conversations": [
                            {"from": "human", "value": PROMPT_TEMPLATE.format(problem=problem)},
                            {"from": "gpt", "value": _tool_call(x, y)},
                        ],
                    }
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    written += 1
                except Exception as exc:
                    name = f"{type(exc).__name__}: {exc}"
                    errors[name] = errors.get(name, 0) + 1
                    skipped += 1
                    if strict:
                        raise
            if stop:
                break

    manifest = {
        "dataset": "GroundCUA",
        "format": "Qwen3-VL SFT JSONL",
        "source_parquet": str(input_path.resolve()),
        "output_jsonl": str(output_path.resolve()),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "rows_written": written,
        "rows_skipped": skipped,
        "limit": limit,
        "coordinate_system": "normalized 0..1000; target is rounded bbox center",
        "output_protocol": "computer_use left_click tool_call",
        "errors": errors,
    }
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=None, help="Write only the first N valid rows (smoke testing).")
    parser.add_argument("--strict", action="store_true", help="Stop on the first invalid row instead of skipping it.")
    args = parser.parse_args()

    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    manifest = convert(args.input, args.output, args.limit, args.strict)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
