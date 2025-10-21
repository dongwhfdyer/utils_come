"""
Build complete evaluator prompts from a captions JSONL file.

This script does NOT call the judge. It only renders text prompts you can
paste into a console/UI to test size/behavior.

Modes:
  - multi: one prompt per sample id including all models' captions
  - single: one prompt per (sample id, model) pair

Usage:
  python build_eval_prompts.py \
    --input /path/to/captions_50.jsonl \
    --outdir /path/to/prompts \
    --mode multi   # or single
"""

import argparse
import json
import os
from typing import Any, Dict, List


HEADER_MULTI = (
    "You are an impartial evaluator specializing in industrial audio. Given the same audio features and multiple model captions, score EACH model on a 0–10 scale.\n\n"
    "Criteria (equal weight):\n"
    "1) Coverage of ALL feature groups in the features JSON\n"
    "2) Technical correctness and consistency with the numbers\n"
    "3) Specificity/utility for distinguishing sounds\n"
    "4) Clarity/conciseness\n\n"
    "Return STRICT JSON only:\n"
    "{\n  \"scores\": {\"<model>\": <number>, ...},\n  \"rationales\": {\"<model>\": \"<1–2 sentences>\", ...},\n  \"winner\": \"<model>\"\n}\n\n"
)

HEADER_SINGLE = (
    "You are an impartial evaluator for industrial audio. Score how well the caption matches the given features (0–10).\n\n"
    "Criteria (equal weight): coverage, correctness, specificity, clarity.\n\n"
    "Output STRICT JSON only:\n{\"score\": <number>, \"rationale\": \"<1–2 sentences>\"}\n\n"
)


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def build_prompts_multi(rows: List[Dict[str, Any]], outdir: str):
    by_id: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        sid = str(r.get('id'))
        by_id.setdefault(sid, []).append(r)

    os.makedirs(outdir, exist_ok=True)
    count = 0
    for sid, items in by_id.items():
        features = items[0].get('features', {}) if items else {}
        model_to_caption: Dict[str, str] = {}
        for it in items:
            m = it.get('model') or '<default>'
            c = it.get('caption', '')
            if m in model_to_caption and c:
                model_to_caption[m] = model_to_caption[m] + "\n\n---\n\n" + c
            else:
                model_to_caption[m] = c

        prompt = (
            HEADER_MULTI
            + "Audio features (JSON):\n"
            + json.dumps(features, ensure_ascii=False, indent=2)
            + "\n\nCaptions to evaluate (JSON: { \"<model>\": \"<caption>\", ... }):\n"
            + json.dumps(model_to_caption, ensure_ascii=False, indent=2)
            + "\n"
        )

        path = os.path.join(outdir, f"prompt_multi_{sid}.txt")
        with open(path, 'w') as f:
            f.write(prompt)
        count += 1

    print(f"Wrote {count} multi-model prompts to {outdir}")


def build_prompts_single(rows: List[Dict[str, Any]], outdir: str):
    os.makedirs(outdir, exist_ok=True)
    count = 0
    for r in rows:
        sid = str(r.get('id'))
        model = r.get('model') or '<default>'
        features = r.get('features', {})
        caption = r.get('caption', '')

        prompt = (
            HEADER_SINGLE
            + "Features (JSON):\n"
            + json.dumps(features, ensure_ascii=False, indent=2)
            + "\n\nCaption:\n"
            + caption
            + "\n"
        )

        path = os.path.join(outdir, f"prompt_single_{sid}_{model.replace('/', '_')}.txt")
        with open(path, 'w') as f:
            f.write(prompt)
        count += 1

    print(f"Wrote {count} single-model prompts to {outdir}")


def main():
    parser = argparse.ArgumentParser(description="Build evaluator prompts from captions JSONL")
    parser.add_argument("--input", type=str, required=True, help="Path to captions JSONL (e.g., captions_50.jsonl)")
    parser.add_argument("--outdir", type=str, required=True, help="Output directory for prompts")
    parser.add_argument("--mode", type=str, default="multi", choices=["multi", "single"], help="Prompt mode")
    args = parser.parse_args()

    rows = load_jsonl(args.input)
    if args.mode == "multi":
        build_prompts_multi(rows, args.outdir)
    else:
        build_prompts_single(rows, args.outdir)


if __name__ == "__main__":
    main()


