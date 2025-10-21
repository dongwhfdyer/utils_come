"""
Evaluate generated captions using an LLM (gpt-5) as a judge.

Input: captions.jsonl from batch generation (fields: id, model, features, caption)
Output: evaluation JSONL with per-record scores and a summary CSV/JSON with per-model averages.

Usage:
  python evaluate_captions.py \
    --input /path/to/captions.jsonl \
    --output /path/to/eval.jsonl \
    --summary /path/to/summary.json \
    --model gpt-5 \
    --workers 8
"""

import argparse
import json
import os
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from dotenv import load_dotenv
from openai import OpenAI


EVAL_SYSTEM_PROMPT = (
    "You are an impartial evaluator specializing in industrial audio. "
    "Given the same audio features and multiple model captions, score EACH model on a 0-10 scale. "
    "Criteria: (1) coverage of all feature groups; (2) technical correctness; (3) specificity/utility; (4) clarity/conciseness. "
    "Return STRICT JSON with this shape: {"
    "\"scores\": {\"<model>\": <number 0-10>, ...}, "
    "\"rationales\": {\"<model>\": \"short reason\", ...}, "
    "\"winner\": \"<best model name>\"}"
)

EVAL_USER_TEMPLATE = (
    "Audio features (JSON):\n"
    "{features_json}\n\n"
    "Captions to evaluate (per model, same sample):\n"
    "{captions_json}\n"
)


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return items


def save_jsonl(path: str, rows: List[Dict[str, Any]]):
    with open(path, 'w') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def call_judge_batch(client: OpenAI, model: str, features: Dict[str, float], model_to_caption: Dict[str, str]) -> Dict[str, Any]:
    user_content = EVAL_USER_TEMPLATE.format(
        features_json=json.dumps(features, ensure_ascii=False, indent=2),
        captions_json=json.dumps(model_to_caption, ensure_ascii=False, indent=2),
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": EVAL_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        max_tokens=1024,
        temperature=0.0,
    )
    # try to parse JSON from response
    content = None
    try:
        choice0 = resp.choices[0]
        message = getattr(choice0, "message", None) or {}
        content = message.get("content") if isinstance(message, dict) else getattr(message, "content", None)
        if content is None:
            content = getattr(choice0, "text", None)
        if not content:
            return {"scores": {}, "rationales": {}, "winner": None, "raw": ""}
        parsed = json.loads(content)
        scores = parsed.get("scores", {}) or {}
        ration = parsed.get("rationales", {}) or {}
        winner = parsed.get("winner")
        # normalize to floats
        norm_scores = {str(k): float(v) for k, v in scores.items() if isinstance(v, (int, float, str))}
        return {"scores": norm_scores, "rationales": {str(k): str(v) for k, v in ration.items()}, "winner": winner, "raw": content}
    except Exception:
        return {"scores": {}, "rationales": {}, "winner": None, "raw": content if content is not None else ""}


def main():
    parser = argparse.ArgumentParser(description="Evaluate captions with an LLM judge (gpt-5)")
    parser.add_argument("--input", type=str, required=True, help="captions.jsonl from generation")
    parser.add_argument("--output", type=str, required=True, help="output eval jsonl path")
    parser.add_argument("--summary", type=str, required=True, help="per-model summary json path")
    parser.add_argument("--per-sample", type=str, default=None, help="optional JSONL path logging per-sample score map and winner")
    parser.add_argument("--model", type=str, default="gpt-5", help="judge model (default: gpt-5)")
    parser.add_argument("--workers", type=int, default=8, help="parallel workers")
    args = parser.parse_args()

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("DOUABAO_API_KEY")
    base_url = (
        os.getenv("OPENAI_BASE_URL")
        or os.getenv("DOUABAO_BASE_URL")
        or "https://doubao.zwchat.cn/v1"
    )
    client = OpenAI(api_key=api_key, base_url=base_url)

    rows = load_jsonl(args.input)
    print(f"Loaded {len(rows)} caption records from {args.input}")

    # Group by sample id and collate model captions for the same sample
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        sid = str(r.get("id"))
        grouped.setdefault(sid, []).append(r)

    eval_rows: List[Dict[str, Any]] = []
    write_lock = Lock()

    def _task_group(sid: str, items: List[Dict[str, Any]]) -> Tuple[str, Dict[str, Any]]:
        # Use the first item's features (all for same sample)
        features = items[0].get("features", {}) if items else {}
        # Combine captions per model; if multiple entries per model, join with separator
        model_to_caps: Dict[str, str] = {}
        for it in items:
            m = it.get("model") or "<default>"
            cap = it.get("caption", "")
            if m in model_to_caps and cap:
                model_to_caps[m] = model_to_caps[m] + " \n\n---\n\n" + cap
            else:
                model_to_caps[m] = cap
        judged = call_judge_batch(client, args.model, features, model_to_caps)
        return sid, {"features": features, "judged": judged}

    per_sample_rows: List[Dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(_task_group, sid, items) for sid, items in grouped.items()]
        for fut in as_completed(futures):
            sid, payload = fut.result()
            judged = payload["judged"]
            scores: Dict[str, float] = judged.get("scores", {})
            rationales: Dict[str, str] = judged.get("rationales", {})
            winner = judged.get("winner")
            raw = judged.get("raw", "")

            # Console output for each sample inference with clear separators and raw content
            print("=" * 80)
            print(f"Sample: {sid}")
            print("- Raw judge response:")
            print(raw if raw else "<empty>")
            pretty_scores = ", ".join([f"{m}: {scores[m]:.2f}" for m in sorted(scores.keys())])
            print("- Parsed scores:")
            print(pretty_scores if pretty_scores else "<none>")
            print(f"- Winner: {winner}")
            print("=" * 80)

            # Collect per-sample summary row
            per_sample_rows.append({
                "id": sid,
                "scores": scores,
                "winner": winner,
                "raw": raw,
            })
            for model, score in scores.items():
                with write_lock:
                    eval_rows.append({
                        "id": sid,
                        "model": model,
                        "score": float(score),
                        "rationale": rationales.get(model, ""),
                    })

    save_jsonl(args.output, eval_rows)
    print(f"Wrote {len(eval_rows)} eval records to {args.output}")

    # Optionally write per-sample JSONL
    if args.per_sample:
        save_jsonl(args.per_sample, per_sample_rows)
        print(f"Wrote {len(per_sample_rows)} per-sample summaries to {args.per_sample}")

    # compute per-model averages
    per_model: Dict[str, List[float]] = {}
    for e in eval_rows:
        model = e.get("model") or "<default>"
        per_model.setdefault(model, []).append(float(e.get("score", 0.0)))

    summary = {
        m: {
            "count": len(scores),
            "avg": (sum(scores) / len(scores)) if scores else 0.0,
            "min": min(scores) if scores else 0.0,
            "max": max(scores) if scores else 0.0,
        }
        for m, scores in per_model.items()
    }

    with open(args.summary, 'w') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Saved per-model summary to {args.summary}")


if __name__ == "__main__":
    main()


