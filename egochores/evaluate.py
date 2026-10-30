"""Score predictions on EgoChores.

    python -m egochores.evaluate --predictions preds.jsonl [--gold PerceptronAI/EgoChores | metadata.jsonl]

``preds.jsonl``: one JSON object per clip with ``clip_id`` and any of the dataset's annotation columns
(``hand_boxes``, ``subtasks``, ``hand_actions``, ``hand_captions``) in the same schema. A clip missing from the file,
or a task missing from a clip, scores 0 for that task.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from egochores.metrics import (
    LENGTH_BUCKETS,
    as_segments,
    centroid_hit_fraction,
    ego4d_verb_accept,
    label_pairs,
    permissible_subsegment_jaccard,
)

TASKS = ("hand_boxes", "subtasks", "hand_actions", "hand_captions")


def load_gold(source: str) -> list[dict]:
    if Path(source).exists():
        return [json.loads(line) for line in Path(source).read_text().splitlines() if line.strip()]
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(source, "data/test/metadata.jsonl", repo_type="dataset")
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def tracks(task: str, value) -> dict:
    if task == "subtasks":
        return {"subtask": as_segments(value)}
    value = value or {}
    return {side: as_segments(value.get(side)) for side in ("left", "right")}


def evaluate(gold: list[dict], preds: dict[str, dict], judge=None) -> dict:
    """Mean per-clip score for each task, with the per-gold-length breakdown for the segment tasks."""
    out = {}
    for task in TASKS:
        if task != "hand_boxes" and task != "hand_actions" and judge is None:
            continue
        if task in ("subtasks", "hand_captions"):
            judge.decide(
                task,
                set().union(
                    *(
                        label_pairs(tracks(task, g[task]), tracks(task, preds.get(g["clip_id"], {}).get(task)))
                        for g in gold
                    )
                ),
            )
        scores, buckets = [], {name: [] for name, _, _ in LENGTH_BUCKETS}
        for g in gold:
            p = preds.get(g["clip_id"], {}).get(task)
            if task == "hand_boxes":
                scores.append(centroid_hit_fraction(g[task], p or [], g["fps"]) if p is not None else 0.0)
                continue
            accept = ego4d_verb_accept if task == "hand_actions" else judge.accept(task)
            score, by_len = permissible_subsegment_jaccard(tracks(task, g[task]), tracks(task, p), accept)
            scores.append(score)
            for name, v in by_len.items():
                if v is not None:
                    buckets[name].append(v)
        out[task] = {"score": sum(scores) / len(scores), "clips": len(scores)}
        if task != "hand_boxes":
            out[task]["by_gold_segment_length"] = {k: (sum(v) / len(v) if v else None) for k, v in buckets.items()}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--gold", default="PerceptronAI/EgoChores")
    ap.add_argument("--judge-model", default=None, help="Gemini model for the label judge (default gemini-3.5-flash)")
    ap.add_argument("--no-judge", action="store_true", help="score only hand boxes and hand actions (no LLM calls)")
    args = ap.parse_args()
    gold = load_gold(args.gold)
    preds = {}
    for line in Path(args.predictions).read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            preds[row["clip_id"]] = row
    judge = None
    if not args.no_judge:
        from egochores.judge import DEFAULT_MODEL, GeminiJudge

        judge = GeminiJudge(args.judge_model or DEFAULT_MODEL)
    print(json.dumps(evaluate(gold, preds, judge), indent=2))


if __name__ == "__main__":
    main()
