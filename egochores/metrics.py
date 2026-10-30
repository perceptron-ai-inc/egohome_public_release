"""EgoChores metrics.

Hand boxes: centroid hit fraction. Each gold (frame, hand) is a hit when the predicted box for that hand at that frame
(linearly interpolated between the prediction's own frames, never extrapolated) has its centroid strictly inside the
gold box. Score = hits / (gold frame-hands + predicted boxes outside the gold track's time range).

Segments (subtask, hand action, hand caption): permissible subsegment Jaccard. A gold/predicted segment pair agrees when
the two overlap in time and their labels are accepted as the same activity (an LLM judge for subtasks and captions,
the Ego4D verb class for actions). Score = (agreeing predicted time + agreeing gold time) / (total predicted time +
total gold time), pooled over both hands for the per-hand tasks. Coarser and finer, but equally valid, segmentations
of the same activity both score 1.
"""

from __future__ import annotations

import bisect
from collections.abc import Callable, Iterable

from egochores.ego4d import normalize_ego4d_fho_verb

Segment = tuple[float, float, str]  # (start_sec, end_sec, label)
Box = tuple[float, float, float, float]  # (x_min, y_min, x_max, y_max), 0-1000
Accept = Callable[[str, str], bool]  # (gold_label, predicted_label) -> same activity?

# Gold-segment length buckets (seconds): (name, lower inclusive, upper exclusive).
LENGTH_BUCKETS: tuple[tuple[str, float, float], ...] = (
    ("lt_0.5s", 0.0, 0.5),
    ("0.5_1s", 0.5, 1.0),
    ("gt_1s", 1.0, float("inf")),
)


# ---------------------------------------------------------------------------
# Segments
# ---------------------------------------------------------------------------
def as_segments(items: Iterable[dict] | None) -> list[Segment]:
    """``[{"start", "end", "label"}]`` -> ``[(start, end, label)]`` (a missing end is a zero-length segment)."""
    out = []
    for s in items or []:
        start = float(s["start"])
        end = float(s.get("end", start))
        out.append((start, max(start, end), str(s.get("label") or "").strip()))
    return out


def _length(s: Segment) -> float:
    return max(0.0, s[1] - s[0])


def _overlap(a: Segment, b: Segment) -> float:
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def _union_length(intervals: list[tuple[float, float]]) -> float:
    if not intervals:
        return 0.0
    intervals = sorted(intervals)
    total, (cs, ce) = 0.0, intervals[0]
    for s, e in intervals[1:]:
        if s <= ce:
            ce = max(ce, e)
        else:
            total += ce - cs
            cs, ce = s, e
    return total + (ce - cs)


def _agreement(gold: list[Segment], pred: list[Segment], accept: Accept) -> tuple[float, float]:
    """(agreeing predicted time + agreeing gold time, total predicted time + total gold time) for one track."""
    total_gold = sum(map(_length, gold))
    total_pred = sum(map(_length, pred))
    if total_gold == 0 or total_pred == 0:
        return 0.0, total_gold + total_pred

    def agrees(g: Segment, p: Segment) -> bool:
        return _overlap(g, p) > 0 and accept(g[2], p[2])

    agree_pred = sum(_union_length([(max(g[0], p[0]), min(g[1], p[1])) for g in gold if agrees(g, p)]) for p in pred)
    agree_gold = sum(_union_length([(max(g[0], p[0]), min(g[1], p[1])) for p in pred if agrees(g, p)]) for g in gold)
    return agree_pred + agree_gold, total_gold + total_pred


def _pooled(gold: dict[str, list[Segment]], pred: dict[str, list[Segment]], accept: Accept) -> float:
    agree = total = 0.0
    for track in set(gold) | set(pred):
        a, t = _agreement(gold.get(track, []), pred.get(track, []), accept)
        agree += a
        total += t
    return 1.0 if total == 0 else agree / total


def label_accept(accept: Accept) -> Accept:
    """Empty labels never agree; case-insensitive identical labels always do; anything else asks ``accept``."""

    def wrapped(gold: str, pred: str) -> bool:
        g, p = gold.strip(), pred.strip()
        if not g or not p:
            return False
        return g.lower() == p.lower() or accept(g, p)

    return wrapped


def ego4d_verb_accept(gold: str, pred: str) -> bool:
    """Hand-action labels agree when both normalize to the same Ego4D verb class; a verb outside the set never does."""
    g, p = normalize_ego4d_fho_verb(gold), normalize_ego4d_fho_verb(pred)
    return g is not None and g == p


def _clip_to(pred: list[Segment], gold: list[Segment]) -> list[Segment]:
    """Predictions restricted to the gold segments' spans (one piece per overlapped gold segment)."""
    return [(max(p[0], g[0]), min(p[1], g[1]), p[2]) for p in pred for g in gold if min(p[1], g[1]) > max(p[0], g[0])]


def permissible_subsegment_jaccard(gold: dict[str, list[Segment]], pred: dict[str, list[Segment]], accept: Accept
                                   ) -> tuple[float, dict[str, float | None]]:
    """The score over all tracks, and the same score restricted to gold segments of each length bucket (predictions
    clipped to those gold spans; None when the clip has no gold segment of that length).

    ``gold`` / ``pred`` map a track ("subtask", or "left" / "right") to its segments."""
    accept = label_accept(accept)
    memo: dict[tuple[str, str], bool] = {}

    def cached(g: str, p: str) -> bool:
        if (g, p) not in memo:
            memo[g, p] = accept(g, p)
        return memo[g, p]

    overall = _pooled(gold, pred, cached)
    buckets: dict[str, float | None] = {}
    for name, lo, hi in LENGTH_BUCKETS:
        gold_b = {k: [g for g in v if lo <= _length(g) < hi] for k, v in gold.items()}
        if not any(gold_b.values()):
            buckets[name] = None
            continue
        buckets[name] = _pooled(gold_b, {k: _clip_to(pred.get(k, []), v) for k, v in gold_b.items()}, cached)
    return overall, buckets


def label_pairs(gold: dict[str, list[Segment]], pred: dict[str, list[Segment]]) -> set[tuple[str, str]]:
    """Every (gold label, predicted label) pair a judge may be asked about (segments overlap on the same track)."""
    return {(g[2], p[2]) for k, golds in gold.items() for g in golds for p in pred.get(k, []) if _overlap(g, p) > 0}


# ---------------------------------------------------------------------------
# Hand boxes
# ---------------------------------------------------------------------------
def _box_at(series: list[tuple[int, Box]], frames: list[int], f: int) -> Box | None:
    """The predicted box at frame ``f``: exact, else linear interpolation between the two neighbouring predicted
    frames; None outside the prediction's own frame range."""
    if not series or f < series[0][0] or f > series[-1][0]:
        return None
    i = bisect.bisect_left(frames, f)
    pf, pbox = series[i]
    if pf == f:
        return pbox
    (af, abox), (bf, bbox) = series[i - 1], series[i]
    w = (f - af) / (bf - af)
    return tuple(a + w * (b - a) for a, b in zip(abox, bbox))  # type: ignore[return-value]


def _centroid_inside(pred: Box, gold: Box) -> bool:
    cx, cy = (pred[0] + pred[2]) / 2, (pred[1] + pred[3]) / 2
    return gold[0] < cx < gold[2] and gold[1] < cy < gold[3]


def _track(boxes: Iterable[dict], side: str, fps: float) -> list[tuple[int, Box]]:
    """``[{"t", "left", "right"}]`` -> sorted ``[(frame index, box)]`` for one hand (frame = round(t * fps))."""
    track = {}
    for b in boxes or []:
        box = b.get(side)
        if box:
            x1, y1, x2, y2 = map(float, box)
            track[round(float(b["t"]) * fps)] = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
    return sorted(track.items())


def centroid_hit_fraction(gold: list[dict], pred: list[dict], fps: float) -> float:
    """Hand-box score for one clip. ``gold`` / ``pred`` are ``[{"t": sec, "left": box|None, "right": box|None}]``."""
    hits = misses = extra = 0
    for side in ("left", "right"):
        g, p = _track(gold, side, fps), _track(pred, side, fps)
        p_frames = [f for f, _ in p]
        for f, gbox in g:
            pbox = _box_at(p, p_frames, f)
            if pbox is not None and _centroid_inside(pbox, gbox):
                hits += 1
            else:
                misses += 1
        if g:
            extra += sum(1 for f in p_frames if f < g[0][0] or f > g[-1][0])
        else:
            extra += len(p)
    total = hits + misses + extra
    return 1.0 if total == 0 else hits / total
