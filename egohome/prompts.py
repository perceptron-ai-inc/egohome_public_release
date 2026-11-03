"""Task prompts (each answer is JSON in the dataset's own schema) and the segment-label judge prompt.

Coordinates are [x_min, y_min, x_max, y_max] normalized to 0-1000 of the frame; times are seconds from the clip start.
"""

from egohome.ego4d import EGO4D_FHO_VERBS

EGO4D_VERBS = ", ".join(v.partition("_(")[0] for v in EGO4D_FHO_VERBS)

_INTRO = "This is an egocentric (first-person) video of a person doing a household chore. "

HAND_BOX_PROMPT = (
    _INTRO + "Track the LEFT hand and the RIGHT hand through the video. For each moment you report, give the time in "
    "seconds and each hand's bounding box as [x_min, y_min, x_max, y_max] normalized to 0-1000, or null when that hand "
    "is not visible. Return only JSON:\n"
    '[{"t": 0.0, "left": [120, 540, 310, 820], "right": null}, ...]'
)

SUBTASK_PROMPT = (
    _INTRO + "Identify the ordered sequence of subtasks the person performs, with the time range of each in seconds. "
    "Return only JSON:\n"
    '[{"start": 0.0, "end": 5.0, "label": "short subtask description"}, ...]'
)

HAND_ACTION_PROMPT = (
    _INTRO + "For the LEFT hand and the RIGHT hand separately, identify the sequence of manipulation actions each hand "
    "performs, with the time range of each in seconds. Label each action with EXACTLY ONE verb from this Ego4D action "
    "set (use the base verb; never a verb outside the set, e.g. label vacuuming as 'clean', gripping as 'hold'):\n"
    f"{EGO4D_VERBS}\n"
    "Return only JSON:\n"
    '{"left": [{"start": 0.0, "end": 3.0, "label": "hold"}, ...], "right": [{"start": 0.0, "end": 3.0, "label": "clean"}, ...]}'
)

HAND_CAPTION_PROMPT = (
    _INTRO + "For the LEFT hand and the RIGHT hand separately, describe what each hand is doing over time as short "
    "natural-language captions, with the time range of each in seconds. Return only JSON:\n"
    '{"left": [{"start": 0.0, "end": 3.0, "label": "caption"}, ...], "right": [{"start": 0.0, "end": 3.0, "label": "caption"}, ...]}'
)

PROMPTS = {
    "hand_boxes": HAND_BOX_PROMPT,
    "subtasks": SUBTASK_PROMPT,
    "hand_actions": HAND_ACTION_PROMPT,
    "hand_captions": HAND_CAPTION_PROMPT,
}


def judge_prompt(task: str, gold: str, pred: str) -> str:
    """The label judge's prompt for one overlapping (gold, predicted) segment pair. Labels are JSON-quoted."""
    import json

    what = {"hand_captions": "hand activity captions"}.get(task, "subtask descriptions")
    return (
        f"You are grading a human annotator against a gold reference for egocentric video {what}. "
        "The two labels describe time segments that overlap; decide if they are talking about the SAME "
        "underlying activity.\n"
        f"Gold label: {json.dumps(gold)}\n"
        f"Annotator label: {json.dumps(pred)}\n"
        "Be lenient about WORDING, OBJECTS, and SURFACES: match=true across synonyms, paraphrases, different "
        "objects/tools/surfaces, and finer-vs-coarser detail, PROVIDED the same physical hand ACTION is being "
        'performed (e.g. "vacuuming the stairs with the hose" ~ "vacuuming the floor" — both vacuuming; '
        '"wash the dishes" ~ "scrub the plate" — both washing/scrubbing).\n'
        "But the ACTION itself must be the same: return match=false when the manipulation differs, EVEN IF the "
        'same object is mentioned (e.g. "vacuuming the stairs with the hose" vs "holding the vacuum cleaner" — '
        'vacuuming vs holding; "wiping a window" vs "opening a drawer"), or when one is empty/placeholder.\n'
        "A label that names no specific physical action never matches a specific one: generic descriptions such as "
        '"doing chores", "cleaning up", "working in the kitchen", "doing the task" or "using the hands" '
        "are match=false against any specific activity.\n"
        "When genuinely ambiguous, lean match=false.\n"
        'Return only JSON: {"match": true|false, "rationale": "short reason"}'
    )
