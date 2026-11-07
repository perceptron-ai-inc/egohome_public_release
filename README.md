# EgoHome: prompts and metrics

Evaluation code for **EgoHome-2h**, 137 human-annotated egocentric household-chore clips (1.99 h, 1920×1080, 30 fps) with four annotation types on the same videos.

| Task | Column | Unit | Metric |
|---|---|---|---|
| Hand boxes | `hand_boxes` | left / right box per frame, 30 fps | centroid hit fraction |
| Subtasks | `subtasks` | open-vocabulary segments | permissible subsegment Jaccard (LLM label judge) |
| Hand actions | `hand_actions` | left / right segments, Ego4D FHO verbs (118) | permissible subsegment Jaccard (verb class) |
| Hand captions | `hand_captions` | left / right open-vocabulary segments | permissible subsegment Jaccard (LLM label judge) |

Dataset: [`PerceptronAI/EgoHome`](https://huggingface.co/datasets/PerceptronAI/EgoHome)

## Install

```bash
pip install "git+https://github.com/perceptron-ai-inc/egohome_public_release#egg=egohome[hub]"
```

No dependencies beyond the standard library. `[hub]` adds `huggingface_hub` to fetch the gold annotations.

## Data format

One row per video. Times are seconds from the clip start. Boxes are `[x_min, y_min, x_max, y_max]` normalized to 0–1000 (x over width, y over height).

```json
{
  "clip_id": "ec3b25ecc793_001",
  "duration_sec": 60.0, "fps": 30, "width": 1920, "height": 1080,
  "hand_boxes": [{"t": 0.0, "left": [238.5, 349.1, 401.0, 661.1], "right": null}, ...],
  "subtasks": [{"start": 0.0, "end": 1.74, "label": "placing teal plate on cabinet shelf"}, ...],
  "hand_actions": {"left": [{"start": 0.0, "end": 13.31, "label": "hold_(support,_grip,_grasp)"}, ...], "right": [...]},
  "hand_captions": {"left": [{"start": 0.0, "end": 13.31, "label": "holding the dishwasher basket"}, ...], "right": [...]}
}
```

## Evaluate

Write one JSON object per clip, in the same schema. Include only the tasks you predict.

```bash
export GEMINI_API_KEY=...            # label judge for subtasks and hand captions
python -m egohome.evaluate --predictions preds.jsonl
python -m egohome.evaluate --predictions preds.jsonl --no-judge   # hand boxes + hand actions only
```

The prompts we use for every task are in [`egohome/prompts.py`](egohome/prompts.py). Each one asks for exactly this JSON.

Results and metric definitions below match the [EgoHome blog post](https://www.perceptron.inc/blog/egohome).

## Evaluation metrics

### Hand box metric: centroid hit fraction

To measure hand box prediction accuracy, we simply ask each model to produce timestamped bounding boxes for the hand locations throughout the video. For every frame with a human-annotated hand, we check whether the centroid of the model’s box for that hand falls inside the human box, then divide by the number of annotated hand-frames, plus any boxes the model predicts outside the annotated range.

Models that sample fewer frames per second are still scored on every annotated frame: between two of a model’s boxes, we linearly interpolate its prediction. A model sampled at 1 fps therefore only loses credit where the hand moves off the straight-line path between its samples.

### Subtask, action, and hand caption metric: permissible subsegment Jaccard

We developed our subtask correctness metric while attempting to find a fair way to calibrate our annotators. We initially used a metric similar to WGO-Bench’s semantic F1, which performs bipartite matching between the target and prediction and then calculating F1 between the predicted and target sets.

Bipartite matching, which is common in object segmentation benchmarks, allows each ground truth segment to be paired with exactly one predicted segment. In other words, during scoring, the mapping between predicted and ground truth is one-to-one.

We’ve found, however, that there are many valid ways to divide up the timeline in a video. Even if you show the same annotator the same video, in one sitting they may break washing the dishes down into a single task or they may break it down into many smaller segments.

Rather than encouraging models to overfit to one particular choice of temporal granularity, we opted to allow the ground truth and predicted segments to match to each other in a many-to-many mapping, and we’ve tailored our LLM as judge prompt to permit predicted tasks to be subtasks of the ground truth or vice versa.

We call our metric **permissible subsegment Jaccard.**

### Worked example: the stairs

Let’s say the ground truth labeled the time span 0 to 5s as one segment, “sweeping the stairs”.

The table below demonstrates how various predictions would be graded under our many-to-many permissible subsegment Jaccard, vs the one-to-one WGO-Bench semantic F1.

| Model output | Permissible subsegment Jaccard | WGO-Bench semantic F1 |
|---|---|---|
| A. “sweeping the stairs” 0–5 s | (5 + 5) / (5 + 5) = **1.0** | IoU 1.0, labels match: 1 match → **1.0** |
| B. “sweeping the top stair” 0–2 s, “sweeping the bottom stair” 2–5 s | both pieces agree: (2 + 3 + 5) / (5 + 5) = **1.0** | IoUs 0.4 and 0.6, both below 0.75: 0 matches of 2 predicted, 1 gold → **0.0** |
| C. “sweeping the stairs” 0–3 s, “putting the broom away” 3–5 s | first piece agrees: (3 + 3) / (5 + 5) = **0.6** | IoU 0.6, below 0.75 → **0.0** |
| D. “carrying the broom” 0–5 s | judge says it’s a different action → **0.0** | timing matches but the label doesn’t → **0.0** |

**Table 2.** Four model outputs for one human segment, “sweeping the stairs” from 0 to 5 s, scored by permissible subsegment Jaccard and WGO-Bench semantic F1. Solid segments agree with the human label under the judge; dashed segments do not.

While calibrating annotators, we frequently ran into Case B: a valid, more detailed breakdown gets a score of zero under one-to-one matching and full credit under ours. Case C shows we still give partial credit for the part the model got right, where a fixed IoU cutoff gives nothing. (WGO-Bench’s semantic F1 pairs segments one-to-one, counts a pair only when its temporal IoU is at least 0.75 and the judge accepts the labels).

We found that this metric adhered much more closely to subjective measures of fairness when evaluating human annotators’ performance during calibration, thus we opted to release it as our official metric for the benchmark.

The formal definitions of both metrics are in the [appendix](#metric-formulas), and the scoring code is in our [GitHub repository](https://github.com/perceptron-ai-inc/egohome_public_release).

## How different models stack up

We compare Perceptron Egocentric, at its default 4 fps and an 8 fps option, with GPT-6 Astra, Gemini 3.8 Flash and Elorian’s Glimpse, each at 4 fps and 1 fps, and with Pantheon’s Argus and NVIDIA’s FLORA. Subtasks, hand actions and hand captions are scored with permissible subsegment Jaccard, and hand boxes with centroid hit fraction; higher is better for both.

The Perceptron Egocentric harness matches or beats every other system on subtask labeling and is close to GPT-6 Astra on hand boxes, but trails it on hand actions and captions. Generally speaking, most systems struggle with identifying individual hand activities, with the 3 hand-related portions of this benchmark quite far from being saturated.

### EgoHome-2h (137 clips)

| System | Subtask ↑ | Hand action ↑ | Hand caption ↑ | Hand box ↑ | Subtask $ / video-hour ↓ |
|---|---:|---:|---:|---:|---:|
| Perceptron Egocentric, 4 fps (default) | 0.787 | 0.421 | 0.613 | 0.942 | $9 |
| Perceptron Egocentric, 8 fps (option) | **0.798** | 0.421 | 0.626 | 0.973 | ≈ $15 |
| GPT-6 Astra, 4 fps frames | 0.797 | **0.498** | **0.749** | **0.979** | ≈ $76 |
| GPT-6 Astra, 1 fps frames | 0.775 | 0.479 | 0.724 | 0.865 | ≈ $20 |
| Gemini 3.8 Flash, 4 fps video | 0.663 | 0.405 | 0.558 | 0.444 | **≈ $1.3** |
| Gemini 3.8 Flash, 1 fps video | 0.639 | 0.390 | 0.557 | 0.522 | n/r |
| Glimpse (Elorian), 4 fps video | 0.689 | 0.390 | 0.560 | 0.489 | ≈ $2.8 |
| Glimpse (Elorian), 1 fps video | 0.590 | 0.353 | 0.560 | 0.247 | n/r |
| Argus (Pantheon), GPT-6 Astra | 0.367 | 0.336 | 0.593 | — | $24.17 |
| Argus (Pantheon), Midcentury's settings: GPT-6.1 Sol | 0.315 | 0.327 | 0.590 | — | $7.76 |
| FLORA (NVIDIA) | 0.747 | — | — | — | $35.39 |

**Table 3.** EgoHome-2h results by system. Permissible subsegment Jaccard (subtask, hand action, hand caption) and centroid hit fraction (hand box). Best in bold. — = the system does not produce that annotation. n/r = not reported for that setting. Inputs: GPT-6 Astra as timestamped frames (768 px), Gemini and Glimpse as native video. All scores cover all 137 clips; Perceptron results are averaged across three runs at 4 fps and two runs at 8 fps, and other systems were run once. Subtask $ uses Perceptron's customer price, direct API charges, or FLORA's GPU runtime cost; Argus costs also include hand labels, and costs marked ≈ are estimates of the subtask share of a run billed across all tasks.

### Fine-grained results by gold segment length

Our benchmark is the only one with sufficient scale of annotations necessary to offer performance breakdowns at varying timescales. This reveals that specialist systems like NVIDIA's FLORA hold up much better at shorter timescales than some of the other models which fare well on the overall metric.

**Subtask**

| System | All ↑ | < 0.5 s ↑ | 0.5–1 s ↑ | > 1 s ↑ |
|---|---:|---:|---:|---:|
| Perceptron Egocentric, 4 fps (default) | 0.787 | 0.559 | 0.615 | 0.814 |
| Perceptron Egocentric, 8 fps (option) | **0.798** | 0.555 | **0.672** | 0.824 |
| GPT-6 Astra, 4 fps frames | 0.797 | 0.503 | 0.649 | **0.831** |
| GPT-6 Astra, 1 fps frames | 0.775 | **0.595** | 0.500 | 0.807 |
| Gemini 3.8 Flash, 4 fps video | 0.663 | 0.375 | 0.392 | 0.695 |
| Gemini 3.8 Flash, 1 fps video | 0.639 | 0.361 | 0.320 | 0.670 |
| Glimpse (Elorian), 4 fps video | 0.689 | 0.380 | 0.410 | 0.720 |
| Glimpse (Elorian), 1 fps video | 0.590 | 0.328 | 0.279 | 0.616 |
| Argus (Pantheon), GPT-6 Astra | 0.367 | 0.135 | 0.133 | 0.381 |
| Argus (Pantheon), Midcentury's settings: GPT-6.1 Sol | 0.315 | 0.098 | 0.106 | 0.332 |
| FLORA (NVIDIA) | 0.747 | 0.579 | 0.615 | 0.777 |
| *Clips with a step of this length* | 137 | 24 | 60 | 137 |

**Table 4.** Subtask score by the length of the gold segment. Best in bold.

**Hand action**

| System | All ↑ | < 0.5 s ↑ | 0.5–1 s ↑ | > 1 s ↑ |
|---|---:|---:|---:|---:|
| Perceptron Egocentric, 4 fps (default) | 0.421 | 0.298 | 0.286 | 0.461 |
| GPT-6 Astra, 4 fps frames | **0.498** | **0.422** | **0.415** | **0.531** |
| GPT-6 Astra, 1 fps frames | 0.479 | 0.341 | 0.355 | 0.520 |
| Gemini 3.8 Flash, 4 fps video | 0.405 | 0.206 | 0.273 | 0.435 |
| Gemini 3.8 Flash, 1 fps video | 0.390 | 0.191 | 0.252 | 0.436 |
| Glimpse (Elorian), 4 fps video | 0.390 | 0.193 | 0.212 | 0.433 |
| Glimpse (Elorian), 1 fps video | 0.353 | 0.212 | 0.185 | 0.392 |
| Argus (Pantheon), GPT-6 Astra | 0.336 | 0.321 | 0.268 | 0.358 |
| Argus (Pantheon), Midcentury's settings: GPT-6.1 Sol | 0.327 | 0.262 | 0.205 | 0.355 |

**Table 5.** Hand action score by the length of the gold segment. Best in bold.

**Hand caption**

| System | All ↑ | < 0.5 s ↑ | 0.5–1 s ↑ | > 1 s ↑ |
|---|---:|---:|---:|---:|
| Perceptron Egocentric, 4 fps (default) | 0.613 | 0.447 | 0.446 | 0.678 |
| GPT-6 Astra, 4 fps frames | **0.749** | **0.560** | **0.642** | **0.830** |
| GPT-6 Astra, 1 fps frames | 0.724 | 0.415 | 0.575 | 0.800 |
| Gemini 3.8 Flash, 4 fps video | 0.558 | 0.343 | 0.379 | 0.618 |
| Gemini 3.8 Flash, 1 fps video | 0.557 | 0.295 | 0.383 | 0.619 |
| Glimpse (Elorian), 4 fps video | 0.560 | 0.281 | 0.351 | 0.625 |
| Glimpse (Elorian), 1 fps video | 0.560 | 0.264 | 0.337 | 0.631 |
| Argus (Pantheon), GPT-6 Astra | 0.593 | 0.414 | 0.490 | 0.647 |
| Argus (Pantheon), Midcentury's settings: GPT-6.1 Sol | 0.590 | 0.433 | 0.441 | 0.648 |

**Table 6.** Hand caption score by the length of the gold segment. Best in bold.

## Appendix

### Metric formulas

#### Centroid hit fraction

Mathematically this can be expressed as

$$
\operatorname{CHF} = \frac{\displaystyle\sum_{(f,h) \in G} \mathbb{1}\left[\operatorname{Centroid}(\hat{p}_{f,h}) \in g_{f,h}\right]}{|G| + |P_{\text{out}}|}
$$

Where G is the set of (frame, hand) pairs with a human box at 30 fps and g(f,h) is that box. p̂(f,h) is the model’s box for hand h at frame f: its own box at that frame, or the linear interpolation between its two nearest boxes; frames outside the span the model covered count as misses. P_out is the set of predicted boxes outside the frames the human labels cover, and the indicator is 1 when the centroid of p̂(f,h) falls inside g(f,h).

#### Permissible subsegment Jaccard

It can be expressed mathematically as follows:

$$
\operatorname{PSJ}(P, T) = \frac{\displaystyle\sum_{p \in P} m(p, T) + \sum_{t \in T} m(t, P)}{\displaystyle\sum_{p \in P} |I(p)| + \sum_{t \in T} |I(t)|}
$$

$$
m(s, S) = \Bigl|\, I(s) \cap \bigcup_{\substack{s' \in S \\ J(s, s') = 1}} I(s') \,\Bigr|
$$

Let P and T be the predicted and ground-truth segments. Each segment s has a time interval I(s) and a label, and |·| is duration. J(s,s′) ∈ {0,1} is the LLM judge’s verdict on whether two labels are permissible matches: the same action, or one a subtask of the other. m(s,S) is the portion of s’s duration that overlaps some segment in S that the judge accepts.

### Judge prompt

For subtasks and hand captions, an LLM judge (Gemini 3.5 Flash) decides whether two overlapping labels describe the same activity; hand actions are matched on their Ego4D verb. This is the judge’s prompt for subtasks. For hand captions, “subtask descriptions” reads “hand activity captions”. The prompts each model receives are in our [GitHub repository](https://github.com/perceptron-ai-inc/egohome_public_release).

```text
You are grading a human annotator against a gold reference for egocentric video subtask descriptions. The two labels describe time segments that overlap; decide if they are talking about the SAME underlying activity.
Gold label: "<gold label>"
Annotator label: "<predicted label>"
Be lenient about WORDING, OBJECTS, and SURFACES: match=true across synonyms, paraphrases, different objects/tools/surfaces, and finer-vs-coarser detail, PROVIDED the same physical hand ACTION is being performed (e.g. "vacuuming the stairs with the hose" ~ "vacuuming the floor" — both vacuuming; "wash the dishes" ~ "scrub the plate" — both washing/scrubbing).
But the ACTION itself must be the same: return match=false when the manipulation differs, EVEN IF the same object is mentioned (e.g. "vacuuming the stairs with the hose" vs "holding the vacuum cleaner" — vacuuming vs holding; "wiping a window" vs "opening a drawer"), or when one is empty/placeholder.
A label that names no specific physical action never matches a specific one: generic descriptions such as "doing chores", "cleaning up", "working in the kitchen", "doing the task" or "using the hands" are match=false against any specific activity.
When genuinely ambiguous, lean match=false.
Return only JSON: {"match": true|false, "rationale": "short reason"}
```

### Degenerate baselines

In order to ensure the metric isn’t trivially gameable, we chose a few degenerate baselines and determined how they'd be rated by the metric. We chose three types of generic baseline:

1. Choosing a generic label like "doing chores" and applying it to the whole timeline
2. Choosing a generic label and applying at fixed 2.5s windows
3. Taking the most common/longest true clip from the ground truth and filling the whole video timeline with it as a single task.

| Answer (no video used) | Subtask | Hand action | Hand caption |
|---|---:|---:|---:|
| One segment for the whole clip, generic label | 0.000 | 0.175 | 0.000 |
| Fixed 2.5 s windows, generic label | 0.000 | 0.175 | 0.000 |
| One segment for the whole clip, the clip’s own most common true label | 0.528 | 0.449 | 0.499 |

**Table 8.** Scores of three degenerate answers that use no video.

Generic labels earn nothing on subtasks and captions. A constant “hold” earns 0.175 on hand actions, the share of time hands really are holding. These degenerate baselines represent a floor of how good a trivial baseline is at the task. The table below shows which models exceed the trivial baseline on the three tasks.

| System | Subtask | Hand action | Hand caption |
|---|---:|---:|---:|
| Perceptron Egocentric, 4 fps (default) | 0.787 ✓ | 0.421 ✗ | 0.613 ✓ |
| Perceptron Egocentric, 8 fps (option) | 0.798 ✓ | 0.421 ✗ | 0.626 ✓ |
| GPT-6 Astra, 4 fps frames | 0.797 ✓ | 0.498 ✓ | 0.749 ✓ |
| GPT-6 Astra, 1 fps frames | 0.775 ✓ | 0.479 ✓ | 0.724 ✓ |
| Gemini 3.8 Flash, 4 fps video | 0.663 ✓ | 0.405 ✗ | 0.558 ✓ |
| Gemini 3.8 Flash, 1 fps video | 0.639 ✓ | 0.390 ✗ | 0.557 ✓ |
| Glimpse (Elorian), 4 fps video | 0.689 ✓ | 0.390 ✗ | 0.560 ✓ |
| Glimpse (Elorian), 1 fps video | 0.590 ✓ | 0.353 ✗ | 0.560 ✓ |
| Argus (Pantheon), GPT-6 Astra | 0.367 ✗ | 0.336 ✗ | 0.593 ✓ |
| Argus (Pantheon), Midcentury's settings: GPT-6.1 Sol | 0.315 ✗ | 0.327 ✗ | 0.590 ✓ |
| FLORA (NVIDIA) | 0.747 ✓ | — | — |
| *Best trivial baseline* | 0.528 | 0.449 | 0.499 |

**Table 9.** Which systems exceed the best trivial baseline on each task. ✓ exceeds the best trivial baseline for that task; ✗ does not. n/r = not reported for that setting. — = the system does not produce that annotation.

### Annotation statistics

#### Subtasks

The distribution of durations is below:

| Subtask segments | EgoHome |
|---|---:|
| Segments | 1,924 |
| Shortest | 0.16 s |
| Median | 2.50 s |
| Mean | 3.57 s |
| 90th percentile | 7.34 s |
| Longest | 55.1 s |

**Table 10.** Subtask segment statistics.

| Duration | EgoHome |
|---|---:|
| < 0.5 s | 30 (1.6%) |
| 0.5–1 s | 120 |
| 1–2 s | 564 |
| 2–5 s | 829 |
| 5–10 s | 277 |
| > 10 s | 104 |

**Table 11.** Subtask segments by duration.

#### Hand boxes

Here is the percentage of time the right and left hand are on screen in the corpus:

| Hand on screen | % of frames |
|---|---:|
| Left | 96.7% |
| Right | 96.6% |
| Both | 94.3% |
| Either | 99.0% |
| Neither | 1.0% |
| Boxes | 417,912 on 216,225 frames |

**Table 12.** Share of frames with each hand on screen.

#### Hand actions

| Hand actions | Left | Right |
|---|---|---|
| Segments per clip | 17.3 | 19.4 |
| Median duration | 1.45 s | 1.34 s |
| Mean duration | 2.49 s | 2.39 s |
| < 0.5 s | 171 | 248 |
| < 1 s | 731 | 939 |
| Distinct labels (both hands) | 47 |  |
| Video where left and right have the same action | 42.7% |  |

**Table 13.** Hand action segments by hand.

| Most frequent verbs | Segments |
|---|---:|
| hold | 802 |
| take | 615 |
| put | 595 |
| fold | 427 |
| move | 310 |
| arrange | 257 |
| adjust | 227 |
| smooth | 214 |
| lift | 207 |
| wash | 163 |

**Table 14.** The ten most frequent hand action verbs.

#### Hand captions

| Hand captions | Left | Right |
|---|---|---|
| Segments per clip | 18.4 | 20.1 |
| Median duration | 1.55 s | 1.49 s |
| Mean duration | 2.50 s | 2.38 s |
| < 0.5 s | 145 | 179 |
| < 1 s | 707 | 844 |
| Distinct labels (both hands) | 3,948 |  |
| Video where left and right have the same caption | 29.3% |  |

**Table 15.** Hand caption segments by hand.

## Citation

```bibtex
@misc{perceptron2026egohome,
  title={{EgoHome}: the world's largest, multi-annotation egocentric benchmark to date},
  author={{Perceptron}},
  year={2026},
  month={nov},
  url={https://www.perceptron.inc/blog/egohome}
}
```

## Acknowledgements

The raw footage for this data was collected by [HumanSignal](https://humansignal.com/), a human data provider specializing in high-quality data collection and annotation for AI teams.

## References

1. Dyna Robotics. Dyna-2: A 1-Million-Hour Scaling Law for World-Action Models. August 2026. [dyna.co/dyna-2](https://www.dyna.co/dyna-2)

2. Macrodata Labs. Segmenting Robot Video into Actionable Subtasks. June 2026. [macrodata.co/blog/annotating-robot-video-subtasks](https://macrodata.co/blog/annotating-robot-video-subtasks)

3. K. Grauman, A. Westbury, E. Byrne, et al. Ego4D: Around the World in 3,000 Hours of Egocentric Video. CVPR 2022. [arXiv:2110.07058](https://arxiv.org/abs/2110.07058)

## Layout

```
egohome/metrics.py    centroid hit fraction, permissible subsegment Jaccard (+ length buckets)
egohome/ego4d.py      the 118 Ego4D FHO verbs and an alias/inflection normalizer
egohome/prompts.py    task prompts and the label-judge prompt
egohome/judge.py      Gemini label judge (cached); any (gold, pred) -> bool callable works
egohome/evaluate.py   CLI
tests/                  metric tests
```
