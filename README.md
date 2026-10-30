# EgoChores: prompts and metrics

Evaluation code for **EgoChores-2h**, 137 human-annotated egocentric household-chore clips (1.99 h, 1920×1080, 30 fps) with four annotation types on the same videos.

| Task | Column | Unit | Metric |
|---|---|---|---|
| Hand boxes | `hand_boxes` | left / right box per frame, 30 fps | centroid hit fraction |
| Subtasks | `subtasks` | open-vocabulary segments | permissible subsegment Jaccard (LLM label judge) |
| Hand actions | `hand_actions` | left / right segments, Ego4D FHO verbs (118) | permissible subsegment Jaccard (verb class) |
| Hand captions | `hand_captions` | left / right open-vocabulary segments | permissible subsegment Jaccard (LLM label judge) |

Dataset: [`PerceptronAI/EgoChores`](https://huggingface.co/datasets/PerceptronAI/EgoChores)

## Install

```bash
pip install "git+https://github.com/perceptron-ai-inc/egochores_public_release#egg=egochores[hub]"
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
python -m egochores.evaluate --predictions preds.jsonl
python -m egochores.evaluate --predictions preds.jsonl --no-judge   # hand boxes + hand actions only
```

The prompts we use for every task are in [`egochores/prompts.py`](egochores/prompts.py). Each one asks for exactly this JSON.

## Metrics

### Hand boxes: centroid hit fraction

For each hand $h$, the prediction is a box track $\hat b_h(f)$, linearly interpolated between the predicted frames $f$ and undefined outside them. Let $\mathcal G$ be the gold (frame, hand) pairs with gold box $b_{f,h}$, and let $\mathcal E$ be the predicted boxes outside the time range of their hand's gold track. Then

$$
\mathrm{CHF} \;=\; \frac{\big|\{(f,h)\in\mathcal G \;:\; \hat b_h(f)\ \text{defined} \;\wedge\; c(\hat b_h(f)) \in \operatorname{int} b_{f,h}\}\big|}{|\mathcal G| + |\mathcal E|},
\qquad c(b) = \Big(\tfrac{x_1+x_2}{2}, \tfrac{y_1+y_2}{2}\Big).
$$

Because of the interpolation the score does not depend on the prediction's frame rate: a 1 fps track scores the same as a 30 fps one wherever the hand moves linearly between samples.

### Subtasks, hand actions, hand captions: permissible subsegment Jaccard

On a track $k$ (the single subtask track, or each hand), gold segments $g$ and predicted segments $p$ *agree* when they overlap and their labels are accepted as the same activity:

$$
A(g,p) = \mathbb 1\big[\,|g\cap p|>0 \;\wedge\; J(\ell_g,\ell_p)\,\big]
$$

$J$ is the LLM judge for subtasks and captions. For actions, $J$ checks that both labels normalize to the same Ego4D verb class, and a verb outside the set never matches.

$$
\mathrm{PSJ} \;=\; \frac{\displaystyle\sum_k\Big(\sum_{p\in P_k}\Big|\bigcup_{g:\,A(g,p)} g\cap p\Big| \;+\; \sum_{g\in G_k}\Big|\bigcup_{p:\,A(g,p)} g\cap p\Big|\Big)}{\displaystyle\sum_k\Big(\sum_{p\in P_k}|p| \;+\; \sum_{g\in G_k}|g|\Big)}
$$

Each predicted second is credited when some gold segment covering it agrees, and each gold second likewise. A prediction that splits a gold segment into finer, equally valid steps, or merges several into one, still scores 1. Label errors and timing errors both cost time on both sides.

**By segment length.** Restricting $G_k$ to gold segments of a given length (< 0.5 s, 0.5–1 s, > 1 s) and clipping predictions to those spans gives the per-length scores reported below.

## Results

See the dataset card for the full table, including models at different frame rates.

## Layout

```
egochores/metrics.py    centroid hit fraction, permissible subsegment Jaccard (+ length buckets)
egochores/ego4d.py      the 118 Ego4D FHO verbs and an alias/inflection normalizer
egochores/prompts.py    task prompts and the label-judge prompt
egochores/judge.py      Gemini label judge (cached); any (gold, pred) -> bool callable works
egochores/evaluate.py   CLI
tests/                  metric tests
```
