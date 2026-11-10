# EgoHome: prompts and metrics

Evaluation code for **EgoHome-2h**, 137 human-annotated egocentric household-chore clips (1.99 h, 1920×1080, 30 fps) with four annotation types on the same videos.

| Task | Column | Unit | Metric |
|---|---|---|---|
| Hand boxes | `hand_boxes` | left / right box per frame, 30 fps | centroid hit fraction |
| Subtasks | `subtasks` | open-vocabulary segments | permissible subsegment Jaccard (LLM label judge) |
| Hand actions | `hand_actions` | left / right segments, Ego4D FHO verbs (118) | permissible subsegment Jaccard (verb class) |
| Hand captions | `hand_captions` | left / right open-vocabulary segments | permissible subsegment Jaccard (LLM label judge) |

Dataset: [`PerceptronAI/EgoHome`](https://huggingface.co/datasets/PerceptronAI/EgoHome)

**License:** code MIT; dataset CC BY-NC 4.0, with evaluation use permitted (see [License](#license)).

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

## Metrics

### Hand boxes: centroid hit fraction

For each hand $h$, the prediction is a box track $\hat b_h(f)$, linearly interpolated between the predicted frames $f$ and undefined outside them. Let $\mathcal G$ be the gold (frame, hand) pairs with gold box $b_{f,h}$, and let $\mathcal E$ be the predicted boxes outside the time range of their hand's gold track. Then

```math
\mathrm{CHF} \;=\; \frac{\big|\{(f,h)\in\mathcal G \;:\; \hat b_h(f)\ \text{defined} \;\wedge\; c(\hat b_h(f)) \in \mathrm{int}\, b_{f,h}\}\big|}{|\mathcal G| + |\mathcal E|},
\qquad c(b) = \Big(\tfrac{x_1+x_2}{2}, \tfrac{y_1+y_2}{2}\Big).
```

Because of the interpolation the score does not depend on the prediction's frame rate: a 1 fps track scores the same as a 30 fps one wherever the hand moves linearly between samples.

### Subtasks, hand actions, hand captions: permissible subsegment Jaccard

On a track $k$ (the single subtask track, or each hand), gold segments $g$ and predicted segments $p$ *agree* when they overlap and their labels are accepted as the same activity:

```math
A(g,p) = \mathbb 1\big[\,|g\cap p|>0 \;\wedge\; J(\ell_g,\ell_p)\,\big]
```

$J$ is the LLM judge for subtasks and captions. For actions, $J$ checks that both labels normalize to the same Ego4D verb class, and a verb outside the set never matches.

```math
\mathrm{PSJ} \;=\; \frac{\displaystyle\sum_k\Big(\sum_{p\in P_k}\Big|\bigcup_{g:\,A(g,p)} g\cap p\Big| \;+\; \sum_{g\in G_k}\Big|\bigcup_{p:\,A(g,p)} g\cap p\Big|\Big)}{\displaystyle\sum_k\Big(\sum_{p\in P_k}|p| \;+\; \sum_{g\in G_k}|g|\Big)}
```

Each predicted second is credited when some gold segment covering it agrees, and each gold second likewise. A prediction that splits a gold segment into finer, equally valid steps, or merges several into one, still scores 1. Label errors and timing errors both cost time on both sides.

**By segment length.** Restricting $G_k$ to gold segments of a given length (< 0.5 s, 0.5–1 s, > 1 s) and clipping predictions to those spans gives the per-length scores reported below.

## Results

The numbers below match the [Perceptron Egocentric 1.0](https://www.perceptron.inc/blog/introducing-perceptron-egocentric-1-0) and [EgoHome](https://www.perceptron.inc/blog/egohome) blog posts.

EgoHome, all 137 clips (higher is better; cost lower is better):

| System | Subtask | Hand action | Hand caption | Hand box | Subtask $ / video-hour |
|---|---|---|---|---|---|
| Perceptron Egocentric, 4 fps (default) | 0.787 | 0.421 | 0.613 | 0.942 | $9 |
| Perceptron Egocentric, 8 fps (option) | **0.798** | 0.421 | 0.626 | 0.973 | ≈ $15 |
| GPT-6 Astra, 4 fps frames | 0.797 | **0.498** | **0.749** | **0.979** | ≈ $76 |
| GPT-6 Astra, 1 fps frames | 0.775 | 0.479 | 0.724 | 0.865 | ≈ $20 |
| Gemini 3.8 Flash, 4 fps video | 0.663 | 0.405 | 0.558 | 0.454 | **≈ $1.3** |
| Gemini 3.8 Flash, 1 fps video | 0.639 | 0.390 | 0.557 | 0.574 | n/r |
| Glimpse (Elorian), 4 fps video | 0.689 | 0.390 | 0.560 | 0.489 | ≈ $2.8 |
| Glimpse (Elorian), 1 fps video | 0.590 | 0.353 | 0.560 | 0.247 | n/r |
| Argus (Pantheon), GPT-6 Astra | 0.367 | 0.336 | 0.593 | — | $24.17 |
| Argus (Pantheon), Midcentury's settings: GPT-6.1 Sol | 0.315 | 0.327 | 0.590 | — | $7.76 |
| FLORA (NVIDIA) | 0.747 | — | — | — | $35.39 |

Best in bold. A failed clip counts 0; — = the system does not produce that annotation; n/r = not reported. Label judge: gemini-3.5-flash.

Inputs: GPT-6 Astra as timestamped frames (768 px), Gemini and Glimpse as native video. Perceptron results are averaged across three runs at 4 fps and two runs at 8 fps; other systems were run once. Gemini stamps its boxes half a sampling interval early, so its hand box scores use timestamps shifted by 0.5 s at 1 fps and 0.125 s at 4 fps. Subtask $ uses Perceptron's customer price, direct API charges, or FLORA's GPU runtime cost; Argus costs also include hand labels, and costs marked ≈ are estimates of the subtask share of a run billed across all tasks.

Subtask score by gold segment length:

| System | All | < 0.5 s | 0.5–1 s | > 1 s |
|---|---|---|---|---|
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
| Clips with a step of this length | 137 | 24 | 60 | 137 |

Hand action score by gold segment length:

| System | All | < 0.5 s | 0.5–1 s | > 1 s |
|---|---|---|---|---|
| Perceptron Egocentric, 4 fps (default) | 0.421 | 0.298 | 0.286 | 0.461 |
| GPT-6 Astra, 4 fps frames | **0.498** | **0.422** | **0.415** | **0.531** |
| GPT-6 Astra, 1 fps frames | 0.479 | 0.341 | 0.355 | 0.520 |
| Gemini 3.8 Flash, 4 fps video | 0.405 | 0.206 | 0.273 | 0.435 |
| Gemini 3.8 Flash, 1 fps video | 0.390 | 0.191 | 0.252 | 0.436 |
| Glimpse (Elorian), 4 fps video | 0.390 | 0.193 | 0.212 | 0.433 |
| Glimpse (Elorian), 1 fps video | 0.353 | 0.212 | 0.185 | 0.392 |
| Argus (Pantheon), GPT-6 Astra | 0.336 | 0.321 | 0.268 | 0.358 |
| Argus (Pantheon), Midcentury's settings: GPT-6.1 Sol | 0.327 | 0.262 | 0.205 | 0.355 |

Hand caption score by gold segment length:

| System | All | < 0.5 s | 0.5–1 s | > 1 s |
|---|---|---|---|---|
| Perceptron Egocentric, 4 fps (default) | 0.613 | 0.447 | 0.446 | 0.678 |
| GPT-6 Astra, 4 fps frames | **0.749** | **0.560** | **0.642** | **0.830** |
| GPT-6 Astra, 1 fps frames | 0.724 | 0.415 | 0.575 | 0.800 |
| Gemini 3.8 Flash, 4 fps video | 0.558 | 0.343 | 0.379 | 0.618 |
| Gemini 3.8 Flash, 1 fps video | 0.557 | 0.295 | 0.383 | 0.619 |
| Glimpse (Elorian), 4 fps video | 0.560 | 0.281 | 0.351 | 0.625 |
| Glimpse (Elorian), 1 fps video | 0.560 | 0.264 | 0.337 | 0.631 |
| Argus (Pantheon), GPT-6 Astra | 0.593 | 0.414 | 0.490 | 0.647 |
| Argus (Pantheon), Midcentury's settings: GPT-6.1 Sol | 0.590 | 0.433 | 0.441 | 0.648 |

### Degenerate baselines

Answers that ignore the video, scored with the same metrics and judge.

| Baseline | Subtask | Hand action | Hand caption |
|---|---|---|---|
| Whole clip, generic label | 0.000 | 0.175 | 0.000 |
| 2.5 s windows, generic label | 0.000 | 0.175 | 0.000 |
| Whole clip, the clip's own majority gold label (oracle) | 0.528 | 0.449 | 0.499 |

## Layout

```
egohome/metrics.py    centroid hit fraction, permissible subsegment Jaccard (+ length buckets)
egohome/ego4d.py      the 118 Ego4D FHO verbs and an alias/inflection normalizer
egohome/prompts.py    task prompts and the label-judge prompt
egohome/judge.py      Gemini label judge (cached); any (gold, pred) -> bool callable works
egohome/evaluate.py   CLI
tests/                  metric tests
```

## Citation

```bibtex
@article{perceptron2026egohome,
  author  = {PerceptronAI},
  title   = {Perceptron EgoHome: A new benchmark for dense, sub-second egocentric understanding},
  journal = {PerceptronAI Research Blog},
  year    = {2026},
  note    = {https://perceptron.inc/blog/egohome},
}
```

## License

The EgoHome dataset ([`PerceptronAI/EgoHome`](https://huggingface.co/datasets/PerceptronAI/EgoHome)) is released under the [Creative Commons Attribution-NonCommercial 4.0](https://creativecommons.org/licenses/by-nc/4.0/) license (CC BY-NC 4.0). As an additional permission, anyone, including commercial organizations, may use EgoHome to evaluate models and publish the results. Other commercial uses, such as selling the data or training commercial models on it, are not permitted.

The code in this repository is released under the [MIT License](LICENSE).
