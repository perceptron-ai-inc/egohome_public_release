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

## How different models stack up

| System | Subtask | Hand action | Hand caption | Hand box | Failed | Timeline $ / video-hour | All tasks $ / video-hour |
|---|---:|---:|---:|---:|---:|---:|---:|
| Perceptron Egocentric, 4 fps (default) | 0.785 | 0.421 | 0.613 | 0.942 | 3/548 | $9 | |
| Perceptron Egocentric, 8 fps (option) | **0.796** | n/r | n/r | n/r | n/r | ≈ $15 | n/r |
| GPT-6 Astra, 4 fps frames | 0.795 | **0.498** | **0.749** | **0.979** | 0/548 | ≈ $76 | $244.66 |
| GPT-6 Astra, 1 fps frames | 0.773 | 0.479 | 0.724 | 0.865 | 0/548 | ≈ $20 | $65.07 |
| Gemini 3.8 Flash, 4 fps video | 0.660 | 0.405 | 0.558 | 0.444 | 0/548 | **≈ $1.3** | $5.55 |
| Gemini 3.8 Flash, 1 fps video | 0.639 ‡ | 0.390 | 0.557 | 0.522 | 0/548 | n/r | **$1.88** |
| Glimpse (Elorian), 4 fps video | 0.685 | 0.390 | 0.560 | 0.489 | 0/548 | ≈ $2.8 | $9.64 |
| Glimpse (Elorian), 1 fps video | 0.590 ‡ | 0.353 | 0.560 | 0.247 | 0/548 | n/r | $9.45 |
| Argus (Pantheon), GPT-6 Astra | 0.361 | 0.336 | 0.593 | — | 0/411 | $24.17 | $24.17 |
| Argus (Pantheon), Midcentury's settings: GPT-6.1 Sol | 0.307 | 0.327 | 0.590 | — | 0/411 | $7.76 | $7.76 |
| FLORA (NVIDIA) | 0.744 | — | — | — | 0/137 | $35.39 | $35.39 |

Permissible subsegment Jaccard (subtask, hand action, hand caption) and centroid hit fraction (hand box); a failed clip counts 0. Best in bold. — = the system does not produce that annotation. n/r = not reported for that setting. Inputs: GPT-6 Astra as timestamped frames (768 px), Gemini and Glimpse as native video. Subtask scores cover the same 135 clips; Perceptron results are averaged across three runs at 4 fps and two runs at 8 fps, and other systems were run once. ‡ Scored over all 137 clips, as are the hand action, hand caption and hand box columns. Timeline $ uses Perceptron's customer price, direct API charges, or FLORA's GPU runtime cost; Argus costs also include hand labels, and costs marked ≈ are estimates of the timeline's share of a run billed across all tasks. All tasks $ is from billed usage (API models), Modal compute + model calls (harness), H100 time (FLORA).

### Subtask score by gold segment length

| System | All | < 0.5 s | 0.5–1 s | > 1 s |
|---|---:|---:|---:|---:|
| Perceptron Egocentric, 4 fps (default) | 0.785 | 0.559 | 0.609 | 0.812 |
| Perceptron Egocentric, 8 fps (option) | **0.796** | 0.555 | **0.667** | 0.822 |
| GPT-6 Astra, 4 fps frames | 0.795 | 0.503 | 0.643 | **0.829** |
| GPT-6 Astra, 1 fps frames | 0.773 | **0.595** | 0.492 | 0.805 |
| Gemini 3.8 Flash, 4 fps video | 0.660 | 0.375 | 0.381 | 0.692 |
| Gemini 3.8 Flash, 1 fps video | 0.639 ‡ | 0.361 ‡ | 0.320 ‡ | 0.670 ‡ |
| Glimpse (Elorian), 4 fps video | 0.685 | 0.380 | 0.400 | 0.716 |
| Glimpse (Elorian), 1 fps video | 0.590 ‡ | 0.328 ‡ | 0.279 ‡ | 0.616 ‡ |
| Argus (Pantheon), GPT-6 Astra | 0.361 | 0.135 | 0.119 | 0.375 |
| Argus (Pantheon), Midcentury's settings: GPT-6.1 Sol | 0.307 | 0.098 | 0.108 | 0.323 |
| FLORA (NVIDIA) | 0.744 | 0.579 | 0.609 | 0.774 |
| *Clips with a step of this length* | 135 | 24 | 59 | 135 |

Best in bold. ‡ Scored over all 137 clips; every other row covers the same 135 clips.

<details>
<summary>Hand action score by gold segment length</summary>

| System | All | < 0.5 s | 0.5–1 s | > 1 s |
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

</details>

<details>
<summary>Hand caption score by gold segment length</summary>

| System | All | < 0.5 s | 0.5–1 s | > 1 s |
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

</details>

## Evaluation metrics

### Hand box metric: centroid hit fraction

To measure hand box prediction accuracy, we simply ask each model to produce timestamped bounding boxes for the hand locations throughout the video. For each frame we check whether the centroid of the predicted box falls inside of the ground truth box, we then divide by the total number of frames for which there's an annotation.

We evaluate each model only on the subset of the ground truth corresponding to its configured FPS. For instance Gemini 1FPS would measure the centroid hit fraction only on the 60 frames sampled. Technically this metric is slightly more lenient to lower FPS configurations since higher frame rates give more opportunities to be incorrect, but in practice we see most models perform better at higher frame rate.

Mathematically this can be expressed as

$$
\frac{\displaystyle\sum_{i \in F_p \cup F_t} \mathbb{1}\left[\operatorname{Centroid}(p_i) \in t_i\right]}{\left| F_p \cup F_t \right|}
$$

Where $F_p$ is the set of frames for which there is a predicted hand, $F_t$ is the set of frames for which there's a ground truth hand, and $\operatorname{Centroid}(p_i)$ is an indicator variable indicating if the predicted box has its centroid inside the target box.

![Centroid hit on one 2 s span: 107 of 120 frame-hands hit, a score of 0.89](assets/hand_box_metric_3d.gif)

### Subtask, action, and hand caption metric: permissible subsegment Jaccard

We developed our subtask correctness metric while attempting to find a fair way to calibrate our annotators. We initially used a metric similar to WGO-Bench's semantic F1, which performs bipartite matching between the target and prediction and then calculating F1 between the predicted and target sets.

Bipartite matching, which is common in object segmentation benchmarks, allows each ground truth segment to be paired with exactly one predicted segment. In other words, during scoring, the mapping between predicted and ground truth is one-to-one.

We've found, however, that there are many valid ways to break down the subtasks in a video. Even if you show the same annotator the same video, in one sitting they may break down washing the dishes into a single task or they may break it down into many smaller segments.

Rather than encouraging models to overfit to one particular choice of temporal granularity, we opted to allow the ground truth and predicted segments to match to each other in a many-to-many mapping, and we've tailored our LLM as judge prompt to permit predicted tasks to be subtasks of the ground truth or vice versa.

We call our metric **permissible subsegment jaccard.**

#### Worked example: the stairs

Let's say the ground truth labeled the time span 0 to 5s as one segment, "sweeping the stairs".

The table below demonstrates how various predictions would be graded under our many-to-many permissible subsegment jaccard, vs the one-to-one WGO-Bench semantic F1.

| Model output | Permissible subsegment Jaccard | WGO-Bench semantic F1 |
|---|---|---|
| A. "sweeping the stairs" 0–5 s | (5 + 5) / (5 + 5) = **1.0** | IoU 1.0, labels match: 1 match → **1.0** |
| B. "sweeping the top stair" 0–2 s, "sweeping the bottom stair" 2–5 s | both pieces agree: (2 + 3 + 5) / (5 + 5) = **1.0** | IoUs 0.4 and 0.6, both below 0.75: 0 matches of 2 predicted, 1 gold → **0.0** |
| C. "sweeping the stairs" 0–3 s, "putting the broom away" 3–5 s | first piece agrees: (3 + 3) / (5 + 5) = **0.6** | IoU 0.6, below 0.75 → **0.0** |
| D. "carrying the broom" 0–5 s | judge says it's a different action → **0.0** | timing matches but the label doesn't → **0.0** |

While calibrating annotators, we frequently ran into Case B: a valid, more detailed breakdown gets a score of  zero under one-to-one matching and full credit under ours. Case C shows we still give partial credit for the part the model got right, where a fixed IoU cutoff gives nothing. (WGO-Bench's semantic F1 pairs segments one-to-one, counts a pair only when its temporal IoU is at least 0.75 and the judge accepts the labels).

We found that this metric adhered much more closely to subjective measures of fairness when evaluating human annotators' performance during calibration, thus we opted to release it as our official metric for the benchmark.

It can be expressed mathematically as follows:

$$
\operatorname{PSJ}(P, T) = \frac{\displaystyle\sum_{p \in P} m(p, T) + \sum_{t \in T} m(t, P)}{\displaystyle\sum_{p \in P} |I(p)| + \sum_{t \in T} |I(t)|}
$$

$$
m(s, S) = \Bigl|\, I(s) \cap \bigcup_{\substack{s' \in S \\ J(s, s') = 1}} I(s') \,\Bigr|
$$

Let $P$ and $T$ be the predicted and ground-truth segments. Each segment $s$ has a time interval $I(s)$ and a label, and $|\cdot|$ is duration. $J(s,s') \in \{0,1\}$ is the LLM judge's verdict on whether two labels are permissible matches: the same action, or one a subtask of the other. $m(s,S)$ is the portion of $s$'s duration that overlaps some segment in $S$ that the judge accepts.

#### Degenerate baselines

In order to ensure the metric isn't trivially gameable check the metric can't be gamed, we scored answers that ignore the video entirely, with the same metric and judge. Generic labels are "doing chores" for subtasks, "using the hands" for captions and "hold", the most frequent verb, for hand actions.

| Answer (no video used) | Subtask | Hand action | Hand caption |
|---|---:|---:|---:|
| One segment for the whole clip, generic label | 0.000 | 0.175 | 0.000 |
| Fixed 2.5 s windows, generic label | 0.000 | 0.175 | 0.000 |
| One segment for the whole clip, the clip's own most common true label | 0.528 | 0.449 | 0.499 |

Generic labels earn nothing on subtasks and captions. A constant "hold" earns 0.175 on hand actions, the share of time hands really are holding. The best cheap answer is the right activity at the coarsest granularity, which earns about half credit, against 0.77–0.80 for the best systems on subtasks.

Label judge: gemini-3.5-flash, temperature 0, thinking off; prompt in the repository. Re-judging the same outputs moves a system's score by about ±0.01.

## Citation

```bibtex
@misc{perceptron2026egohome,
  title={EgoHome: 2 hours of human-annotated egocentric video},
  author={{Perceptron}},
  year={2026},
  month={nov},
  url={https://www.perceptron.inc/blog/egohome}
}
```

## References

```bibtex
@misc{dyna2026dyna2,
  title={Dyna-2: A 1-Million-Hour Scaling Law for World-Action Models},
  author={{Dyna Robotics}},
  year={2026},
  month={aug},
  url={https://www.dyna.co/dyna-2}
}

@misc{macrodata2026segmenting,
  title={Segmenting Robot Video into Actionable Subtasks},
  author={{Macrodata Labs}},
  year={2026},
  month={jun},
  url={https://macrodata.co/blog/annotating-robot-video-subtasks}
}

@inproceedings{grauman2022ego4d,
  title={Ego4D: Around the World in 3,000 Hours of Egocentric Video},
  author={Grauman, Kristen and Westbury, Andrew and Byrne, Eugene and Chavis, Zachary and Furnari, Antonino and Girdhar, Rohit and Hamburger, Jackson and Jiang, Hao and Liu, Miao and Liu, Xingyu and Martin, Miguel and Nagarajan, Tushar and Radosavovic, Ilija and Ramakrishnan, Santhosh Kumar and Ryan, Fiona and Sharma, Jayant and Wray, Michael and Xu, Mengmeng and Xu, Eric Zhongcong and Zhao, Chen and Bansal, Siddhant and Batra, Dhruv and Cartillier, Vincent and Crane, Sean and Do, Tien and Doulaty, Morrie and Erapalli, Akshay and Feichtenhofer, Christoph and Fragomeni, Adriano and Fu, Qichen and Gebreselasie, Abrham and Gonzalez, Cristina and Hillis, James and Huang, Xuhua and Huang, Yifei and Jia, Wenqi and Khoo, Weslie and Kolar, Jachym and Kottur, Satwik and Kumar, Anurag and Landini, Federico and Li, Chao and Li, Yanghao and Li, Zhenqiang and Mangalam, Karttikeya and Modhugu, Raghava and Munro, Jonathan and Murrell, Tullie and Nishiyasu, Takumi and Price, Will and Ruiz Puentes, Paola and Ramazanova, Merey and Sari, Leda and Somasundaram, Kiran and Southerland, Audrey and Sugano, Yusuke and Tao, Ruijie and Vo, Minh and Wang, Yuchen and Wu, Xindi and Yagi, Takuma and Zhao, Ziwei and Zhu, Yunyi and Arbelaez, Pablo and Crandall, David and Damen, Dima and Farinella, Giovanni Maria and Fuegen, Christian and Ghanem, Bernard and Ithapu, Vamsi Krishna and Jawahar, C. V. and Joo, Hanbyul and Kitani, Kris and Li, Haizhou and Newcombe, Richard and Oliva, Aude and Park, Hyun Soo and Rehg, James M. and Sato, Yoichi and Shi, Jianbo and Shou, Mike Zheng and Torralba, Antonio and Torresani, Lorenzo and Yan, Mingfei and Malik, Jitendra},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2022},
  url={https://arxiv.org/abs/2110.07058}
}
```

## Layout

```
egohome/metrics.py    centroid hit fraction, permissible subsegment Jaccard (+ length buckets)
egohome/ego4d.py      the 118 Ego4D FHO verbs and an alias/inflection normalizer
egohome/prompts.py    task prompts and the label-judge prompt
egohome/judge.py      Gemini label judge (cached); any (gold, pred) -> bool callable works
egohome/evaluate.py   CLI
tests/                  metric tests
```
