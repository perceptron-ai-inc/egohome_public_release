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

| System | Subtask | Hand action | Hand caption | Hand box | Timeline $ / video-hour |
|---|---:|---:|---:|---:|---:|
| Perceptron Egocentric, 4 fps (default) | 0.785 | 0.421 | 0.613 | 0.942 | $9 |
| Perceptron Egocentric, 8 fps (option) | **0.796** | n/r | n/r | n/r | ≈ $15 |
| GPT-6 Astra, 4 fps frames | 0.795 | **0.498** | **0.749** | **0.979** | ≈ $76 |
| GPT-6 Astra, 1 fps frames | 0.773 | 0.479 | 0.724 | 0.865 | ≈ $20 |
| Gemini 3.8 Flash, 4 fps video | 0.660 | 0.405 | 0.558 | 0.444 | **≈ $1.3** |
| Gemini 3.8 Flash, 1 fps video | 0.639 ‡ | 0.390 | 0.557 | 0.522 | n/r |
| Glimpse (Elorian), 4 fps video | 0.685 | 0.390 | 0.560 | 0.489 | ≈ $2.8 |
| Glimpse (Elorian), 1 fps video | 0.590 ‡ | 0.353 | 0.560 | 0.247 | n/r |
| Argus (Pantheon), GPT-6 Astra | 0.361 | 0.336 | 0.593 | — | $24.17 |
| Argus (Pantheon), Midcentury's settings: GPT-6.1 Sol | 0.307 | 0.327 | 0.590 | — | $7.76 |
| FLORA (NVIDIA) | 0.744 | — | — | — | $35.39 |

Best in bold. A failed clip counts 0; — means the system does not produce that annotation; n/r means not reported for that setting. Subtask scores cover the same 135 clips; ‡ marks scores over all 137 clips, as are the hand action, hand caption and hand box columns. Perceptron subtask results are averaged across three runs at 4 fps and two runs at 8 fps; other systems were run once. Costs marked ≈ are estimates. Label judge: gemini-3.5-flash.

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

### Degenerate baselines

Answers that ignore the video, scored with the same metrics and judge.

| Baseline | Subtask | Hand action | Hand caption |
|---|---|---|---|
| Whole clip, generic label | 0.000 | 0.175 | 0.000 |
| 2.5 s windows, generic label | 0.000 | 0.175 | 0.000 |
| Whole clip, the clip's own majority gold label (oracle) | 0.528 | 0.449 | 0.499 |

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
