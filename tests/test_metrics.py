from egochores.metrics import centroid_hit_fraction, ego4d_verb_accept, permissible_subsegment_jaccard


def same_words(g, p):
    return g.split()[0] == p.split()[0]


def test_identical_segmentations_score_one():
    gold = {"subtask": [(0, 5, "sweep the stairs")]}
    assert permissible_subsegment_jaccard(gold, gold, same_words)[0] == 1.0


def test_a_finer_but_equally_valid_segmentation_scores_one():
    gold = {"subtask": [(0, 5, "sweep the stairs")]}
    pred = {"subtask": [(0, 2, "sweep the top stair"), (2, 5, "sweep the bottom stair")]}
    assert permissible_subsegment_jaccard(gold, pred, same_words)[0] == 1.0


def test_a_wrong_label_scores_zero_and_partial_overlap_is_partial():
    gold = {"subtask": [(0, 4, "wipe the table")]}
    assert permissible_subsegment_jaccard(gold, {"subtask": [(0, 4, "open the drawer")]}, same_words)[0] == 0.0
    # Agreeing time 2 s (pred) + 2 s (gold) over 2 s + 4 s total.
    assert permissible_subsegment_jaccard(gold, {"subtask": [(2, 4, "wipe it")]}, same_words)[0] == 4 / 6


def test_hands_are_pooled_and_length_buckets_reported():
    gold = {"left": [(0, 0.4, "hold")], "right": [(0, 2, "take")]}
    pred = {"left": [(0, 0.4, "hold")], "right": [(0, 2, "put")]}
    score, buckets = permissible_subsegment_jaccard(gold, pred, ego4d_verb_accept)
    assert score == 0.8 / 4.8
    assert buckets["lt_0.5s"] == 1.0 and buckets["gt_1s"] == 0.0 and buckets["0.5_1s"] is None


def test_ego4d_verbs_are_matched_by_class():
    assert ego4d_verb_accept("take_(pick,_grab,_get)", "picking")
    assert not ego4d_verb_accept("hold_(support,_grip,_grasp)", "vacuum")  # outside the closed set


def test_centroid_hit_interpolates_between_predicted_frames():
    gold = [{"t": i / 30, "left": [100, 100, 200, 200], "right": None} for i in range(31)]
    pred = [{"t": 0.0, "left": [110, 110, 190, 190]}, {"t": 1.0, "left": [110, 110, 190, 190]}]
    assert centroid_hit_fraction(gold, pred, 30) == 1.0
    moved = [{"t": 0.0, "left": [110, 110, 190, 190]}, {"t": 1.0, "left": [510, 110, 590, 190]}]
    # The centroid moves 150 -> 550 over 30 frames: it stays inside x < 200 for frames 0-3 only.
    assert centroid_hit_fraction(gold, moved, 30) == 4 / 31


def test_predictions_outside_the_gold_range_count_against():
    gold = [{"t": 0.0, "left": [100, 100, 200, 200]}]
    pred = [{"t": 0.0, "left": [110, 110, 190, 190]}, {"t": 5.0, "left": [110, 110, 190, 190]}]
    assert centroid_hit_fraction(gold, pred, 30) == 0.5
