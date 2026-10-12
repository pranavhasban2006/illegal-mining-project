import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mining_detector import hard_case_labels as hcl


def _chips():
    return [
        {"chip_id": "a", "category": "cat1", "row": 5, "col": 5},
        {"chip_id": "b", "category": "cat1", "row": 15, "col": 15},
        {"chip_id": "c", "category": "cat2", "row": 25, "col": 25},
        {"chip_id": "d", "category": "cat2", "row": 35, "col": 35},
    ]


def test_load_votes_and_chips_excludes_unsure():
    votes = {"a": "mining", "b": "unsure", "c": "not_mining", "d": "mining"}
    joined = hcl.load_votes_and_chips(votes, _chips())
    ids = {c["chip_id"] for c in joined}
    assert ids == {"a", "c", "d"}


def test_load_votes_and_chips_maps_labels_correctly():
    votes = {"a": "mining", "c": "not_mining"}
    joined = hcl.load_votes_and_chips(votes, _chips())
    by_id = {c["chip_id"]: c["human_label"] for c in joined}
    assert by_id["a"] == hcl.MINING
    assert by_id["c"] == hcl.NOT_MINING


def test_stratified_split_covers_every_category():
    votes = {"a": "mining", "b": "not_mining", "c": "mining", "d": "not_mining"}
    joined = hcl.load_votes_and_chips(votes, _chips())
    train, test = hcl.stratified_split(joined, train_fraction=0.5, seed=0)

    train_cats = {c["category"] for c in train}
    test_cats = {c["category"] for c in test}
    assert train_cats == {"cat1", "cat2"}
    assert test_cats == {"cat1", "cat2"}
    assert len(train) + len(test) == len(joined)


def test_stratified_split_no_overlap():
    votes = {"a": "mining", "b": "not_mining", "c": "mining", "d": "not_mining"}
    joined = hcl.load_votes_and_chips(votes, _chips())
    train, test = hcl.stratified_split(joined, train_fraction=0.5, seed=1)
    train_ids = {c["chip_id"] for c in train}
    test_ids = {c["chip_id"] for c in test}
    assert train_ids.isdisjoint(test_ids)


def test_apply_hard_case_labels_overrides_neighborhood():
    labels = np.full((40, 40), -1, dtype=np.int8)
    chips = [{"row": 20, "col": 20, "human_label": hcl.MINING}]
    result = hcl.apply_hard_case_labels(labels, chips, window=2)

    assert (result[18:23, 18:23] == hcl.MINING).all()
    assert result[0, 0] == -1  # untouched elsewhere


def test_apply_hard_case_labels_clips_at_boundary():
    labels = np.full((10, 10), -1, dtype=np.int8)
    chips = [{"row": 0, "col": 0, "human_label": hcl.NOT_MINING}]
    # Should not raise despite window extending past the array edge.
    result = hcl.apply_hard_case_labels(labels, chips, window=3)
    assert result[0, 0] == hcl.NOT_MINING


def test_evaluate_on_chips_confusion_matrix():
    proba = np.array([
        [0.9, 0.1],
        [0.2, 0.8],
    ])
    chips = [
        {"row": 0, "col": 0, "human_label": hcl.MINING},      # TP
        {"row": 0, "col": 1, "human_label": hcl.NOT_MINING},  # TN
        {"row": 1, "col": 0, "human_label": hcl.MINING},      # FN (pred 0.2 < 0.5)
        {"row": 1, "col": 1, "human_label": hcl.NOT_MINING},  # FP (pred 0.8 >= 0.5)
    ]
    metrics = hcl.evaluate_on_chips(proba, chips, threshold=0.5)
    assert metrics["tp"] == 1
    assert metrics["tn"] == 1
    assert metrics["fn"] == 1
    assert metrics["fp"] == 1
    assert metrics["accuracy"] == 0.5


def test_dashboard_votes_are_skipped_not_crashing():
    # Dashboard entries use "chip__labeler" keys with dict values; they must
    # not reach the chip lookup (regression: KeyError on the first such key).
    votes = {
        "a": "mining",
        "c": "not_mining",
        "b__user": {"chip_id": "b", "label": "mining", "labeler": "user"},
    }
    joined = hcl.load_votes_and_chips(votes, _chips())
    assert sorted(j["chip_id"] for j in joined) == ["a", "c"]
