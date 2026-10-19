import numpy as np

from app.models.lvq import LVQ1, evaluate_lvq, merge_small_classes


def test_high_accuracy_on_separable_data(blobs):
    X, y = blobs
    ev = evaluate_lvq(X, y, folds=5)
    assert ev.accuracy > 0.95
    assert ev.accuracy > ev.baseline_accuracy
    assert np.array(ev.confusion).sum() == len(y)


def test_prototype_moves_towards_correct_and_away_from_wrong():
    m = LVQ1()
    m.prototypes = np.array([[0.0, 0.0], [10.0, 10.0]])
    m.proto_labels = np.array(["A", "B"], dtype=object)
    x = np.array([1.0, 1.0])                     # winner is prototype 0

    before = np.linalg.norm(x - m.prototypes[0])
    m.step(x, "A", eta=0.5)                      # correct class -> closer
    assert np.linalg.norm(x - m.prototypes[0]) < before
    np.testing.assert_allclose(m.prototypes[0], [0.5, 0.5])

    before = np.linalg.norm(x - m.prototypes[0])
    m.step(x, "B", eta=0.5)                      # wrong class -> further away
    assert np.linalg.norm(x - m.prototypes[0]) > before
    np.testing.assert_allclose(m.prototypes[1], [10, 10])    # loser untouched


def test_small_sectors_merged_into_other():
    y = np.array(["A"] * 5 + ["B"] * 2 + ["C"] * 1 + ["D"] * 3, dtype=object)
    merged = merge_small_classes(y, min_size=3)
    assert set(merged) == {"A", "D", "Other"}
    assert (merged == "Other").sum() == 3
