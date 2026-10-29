"""Tests for the leakage-safe splitting helpers and the metric conventions.

The splits in this module decide what a paper may claim, so the invariants are
asserted explicitly rather than assumed:

* the partitions are disjoint and cover every sample;
* no group (record id / subject id) appears in two partitions;
* the split is deterministic for a fixed seed;
* the macro F1 only averages over the classes that are actually present.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.data.splitting import (
    class_distribution,
    grouped_split_indices,
    grouped_split_indices_3way,
)


def _toy_dataset(n_groups: int = 30, per_group: int = 6):
    """A toy grouped dataset with three classes and a clean group structure."""
    labels = []
    groups = []
    for group in range(n_groups):
        # Every group carries all three classes in a fixed pattern.
        for repeat in range(per_group):
            labels.append(group % 3)
            groups.append(f"rec-{group:03d}")
    return np.asarray(labels), np.asarray(groups)


def test_two_way_split_is_disjoint_and_covers_everything():
    labels, groups = _toy_dataset()
    train_idx, val_idx = grouped_split_indices(labels, groups, 0.2, seed=42)

    assert len(np.intersect1d(train_idx, val_idx)) == 0
    assert sorted(train_idx.tolist() + val_idx.tolist()) == list(range(len(labels)))


def test_two_way_split_never_splits_a_group():
    labels, groups = _toy_dataset()
    train_idx, val_idx = grouped_split_indices(labels, groups, 0.2, seed=42)

    train_groups = set(groups[train_idx].tolist())
    val_groups = set(groups[val_idx].tolist())
    assert train_groups & val_groups == set()


def test_three_way_split_partitions_are_disjoint_and_complete():
    labels, groups = _toy_dataset()
    train_idx, val_idx, test_idx = grouped_split_indices_3way(
        labels, groups, 0.15, 0.15, seed=42
    )

    parts = [train_idx, val_idx, test_idx]
    assert len(np.intersect1d(parts[0], parts[1])) == 0
    assert len(np.intersect1d(parts[0], parts[2])) == 0
    assert len(np.intersect1d(parts[1], parts[2])) == 0
    assert sorted(
        train_idx.tolist() + val_idx.tolist() + test_idx.tolist()
    ) == list(range(len(labels)))


def test_three_way_split_never_splits_a_group():
    labels, groups = _toy_dataset()
    train_idx, val_idx, test_idx = grouped_split_indices_3way(
        labels, groups, 0.15, 0.15, seed=42
    )

    train_groups = set(groups[train_idx].tolist())
    val_groups = set(groups[val_idx].tolist())
    test_groups = set(groups[test_idx].tolist())

    assert train_groups & val_groups == set()
    assert train_groups & test_groups == set()
    assert val_groups & test_groups == set()


def test_three_way_split_is_deterministic_for_a_fixed_seed():
    labels, groups = _toy_dataset()
    first = grouped_split_indices_3way(labels, groups, 0.15, 0.15, seed=42)
    second = grouped_split_indices_3way(labels, groups, 0.15, 0.15, seed=42)

    for left, right in zip(first, second):
        assert np.array_equal(left, right)


def test_three_way_split_respects_the_sample_budgets():
    labels, groups = _toy_dataset()
    train_idx, val_idx, test_idx = grouped_split_indices_3way(
        labels, groups, 0.15, 0.15, seed=42
    )

    total = len(labels)
    # A group is never broken apart, so the partitions may overshoot the
    # requested ratio slightly - but they must stay in the same ballpark and
    # the training side must stay the largest one.
    assert len(val_idx) <= total * 0.30
    assert len(test_idx) <= total * 0.30
    assert len(train_idx) > len(val_idx)
    assert len(train_idx) > len(test_idx)


def test_three_way_split_accepts_integer_group_ids():
    """Integer group ids must work and match their string equivalents.

    The partition is derived from the *string* form of the group id (the ids
    are sorted and shuffled before the greedy assignment), so ``17`` and
    ``"17"`` have to produce exactly the same partition.
    """
    labels, _ = _toy_dataset()
    integer_groups = np.asarray([index % 30 for index in range(len(labels))])
    string_groups = np.asarray([str(value) for value in integer_groups])

    integer_split = grouped_split_indices_3way(labels, integer_groups, 0.15, 0.15, seed=42)
    string_split = grouped_split_indices_3way(labels, string_groups, 0.15, 0.15, seed=42)

    for left, right in zip(integer_split, string_split):
        assert np.array_equal(left, right)

    # And the result is a valid partition, not an all-false mask.
    train_idx, val_idx, test_idx = integer_split
    assert sorted(
        train_idx.tolist() + val_idx.tolist() + test_idx.tolist()
    ) == list(range(len(labels)))


def test_three_way_split_falls_back_to_two_way_when_test_ratio_is_zero():
    labels, groups = _toy_dataset()
    train_idx, val_idx, test_idx = grouped_split_indices_3way(
        labels, groups, 0.2, 0.0, seed=42
    )

    assert len(test_idx) == 0
    reference_train, reference_val = grouped_split_indices(labels, groups, 0.2, seed=42)
    assert np.array_equal(train_idx, reference_train)
    assert np.array_equal(val_idx, reference_val)


def test_three_way_split_rejects_ratios_that_do_not_fit():
    labels, groups = _toy_dataset()
    with pytest.raises(ValueError):
        grouped_split_indices_3way(labels, groups, 0.6, 0.5, seed=42)


def test_class_distribution_counts_every_class():
    labels, groups = _toy_dataset(n_groups=9)
    counts = class_distribution(labels, ["A", "B", "C", "D"])
    assert counts["A"] == 18
    assert counts["B"] == 18
    assert counts["C"] == 18
    assert counts["D"] == 0


def test_macro_f1_only_averages_over_present_classes():
    """A binary task scored against a four-class head must not be halved."""
    from app.ml.training import compute_metrics

    y_true = np.asarray([0, 0, 1, 1])
    y_pred = np.asarray([0, 1, 1, 1])
    probabilities = np.asarray([[0.9, 0.1], [0.6, 0.4], [0.2, 0.8], [0.3, 0.7]])

    metrics = compute_metrics(y_true, y_pred, probabilities, n_classes=4)

    assert metrics["f1_0"] == pytest.approx(2 / 3, rel=1e-6)
    assert metrics["f1_1"] == pytest.approx(0.8, rel=1e-6)
    assert metrics["f1_2"] == 0.0
    # mean of the two *present* classes only
    assert metrics["f1_macro"] == pytest.approx((2 / 3 + 0.8) / 2, rel=1e-6)


def test_network_initialisation_is_reproducible_after_set_seed():
    """Seeding must happen before the network is built, not inside the loop."""
    import torch

    from app.ml.ecg.model import ECGNet
    from app.ml.training import set_seed

    set_seed(42)
    first = ECGNet(num_classes=5).state_dict()
    set_seed(42)
    second = ECGNet(num_classes=5).state_dict()

    assert set(first.keys()) == set(second.keys())
    for key in first:
        assert torch.equal(first[key], second[key]), key
