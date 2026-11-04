import math

import numpy as np
import pytest

import audmetric


def r_value_reference(n_truth, n_prediction, n_hit):
    r"""R-value as defined in Räsänen et al. (2009), using percentages."""
    hr = 100 * n_hit / n_truth
    os = 100 * (n_prediction / n_truth - 1)
    r1 = math.sqrt((100 - hr) ** 2 + os**2)
    r2 = (-os + hr - 100) / math.sqrt(2)
    return 1 - (abs(r1) + abs(r2)) / 200


@pytest.mark.parametrize(
    "truth, prediction, tolerance, expected",
    [
        # Perfect segmentation
        ([0.1, 0.5, 0.9], [0.1, 0.5, 0.9], 0.02, 1.0),
        (np.array([0.1, 0.5, 0.9]), np.array([0.1, 0.5, 0.9]), 0.02, 1.0),
        # Unsorted input
        ([0.9, 0.1, 0.5], [0.5, 0.9, 0.1], 0.02, 1.0),
        # Within tolerance
        ([0.1, 0.5, 0.9], [0.115, 0.485, 0.915], 0.02, 1.0),
        # Outside tolerance
        ([0.1, 0.5, 0.9], [0.13, 0.5, 0.9], 0.02, r_value_reference(3, 3, 2)),
        # Larger tolerance
        ([0.1, 0.5, 0.9], [0.13, 0.5, 0.9], 0.05, 1.0),
        # Under-segmentation
        ([0.1, 0.5, 0.9], [0.1, 0.5], 0.02, r_value_reference(3, 2, 2)),
        # Over-segmentation
        (
            [0.1, 0.5, 0.9],
            [0.1, 0.3, 0.5, 0.7, 0.9],
            0.02,
            r_value_reference(3, 5, 3),
        ),
        # No prediction
        ([0.1, 0.5, 0.9], [], 0.02, r_value_reference(3, 0, 0)),
        # Two predictions close to the same truth boundary
        # only count as one hit
        ([0.5], [0.49, 0.51], 0.02, r_value_reference(1, 2, 1)),
        # One prediction close to two truth boundaries
        # only counts as one hit
        ([0.49, 0.51], [0.5], 0.02, r_value_reference(2, 1, 1)),
        # Greedy matching must not block a later hit
        ([0.10, 0.13], [0.115, 0.145], 0.02, 1.0),
        # Multiple sequences are pooled
        (
            [[0.1, 0.5], [0.2]],
            [[0.1, 0.5], [0.4]],
            0.02,
            r_value_reference(3, 3, 2),
        ),
        (
            [[0.1, 0.5], []],
            [[0.1, 0.5], [0.3]],
            0.02,
            r_value_reference(2, 3, 2),
        ),
        # Boundaries are not matched across sequences
        ([[0.1], [0.5]], [[0.5], [0.1]], 0.02, r_value_reference(2, 2, 0)),
    ],
)
def test_r_value(truth, prediction, tolerance, expected):
    r = audmetric.r_value(truth, prediction, tolerance=tolerance)
    assert isinstance(r, float)
    np.testing.assert_almost_equal(r, expected)


def test_r_value_paper_example():
    # HR = 100 %, OS = 0 % gives 1,
    # predicting a boundary at every possible position
    # gives a negative R-value
    truth = [0.5]
    prediction = list(np.arange(0, 1, 0.01))
    r = audmetric.r_value(truth, prediction)
    assert r < 0


@pytest.mark.parametrize(
    "truth, prediction, tolerance, error_msg",
    [
        ([], [0.1], 0.02, "at least one boundary"),
        ([[], []], [[0.1], []], 0.02, "at least one boundary"),
        ([0.1], [0.1], -0.01, "non-negative"),
        ([[0.1], [0.2]], [[0.1]], 0.02, "length"),
    ],
)
def test_r_value_errors(truth, prediction, tolerance, error_msg):
    with pytest.raises(ValueError, match=error_msg):
        audmetric.r_value(truth, prediction, tolerance=tolerance)
