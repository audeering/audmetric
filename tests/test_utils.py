import numpy as np
import pytest

import audmetric


@pytest.mark.parametrize(
    'truth,prediction,protected_variable,metric,labels,subgroups,'
    'zero_division,expected',
    [
        pytest.param(
            [], [], [], None, [], [0], None, {},
            marks=pytest.mark.xfail(raises=ValueError)
        ),
        (
            [], [], [], audmetric.recall_per_class, [], [], 0., {}
        ),
        (
            [1], [0], [0], audmetric.recall_per_class, [0, 1],
            [0], 0., {0: {0: 0.0, 1: 0.0}}
        ),
        (
            [1], [0], [0], audmetric.precision_per_class, [0, 1],
            [0], 0., {0: {0: 0.0, 1: 0.0}}
        ),
        (
            [1, 1], [0, 1], [0, 1], audmetric.recall_per_class, [0, 1],
            [0, 1], np.nan, {0: {0: np.nan, 1: 0.0}, 1: {0: np.nan, 1: 1.0}}
        ),
        (
            [1, 1], [0, 1], [0, 1], audmetric.recall_per_class, [1],
            [0, 1], np.nan, {0: {1: np.nan}, 1: {1: 1.0}}
        ),
        (
            [1, 1], [0, 1], [0, 1], audmetric.precision_per_class, [0, 1],
            [0, 1], np.nan, {0: {0: 0.0, 1: np.nan}, 1: {0: np.nan, 1: 1.0}}
        )
    ]
)
def test_scores_per_subgroup_and_class(
        truth, prediction, protected_variable, metric, labels, subgroups,
        zero_division, expected):
    np.testing.assert_equal(
        audmetric.core.utils.scores_per_subgroup_and_class(
            truth, prediction, protected_variable, metric,
            labels=labels,
            subgroups=subgroups,
            zero_division=zero_division
        ), expected
    )
