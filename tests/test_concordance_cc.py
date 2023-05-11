import numpy as np
import pandas as pd
import pytest

import audmetric


def expected_ccc(truth, prediction, ignore_nan):
    r"""Expecte Concordance Correlation Coefficient.

    This is a direct implementation of its math equation.

    If only a single sample is given,
    it should return NaN.

    """
    prediction = np.array(list(prediction))
    truth = np.array(list(truth))

    if ignore_nan:
        mask = ~(np.isnan(truth) | np.isnan(prediction))
        truth = truth[mask]
        prediction = prediction[mask]

    if len(prediction) < 2:
        ccc = np.NaN
    else:
        denominator = (
            prediction.std() ** 2
            + truth.std() ** 2
            + (prediction.mean() - truth.mean()) ** 2
        )
        if denominator == 0:
            ccc = np.NaN
        else:
            r = np.corrcoef(list(prediction), list(truth))[0][1]
            numerator = 2 * r * prediction.std() * truth.std()
            ccc = numerator / denominator
    return ccc


@pytest.mark.parametrize(
    'truth, prediction, ignore_nan',
    [
        (
            np.random.randint(0, 10, size=5),
            np.random.randint(0, 10, size=5),
            False,
        ),
        (
            pd.Series(np.random.randint(0, 10, size=5)).astype('Int64'),
            pd.Series(np.random.randint(0, 10, size=5)).astype('Int64'),
            False,
        ),
        (
            np.random.randint(0, 10, size=1),
            np.random.randint(0, 10, size=1),
            False,
        ),
        (
            np.random.randint(0, 10, size=10),
            np.random.randint(0, 10, size=10),
            False,
        ),
        (
            np.random.randint(0, 2, size=100),
            np.random.randint(0, 2, size=100),
            False,
        ),
        (
            np.array([]),
            np.array([]),
            False,
        ),
        (
            np.zeros(10),
            np.zeros(10),
            False,
        ),
        (
            [0, 1, 2, 3, 4, 5, 6, np.NaN],
            [0, 2, 3, 5, 6, 7, 7, np.NaN],
            False,
        ),
        (
            [0, 1, 2, 3, 4, 5, 6, np.NaN],
            [0, 2, 3, 5, 6, 7, 7, np.NaN],
            True,
        ),
    ]
)
def test_concordance_cc(truth, prediction, ignore_nan):

    ccc = audmetric.concordance_cc(truth, prediction, ignore_nan=ignore_nan)

    np.testing.assert_almost_equal(
        ccc,
        expected_ccc(truth, prediction, ignore_nan),
    )


@pytest.mark.parametrize('ignore_nan', [True, False])
@pytest.mark.parametrize(
    'truth, prediction',
    [
        (
            [],
            [],
        ),
        (
            [0],
            [0],
        ),
        (
            [0, np.NaN],
            [0, np.NaN],
        ),
    ]
)
def test_concordance_cc_expected_nan(truth, prediction, ignore_nan):
    ccc = audmetric.concordance_cc(truth, prediction, ignore_nan=ignore_nan)
    assert np.isnan(ccc)
