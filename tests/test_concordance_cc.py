import numpy as np
import pandas as pd
import pytest

import audmetric


def expected_ccc(truth, prediction):
    r"""Expected Concordance Correlation Coefficient.

    This is a direct implementation of its math equation.

    """
    prediction = np.array(list(prediction))
    truth = np.array(list(truth))

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


@pytest.mark.parametrize("ignore_nan", [True, False])
@pytest.mark.parametrize(
    "truth, prediction",
    [
        # NOTE: this test assumes
        # that all truth and prediction values
        # do not contain NaN
        (
            np.random.randint(0, 10, size=5),
            np.random.randint(0, 10, size=5),
        ),
        (
            pd.Series(np.random.randint(0, 10, size=5)).astype("Int64"),
            pd.Series(np.random.randint(0, 10, size=5)).astype("Int64"),
        ),
        (
            np.random.randint(0, 10, size=1),
            np.random.randint(0, 10, size=1),
        ),
        (
            np.random.randint(0, 10, size=10),
            np.random.randint(0, 10, size=10),
        ),
        (
            np.random.randint(0, 2, size=100),
            np.random.randint(0, 2, size=100),
        ),
        (
            np.array([]),
            np.array([]),
        ),
        (
            np.zeros(10),
            np.zeros(10),
        ),
    ],
)
def test_concordance_cc(truth, prediction, ignore_nan):
    ccc = audmetric.concordance_cc(truth, prediction, ignore_nan=ignore_nan)

    np.testing.assert_almost_equal(
        ccc,
        expected_ccc(truth, prediction),
    )


@pytest.mark.parametrize(
    "truth, prediction, ignore_nan, expected",
    [
        (
            [],
            [],
            True,
            np.NaN,
        ),
        (
            [],
            [],
            False,
            np.NaN,
        ),
        (
            [0],
            [0],
            True,
            np.NaN,
        ),
        (
            [0],
            [0],
            False,
            np.NaN,
        ),
        (
            [0, np.NaN],
            [0, np.NaN],
            True,
            np.NaN,
        ),
        (
            [0, np.NaN],
            [0, np.NaN],
            False,
            np.NaN,
        ),
        (
            [0, 1, 2, 3],
            [1, 2, 3, 4],
            True,
            expected_ccc([0, 1, 2, 3], [1, 2, 3, 4]),
        ),
        (
            [np.NaN, 1, 2, 3],
            [np.NaN, 2, 3, 4],
            True,
            expected_ccc([1, 2, 3], [2, 3, 4]),
        ),
        (
            [np.NaN, 1, 2, 3],
            [1, 2, 3, np.NaN],
            True,
            expected_ccc([1, 2], [2, 3]),
        ),
        (
            [0, np.NaN, 2, 3],
            [1, 2, 3, 4],
            True,
            expected_ccc([0, 2, 3], [1, 3, 4]),
        ),
        (
            [0, 1, 2, 3],
            [1, 2, np.NaN, 4],
            True,
            expected_ccc([0, 1, 3], [1, 2, 4]),
        ),
        (
            [np.NaN, np.NaN, 2, 3],
            [1, 2, 3, np.NaN],
            True,
            expected_ccc([2], [3]),
        ),
        (
            [np.NaN, np.NaN, 2, 3],
            [1, 2, 3, np.NaN],
            False,
            np.NaN,
        ),
    ],
)
def test_concordance_cc_ignore_nan(
    truth,
    prediction,
    ignore_nan,
    expected,
):
    ccc = audmetric.concordance_cc(truth, prediction, ignore_nan=ignore_nan)

    np.testing.assert_almost_equal(
        ccc,
        expected,
    )
