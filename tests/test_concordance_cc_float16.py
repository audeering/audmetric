import numpy as np
import pytest

import audmetric


def test_float16_perfect_concordance():
    values = np.resize(np.array([-1, 1], dtype=np.float16), 40000)
    with np.errstate(over="raise", invalid="raise"):
        assert audmetric.concordance_cc(values, values) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "size, amplitude",
    [(40000, 1.0), (80000, 0.125), (4, 65504.0)],
)
@pytest.mark.parametrize(
    "truth_dtype, prediction_dtype",
    [(np.float16, np.float16), (np.float16, np.float32), (np.float64, np.float16)],
)
@pytest.mark.parametrize("ignore_nan", [False, True])
def test_float16_concordance(
    size, amplitude, truth_dtype, prediction_dtype, ignore_nan
):
    truth = np.resize(np.array([-amplitude, amplitude]), size).astype(truth_dtype)
    prediction = (truth / 2).astype(prediction_dtype)
    if ignore_nan:
        truth = np.append(truth, truth_dtype(np.nan))
        prediction = np.append(prediction, prediction_dtype(np.nan))
    truth_before = truth.copy()
    prediction_before = prediction.copy()
    truth.flags.writeable = False
    prediction.flags.writeable = False

    with np.errstate(over="raise", invalid="raise"):
        result = audmetric.concordance_cc(truth, prediction, ignore_nan=ignore_nan)

    # For a zero-mean sequence and a gain of 1/2, CCC = 2 * (1/2) / (1 + 1/4).
    assert result == pytest.approx(0.8, abs=1e-6)
    np.testing.assert_array_equal(truth, truth_before)
    np.testing.assert_array_equal(prediction, prediction_before)


@pytest.mark.parametrize("values", [[], [1], [1, 1]])
def test_float16_undefined_concordance(values):
    values = np.array(values, dtype=np.float16)
    assert np.isnan(audmetric.concordance_cc(values, values))
