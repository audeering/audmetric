import numpy as np
import pytest

import audmetric


@pytest.mark.parametrize(
    "metric, power",
    [(audmetric.mean_absolute_error, 1), (audmetric.mean_squared_error, 2)],
)
@pytest.mark.parametrize(
    "dtype",
    [np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64],
)
def test_integer_limits(metric, power, dtype):
    limits = np.iinfo(dtype)
    truth = np.array([limits.min, limits.max, 0], dtype=dtype)
    prediction = np.array([limits.max, limits.min, limits.max], dtype=dtype)
    originals = truth.copy(), prediction.copy()
    expected = sum(
        abs(int(t) - int(p)) ** power for t, p in zip(truth, prediction)
    ) / len(truth)
    assert metric(truth, prediction) == pytest.approx(expected, rel=1e-15)
    np.testing.assert_array_equal(truth, originals[0])
    np.testing.assert_array_equal(prediction, originals[1])


@pytest.mark.parametrize(
    "metric",
    [audmetric.mean_absolute_error, audmetric.mean_squared_error],
)
@pytest.mark.parametrize(
    "dtype, origin",
    [(np.int64, -(2**63)), (np.int64, 2**63 - 4), (np.uint64, 2**64 - 4)],
)
def test_large_integer_small_difference(metric, dtype, origin):
    truth = np.array([origin, origin + 2], dtype=dtype)
    prediction = np.array([origin + 1, origin + 3], dtype=dtype)
    assert metric(truth, prediction) == 1.0


@pytest.mark.parametrize(
    "metric, power",
    [(audmetric.mean_absolute_error, 1), (audmetric.mean_squared_error, 2)],
)
def test_mixed_integer_limits(metric, power):
    truth = np.array([-(2**63), 2**63 - 1], dtype=np.int64)
    prediction = np.array([2**64 - 1, 2**63], dtype=np.uint64)
    expected = sum(
        abs(int(t) - int(p)) ** power for t, p in zip(truth, prediction)
    ) / len(truth)
    assert metric(truth, prediction) == pytest.approx(expected, rel=1e-15)


@pytest.mark.parametrize(
    "metric, power",
    [(audmetric.mean_absolute_error, 1), (audmetric.mean_squared_error, 2)],
)
@pytest.mark.parametrize(
    "truth_dtype, prediction_dtype",
    [(np.float32, np.float32), (np.float64, np.float64), (np.int16, np.float32)],
)
def test_floating_computation_preserved(metric, power, truth_dtype, prediction_dtype):
    truth = np.array([0, 1, 2, 5], dtype=truth_dtype)
    prediction = np.array([0.125, 1.5, 3.25, 4.75], dtype=prediction_dtype)
    difference = truth - prediction
    expected = float(np.mean(np.abs(difference) ** power))
    assert metric(truth, prediction) == expected
