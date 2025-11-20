import os

import numpy as np
import pandas as pd
import pytest

import audeer
import audformat

import audmetric


REFERENCE_DIR = os.path.join(audeer.script_dir(), "assets", "error_rates")


@pytest.mark.parametrize(
    ("truth, prediction, expected_der"),
    [
        # Empty series
        (
            pd.Series(
                index=audformat.segmented_index(),
            ),
            pd.Series(
                index=audformat.segmented_index(),
            ),
            0.0,
        ),
        # Series with incorrect index passed
        pytest.param(
            pd.Series(),
            pd.Series(),
            0.0,
            marks=pytest.mark.xfail(raises=ValueError),
        ),
        # Perfect overlap
        (
            pd.Series(
                index=audformat.segmented_index(
                    files=["f1.wav"],
                    starts=[0],
                    ends=[0.1],
                ),
                data=["a"],
            ),
            pd.Series(
                index=audformat.segmented_index(
                    files=["f1.wav"],
                    starts=[0],
                    ends=[0.1],
                ),
                data=["b"],
            ),
            0.0,
        ),
        # Partial overlap
        (
            pd.Series(
                index=audformat.segmented_index(
                    files=["f1.wav"] * 4,
                    starts=[0, 0.1, 0.2, 0.3],
                    ends=[0.1, 0.2, 0.3, 0.4],
                ),
                data=["a", "a", "b", "b"],
            ),
            pd.Series(
                index=audformat.segmented_index(
                    files=["f1.wav"] * 4,
                    starts=[0.01, 0.12, 0.21, 0.3],
                    ends=[0.1, 0.2, 0.3, 0.42],
                ),
                data=["c", "c", "d", "d"],
            ),
            0.06 / 0.4,
        ),
        # Partial overlap with multiclass
        (
            pd.Series(
                index=audformat.segmented_index(
                    files=["f1.wav"] * 4 + ["f2.wav"] * 3,
                    starts=[0, 0.1, 0.2, 0.3, 0.2, 0.4, 1.0],
                    ends=[0.1, 0.2, 0.3, 0.4, 0.35, 0.8, 3.0],
                ),
                data=["a", "b", "a", "a", "b", "b", "a"],
            ),
            pd.Series(
                index=audformat.segmented_index(
                    files=["f1.wav"] * 4 + ["f2.wav"] * 3,
                    starts=[0.01, 0.1, 0.18, 0.3, 0.22, 0.4, 1.02],
                    ends=[0.11, 0.18, 0.3, 0.4, 0.37, 0.82, 2.7],
                ),
                data=["c", "d", "c", "c", "d", "d", "c"],
            ),
            0.42 / 2.95,
        ),
        # Confusions, misses and false alarms
        (
            pd.Series(
                index=audformat.segmented_index(
                    files=["f1.wav"] * 2 + ["f2.wav"] * 4 + ["f3.wav"],
                    starts=[0, 0.1, 0.2, 0.4, 1.0, 3.5, 0.0],
                    ends=[0.1, 0.2, 0.35, 0.8, 3.0, 4.0, 1.0],
                ),
                data=["a", "b", "a", "a", "a", "c", "a"],
            ),
            pd.Series(
                index=audformat.segmented_index(
                    files=["f1.wav"] * 2 + ["f2.wav"] * 4,
                    starts=[0.01, 0.1, 0.22, 0.4, 1.0, 5.0],
                    ends=[0.09, 0.2, 0.35, 0.8, 2.8, 6.0],
                ),
                data=["d", "d", "c", "c", "c", "c"],
            ),
            2.82 / 4.25,
        ),
    ],
)
def test_der(
    truth,
    prediction,
    expected_der,
):
    der = audmetric.diarization_error_rate(
        truth,
        prediction,
    )
    np.testing.assert_almost_equal(der, expected_der)


@pytest.mark.parametrize(
    ("truth, prediction, expected_ier"),
    [
        # Empty series
        (
            pd.Series(
                index=audformat.segmented_index(),
            ),
            pd.Series(
                index=audformat.segmented_index(),
            ),
            0.0,
        ),
        # Series with incorrect index passed
        pytest.param(
            pd.Series(),
            pd.Series(),
            0.0,
            marks=pytest.mark.xfail(raises=ValueError),
        ),
        # No overlap
        (
            pd.Series(
                index=audformat.segmented_index(
                    files=["f1.wav"],
                    starts=[0],
                    ends=[0.1],
                ),
                data=["a"],
            ),
            pd.Series(
                index=audformat.segmented_index(
                    files=["f1.wav"],
                    starts=[0],
                    ends=[0.1],
                ),
                data=["b"],
            ),
            1.0,
        ),
        # Perfect overlap
        (
            pd.Series(
                index=audformat.segmented_index(
                    files=["f1.wav"] * 4,
                    starts=[0, 0.1, 0.2, 0.3],
                    ends=[0.1, 0.2, 0.3, 0.4],
                ),
                data=["a"] * 4,
            ),
            pd.Series(
                index=audformat.segmented_index(
                    files=["f1.wav"] * 4,
                    starts=[0, 0.1, 0.2, 0.3],
                    ends=[0.1, 0.2, 0.3, 0.4],
                ),
                data=["a"] * 4,
            ),
            0.0,
        ),
        # Partial overlap
        (
            pd.Series(
                index=audformat.segmented_index(
                    files=["f1.wav"] * 4,
                    starts=[0, 0.1, 0.2, 0.3],
                    ends=[0.1, 0.2, 0.3, 0.4],
                ),
                data=["speech"] * 4,
            ),
            pd.Series(
                index=audformat.segmented_index(
                    files=["f1.wav"] * 4,
                    starts=[0.01, 0.12, 0.21, 0.3],
                    ends=[0.1, 0.2, 0.3, 0.42],
                ),
                data=["speech"] * 4,
            ),
            0.06 / 0.4,
        ),
        # Partial overlap with multiclass
        (
            pd.Series(
                index=audformat.segmented_index(
                    files=["f1.wav"] * 4 + ["f2.wav"] * 3,
                    starts=[0, 0.1, 0.2, 0.3, 0.2, 0.4, 1.0],
                    ends=[0.1, 0.2, 0.3, 0.4, 0.35, 0.8, 3.0],
                ),
                data=["a", "b", "a", "a", "b", "b", "a"],
            ),
            pd.Series(
                index=audformat.segmented_index(
                    files=["f1.wav"] * 4 + ["f2.wav"] * 3,
                    starts=[0.01, 0.1, 0.18, 0.3, 0.22, 0.4, 1.02],
                    ends=[0.11, 0.18, 0.3, 0.4, 0.37, 0.82, 2.7],
                ),
                data=["a", "b", "a", "a", "b", "b", "a"],
            ),
            0.42 / 2.95,
        ),
        # Confusions, misses and false alarms
        (
            pd.Series(
                index=audformat.segmented_index(
                    files=["f1.wav"] * 2 + ["f2.wav"] * 4 + ["f3.wav"],
                    starts=[0, 0.1, 0.2, 0.4, 1.0, 3.5, 0.0],
                    ends=[0.1, 0.2, 0.35, 0.8, 3.0, 4.0, 1.0],
                ),
                data=["a", "b", "a", "a", "a", "c", "a"],
            ),
            pd.Series(
                index=audformat.segmented_index(
                    files=["f1.wav"] * 2 + ["f2.wav"] * 4,
                    starts=[0.01, 0.1, 0.22, 0.4, 1.0, 5.0],
                    ends=[0.09, 0.2, 0.35, 0.8, 2.8, 6.0],
                ),
                data=["b", "b", "a", "a", "a", "a"],
            ),
            2.82 / 4.25,
        ),
    ],
)
def test_ier(
    truth,
    prediction,
    expected_ier,
):
    ier = audmetric.identification_error_rate(
        truth,
        prediction,
    )
    np.testing.assert_almost_equal(ier, expected_ier)


@pytest.mark.parametrize("testcase", [0, 1, 2, 3, 4])
def test_pyannote_ier(testcase):
    # Test cases are generated in tests/assets/error_rates/ier
    # using pyannote
    reference_dir = os.path.join(REFERENCE_DIR, "ier", str(testcase))
    truth = audformat.utils.read_csv(os.path.join(reference_dir, "truth.csv"))
    prediction = audformat.utils.read_csv(os.path.join(reference_dir, "prediction.csv"))
    expected_result = (
        pd.read_csv(os.path.join(reference_dir, "result.csv"), index_col=0)
        .transpose()
        .iloc[0]
    )
    expected_result = expected_result.replace({np.nan: None})
    ier = audmetric.identification_error_rate(truth, prediction)
    np.testing.assert_almost_equal(ier, expected_result["ier"], decimal=5)


@pytest.mark.parametrize("testcase", [0, 1, 2, 3, 4])
def test_pyannote_der(testcase):
    # Test cases are generated in tests/assets/error_rates/der
    # using pyannote
    reference_dir = os.path.join(REFERENCE_DIR, "der", str(testcase))
    truth = audformat.utils.read_csv(os.path.join(reference_dir, "truth.csv"))
    prediction = audformat.utils.read_csv(os.path.join(reference_dir, "prediction.csv"))
    expected_result = (
        pd.read_csv(os.path.join(reference_dir, "result.csv"), index_col=0)
        .transpose()
        .iloc[0]
    )
    expected_result = expected_result.replace({np.nan: None})
    der = audmetric.diarization_error_rate(truth, prediction)
    np.testing.assert_almost_equal(der, expected_result["der"], decimal=5)
