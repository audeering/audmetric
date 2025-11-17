import os

import numpy as np
import pandas as pd
import pytest

import audeer
import audformat

import audmetric


REFERENCE_DIR = os.path.join(audeer.script_dir(), "assets", "event_based")


@pytest.mark.parametrize(
    (
        "truth, prediction, labels, zero_division, propagate_nans, "
        "onset_tol, offset_tol, duration_tol, "
        "expected_conf, expected_rpc, expected_ppc, expected_fpc, expected_f"
    ),
    [
        # Empty series
        (
            pd.Series(
                index=audformat.segmented_index(),
            ),
            pd.Series(
                index=audformat.segmented_index(),
            ),
            ["label"],
            0.0,
            False,
            0,
            0,
            0,
            [[0, 0], [0, 0]],
            {"label": 0.0},
            {"label": 0.0},
            {"label": 0.0},
            0.0,
        ),
        # Series with incorrect index passed
        pytest.param(
            pd.Series(),
            pd.Series(),
            None,
            0.0,
            False,
            0,
            0,
            0,
            None,
            None,
            None,
            None,
            None,
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
            None,
            0.0,
            False,
            0,
            0,
            0,
            [[0, 1, 0], [0, 0, 0], [0, 0, 0]],
            {"a": 0.0, "b": 0.0},
            {"a": 0.0, "b": 0.0},
            {"a": 0.0, "b": 0.0},
            0.0,
        ),
        # No overlap with nan
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
            None,
            np.nan,
            False,
            0,
            0,
            0,
            [[0, 1, 0], [0, 0, 0], [0, 0, 0]],
            {"a": 0.0, "b": np.nan},
            {"a": np.nan, "b": 0.0},
            {"a": 0.0, "b": 0.0},
            0.0,
        ),
        # Perfect overlap
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
                    starts=[0, 0.1, 0.2, 0.3],
                    ends=[0.1, 0.2, 0.3, 0.4],
                ),
                data=["speech"] * 4,
            ),
            ["speech"],
            0.0,
            False,
            0,
            0,
            0,
            [[4, 0], [0, 0]],
            {"speech": 1.0},
            {"speech": 1.0},
            {"speech": 1.0},
            1.0,
        ),
        # All segments overlap within tolerances
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
                    starts=[0.01, 0.08, 0.21, 0.3],
                    ends=[0.1, 0.19, 0.32, 0.424],
                ),
                data=["speech"] * 4,
            ),
            ["speech"],
            0.0,
            False,
            0.025,
            0.025,
            None,
            [[4, 0], [0, 0]],
            {"speech": 1.0},
            {"speech": 1.0},
            {"speech": 1.0},
            1.0,
        ),
        # All segments overlap within tolerances, multiclass and multifile,
        # and proportion based overlap
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
                    starts=[0.01, 0.1, 0.18, 0.3, 0.22, 0.401, 1.02],
                    ends=[0.11, 0.18, 0.3, 0.4, 0.37, 0.82, 2.7],
                ),
                data=["a", "b", "a", "a", "b", "b", "a"],
            ),
            None,
            0.0,
            False,
            0.025,
            0.025,
            0.2,
            [[4, 0, 0], [0, 3, 0], [0, 0, 0]],
            {"a": 1.0, "b": 1.0},
            {"a": 1.0, "b": 1.0},
            {"a": 1.0, "b": 1.0},
            1.0,
        ),
        # Overlapping start and end time points but some incorrect labels
        (
            pd.Series(
                index=audformat.segmented_index(
                    files=["f1.wav"] * 4 + ["f2.wav"] * 4,
                    starts=[0, 0.1, 0.2, 0.3, 0, 0.1, 0.2, 1.0],
                    ends=[0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.35, 3.0],
                ),
                data=["a", "b", "a", "a", "b", "a", "a", "a"],
            ),
            pd.Series(
                index=audformat.segmented_index(
                    files=["f1.wav"] * 4 + ["f2.wav"] * 4,
                    starts=[0.01, 0.1, 0.18, 0.3, 0.02, 0.08, 0.18, 1.02],
                    ends=[0.11, 0.18, 0.3, 0.4, 0.1, 0.2, 0.37, 2.7],
                ),
                data=["b", "b", "a", "a", "a", "a", "b", "b"],
            ),
            None,
            0.0,
            False,
            0.025,
            0.025,
            0.2,
            [[3, 3, 0], [1, 1, 0], [0, 0, 0]],
            {"a": 0.5, "b": 0.5},
            {"a": 0.75, "b": 0.25},
            {"a": 0.6, "b": 1 / 3},
            (0.6 + 1 / 3) / 2,
        ),
        # Include missing predictions and some false positive predictions
        (
            pd.Series(
                index=audformat.segmented_index(
                    files=["f1.wav"] * 2 + ["f2.wav"] * 4 + ["f3.wav"],
                    starts=[0, 0.1, 0.2, 0.4, 1.0, 3.5, 0.0],
                    ends=[0.1, 0.2, 0.35, 0.8, 3.0, 4.0, 1.0],
                ),
                data=["a", "b", "a", "b", "a", "c", "a"],
            ),
            pd.Series(
                index=audformat.segmented_index(
                    files=["f1.wav"] * 2 + ["f2.wav"] * 4,
                    starts=[0.01, 0.1, 0.22, 0.401, 1.02, 5.0],
                    ends=[0.11, 0.18, 0.37, 0.82, 2.7, 6.0],
                ),
                data=["b", "b", "a", "a", "a", "a"],
            ),
            None,
            0.0,
            False,
            0.025,
            0.025,
            0.2,
            [[2, 1, 0, 1], [1, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0]],
            {"a": 0.5, "b": 0.5, "c": 0.0},
            {"a": 0.5, "b": 0.5, "c": 0.0},
            {"a": 0.5, "b": 0.5, "c": 0.0},
            1 / 3,
        ),
        # NaN propagation set to True when only one of recall/precision is NaN
        (
            pd.Series(
                index=audformat.segmented_index(
                    files=["f1.wav"] * 4,
                    starts=[0, 0.1, 0.2, 0.3],
                    ends=[0.1, 0.2, 0.3, 0.4],
                ),
                data=["a", "a", "a", "a"],
            ),
            pd.Series(
                index=audformat.segmented_index(
                    files=["f1.wav"] * 4,
                    starts=[0.01, 0.1, 0.18, 0.3],
                    ends=[0.11, 0.18, 0.3, 0.4],
                ),
                data=["a", "b", "a", "a"],
            ),
            None,
            np.nan,
            True,
            0.025,
            0.025,
            0.2,
            [[3, 1, 0], [0, 0, 0], [0, 0, 0]],
            {"a": 0.75, "b": np.nan},
            {"a": 1.0, "b": 0.0},
            {"a": 2 * 0.75 / (1.0 + 0.75), "b": np.nan},
            2 * 0.75 / (1.0 + 0.75),
        ),
        # NaN propagation set to False when only one of recall/precision is NaN
        (
            pd.Series(
                index=audformat.segmented_index(
                    files=["f1.wav"] * 4,
                    starts=[0, 0.1, 0.2, 0.3],
                    ends=[0.1, 0.2, 0.3, 0.4],
                ),
                data=["a", "a", "a", "a"],
            ),
            pd.Series(
                index=audformat.segmented_index(
                    files=["f1.wav"] * 4,
                    starts=[0.01, 0.1, 0.18, 0.3],
                    ends=[0.11, 0.18, 0.3, 0.4],
                ),
                data=["a", "b", "a", "a"],
            ),
            None,
            np.nan,
            False,
            0.025,
            0.025,
            0.2,
            [[3, 1, 0], [0, 0, 0], [0, 0, 0]],
            {"a": 0.75, "b": np.nan},
            {"a": 1.0, "b": 0.0},
            {"a": 2 * 0.75 / (1.0 + 0.75), "b": 0.0},
            0.75 / (1.0 + 0.75),
        ),
    ],
)
def test_event_based_metrics(
    truth,
    prediction,
    labels,
    zero_division,
    propagate_nans,
    onset_tol,
    offset_tol,
    duration_tol,
    expected_conf,
    expected_rpc,
    expected_ppc,
    expected_fpc,
    expected_f,
):
    confusion = audmetric.event_confusion_matrix(
        truth,
        prediction,
        labels,
        onset_tolerance=onset_tol,
        offset_tolerance=offset_tol,
        duration_tolerance=duration_tol,
        normalize=False,
    )
    np.testing.assert_equal(confusion, expected_conf)
    norm_confusion = audmetric.event_confusion_matrix(
        truth,
        prediction,
        labels,
        onset_tolerance=onset_tol,
        offset_tolerance=offset_tol,
        duration_tolerance=duration_tol,
        normalize=True,
    )
    for i, row in enumerate(confusion):
        total_sum = sum(row)
        if total_sum != 0:
            for j, col in enumerate(row):
                np.testing.assert_almost_equal(norm_confusion[i][j], col / total_sum)

    rpc = audmetric.event_recall_per_class(
        truth,
        prediction,
        labels,
        zero_division=zero_division,
        onset_tolerance=onset_tol,
        offset_tolerance=offset_tol,
        duration_tolerance=duration_tol,
    )
    np.testing.assert_equal(rpc, expected_rpc)

    ppc = audmetric.event_precision_per_class(
        truth,
        prediction,
        labels,
        zero_division=zero_division,
        onset_tolerance=onset_tol,
        offset_tolerance=offset_tol,
        duration_tolerance=duration_tol,
    )
    np.testing.assert_equal(ppc, expected_ppc)

    fpc = audmetric.event_fscore_per_class(
        truth,
        prediction,
        labels,
        zero_division=zero_division,
        propagate_nans=propagate_nans,
        onset_tolerance=onset_tol,
        offset_tolerance=offset_tol,
        duration_tolerance=duration_tol,
    )
    np.testing.assert_equal(fpc, expected_fpc)

    uaf = audmetric.event_unweighted_average_fscore(
        truth,
        prediction,
        labels,
        zero_division=zero_division,
        propagate_nans=propagate_nans,
        onset_tolerance=onset_tol,
        offset_tolerance=offset_tol,
        duration_tolerance=duration_tol,
    )
    np.testing.assert_equal(uaf, expected_f)


@pytest.mark.parametrize("testcase", [0, 1, 2, 3, 4])
def test_sed_eval_comparison(testcase):
    # Test cases are generated in tests/assets/event_based
    # using the sed-eval package
    reference_dir = os.path.join(REFERENCE_DIR, str(testcase))
    truth = audformat.utils.read_csv(os.path.join(reference_dir, "truth.csv"))
    prediction = audformat.utils.read_csv(os.path.join(reference_dir, "prediction.csv"))
    expected_result = (
        pd.read_csv(os.path.join(reference_dir, "result.csv"), index_col=0)
        .transpose()
        .iloc[0]
    )
    expected_result = expected_result.replace({np.nan: None})
    n_labels = int(expected_result["n_labels"])
    labels = [f"label_{i}" for i in range(n_labels)]
    confusion = audmetric.event_confusion_matrix(
        truth,
        prediction,
        labels,
        onset_tolerance=expected_result["onset_tol"],
        offset_tolerance=expected_result["offset_tol"],
        duration_tolerance=expected_result["duration_tol"],
        normalize=False,
    )
    uaf = audmetric.event_unweighted_average_fscore(
        truth,
        prediction,
        labels,
        zero_division=np.nan,
        propagate_nans=True,
        onset_tolerance=expected_result["onset_tol"],
        offset_tolerance=expected_result["offset_tol"],
        duration_tolerance=expected_result["duration_tol"],
    )
    confusion = np.array(confusion)
    n_truth = confusion.sum(axis=1)
    n_pred = confusion.sum(axis=0)
    for i, label in enumerate(labels):
        n_tp = confusion[i, i]
        n_fn = n_truth[i] - n_tp
        n_fp = n_pred[i] - n_tp
        assert expected_result[f"{label}.Nref"] == n_truth[i]
        assert expected_result[f"{label}.Nsys"] == n_pred[i]
        assert expected_result[f"{label}.Ntp"] == n_tp
        assert expected_result[f"{label}.Nfn"] == n_fn
        assert expected_result[f"{label}.Nfp"] == n_fp
        assert n_truth[i] == len(truth[truth == label])
        assert n_pred[i] == len(prediction[prediction == label])
    np.testing.assert_almost_equal(uaf, expected_result["uaf"])
