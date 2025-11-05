import random

import numpy as np
import pandas as pd
import pytest
import sed_eval

import audformat

import audmetric


def generate_random_segments(
    n_segments, labels, file_duration, min_duration, max_duration
):
    starts = np.random.uniform(
        low=0, high=file_duration - max_duration, size=n_segments
    )
    durations = np.random.uniform(low=min_duration, high=max_duration, size=n_segments)
    index = audformat.segmented_index(
        files=["f1.wav"] * n_segments,
        starts=starts,
        ends=starts + durations,
    )
    segments = pd.Series(index=index, data=random.choices(labels, k=n_segments))
    segments.name = "label"
    return segments


def eventlist_from_series(series: pd.Series):
    result = series.to_frame()
    result.reset_index(inplace=True)
    result.rename(
        columns={"start": "event_onset", "end": "event_offset", "label": "event_label"},
        inplace=True,
    )
    result["event_onset"] = result["event_onset"].dt.total_seconds()
    result["event_offset"] = result["event_offset"].dt.total_seconds()
    result = result.to_dict("records")
    return result


@pytest.mark.parametrize(
    (
        "truth, prediction, labels, zero_division, "
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
                    starts=[
                        0,
                    ],
                    ends=[0.1],
                ),
                data=["a"],
            ),
            pd.Series(
                index=audformat.segmented_index(
                    files=["f1.wav"],
                    starts=[
                        0,
                    ],
                    ends=[
                        0.1,
                    ],
                ),
                data=["b"],
            ),
            None,
            0.0,
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
                    starts=[
                        0,
                    ],
                    ends=[0.1],
                ),
                data=["a"],
            ),
            pd.Series(
                index=audformat.segmented_index(
                    files=["f1.wav"],
                    starts=[
                        0,
                    ],
                    ends=[
                        0.1,
                    ],
                ),
                data=["b"],
            ),
            None,
            np.nan,
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
            0.025,
            0.025,
            0.2,
            [[2, 1, 0, 1], [1, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0]],
            {"a": 0.5, "b": 0.5, "c": 0.0},
            {"a": 0.5, "b": 0.5, "c": 0.0},
            {"a": 0.5, "b": 0.5, "c": 0.0},
            1 / 3,
        ),
    ],
)
def test_event_based_metrics(
    truth,
    prediction,
    labels,
    zero_division,
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
                assert norm_confusion[i][j] == col / total_sum

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
        onset_tolerance=onset_tol,
        offset_tolerance=offset_tol,
        duration_tolerance=duration_tol,
    )
    np.testing.assert_equal(uaf, expected_f)


@pytest.mark.parametrize("n_truth", [10, 30, 100])
@pytest.mark.parametrize("n_pred", [10, 30, 100])
@pytest.mark.parametrize("n_labels", [1, 10])
@pytest.mark.parametrize(
    "onset_tol, offset_tol, duration_tol",
    [(0.2, 0.2, 0.4), (1.0, 1.0, None), (0.2, None, None), (None, 0.2, 0.5)],
)
@pytest.mark.parametrize(
    "segment_duration_max, file_duration", [(0.5, 5.0), (5.0, 30.0)]
)
def test_sed_eval_comparison(
    n_truth,
    n_pred,
    n_labels,
    onset_tol,
    offset_tol,
    duration_tol,
    segment_duration_max,
    file_duration,
):
    min_duration = 0
    if onset_tol is not None:
        min_duration = onset_tol
    if offset_tol is not None:
        min_duration = max(min_duration, offset_tol)
    labels = [f"label_{i}" for i in range(n_labels)]

    truth = generate_random_segments(
        n_truth,
        labels,
        file_duration=file_duration,
        min_duration=min_duration,
        max_duration=segment_duration_max,
    )
    prediction = generate_random_segments(
        n_pred,
        labels,
        file_duration=file_duration,
        min_duration=min_duration,
        max_duration=segment_duration_max,
    )

    sed_metrics = sed_eval.sound_event.EventBasedMetrics(
        event_label_list=labels,
        evaluate_onset=onset_tol is not None,
        evaluate_offset=offset_tol is not None or duration_tol is not None,
        # For sed_eval, onset_tolerance and offset_tolerance have to be the same
        t_collar=onset_tol if onset_tol is not None else offset_tol,
        percentage_of_length=duration_tol if duration_tol is not None else 0.0,
        event_matching_type="optimal",
    )
    truth_event_list = eventlist_from_series(truth)
    pred_event_list = eventlist_from_series(prediction)
    sed_metrics.evaluate(
        reference_event_list=truth_event_list, estimated_event_list=pred_event_list
    )
    confusion = audmetric.event_confusion_matrix(
        truth,
        prediction,
        labels,
        onset_tolerance=onset_tol,
        offset_tolerance=offset_tol,
        duration_tolerance=duration_tol,
        normalize=False,
    )
    uaf = audmetric.event_unweighted_average_fscore(
        truth,
        prediction,
        labels,
        zero_division=np.nan,
        propagate_nans=True,
        onset_tolerance=onset_tol,
        offset_tolerance=offset_tol,
        duration_tolerance=duration_tol,
    )
    confusion = np.array(confusion)
    n_truth = confusion.sum(axis=1)
    n_pred = confusion.sum(axis=0)
    for i, label in enumerate(labels):
        n_tp = confusion[i, i]
        n_fn = n_truth[i] - n_tp
        n_fp = n_pred[i] - n_tp
        assert sed_metrics.class_wise[label]["Nref"] == n_truth[i]
        assert sed_metrics.class_wise[label]["Nsys"] == n_pred[i]
        assert sed_metrics.class_wise[label]["Ntp"] == n_tp
        assert sed_metrics.class_wise[label]["Nfn"] == n_fn
        assert sed_metrics.class_wise[label]["Nfp"] == n_fp

    expected_uaf = sed_metrics.results_class_wise_average_metrics()["f_measure"][
        "f_measure"
    ]
    assert uaf == expected_uaf
