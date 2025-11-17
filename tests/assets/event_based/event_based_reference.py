# /// script
# requires-python = "<=3.10"
# dependencies = [
#   "audeer",
#   "audformat",
#   "numpy",
#   "pandas",
#   "sed-eval",
#   "setuptools<71",
# ]
# ///

import os
import random

import numpy as np
import pandas as pd
import sed_eval

import audeer
import audformat


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


def main():
    random.seed(1)
    np.random.seed(1)
    current_dir = audeer.script_dir()
    test_cases = [
        (10, 10, 1, 0.2, 0.2, 0.4, 0.5, 5.0),
        (10, 30, 5, 1.0, 1.0, None, 5.0, 30.0),
        (50, 60, 2, 0.3, None, 0.4, 5.0, 30.0),
        (20, 20, 2, None, 0.2, 0.4, 1.0, 10.0),
        (100, 100, 10, 0.1, 0.1, 0.3, 0.5, 5.0),
    ]
    for i, (
        n_pred,
        n_truth,
        n_labels,
        onset_tol,
        offset_tol,
        duration_tol,
        segment_duration_max,
        file_duration,
    ) in enumerate(test_cases):
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
        segment_dir = os.path.join(current_dir, str(i))
        audeer.mkdir(segment_dir)
        truth.to_csv(os.path.join(segment_dir, "truth.csv"))
        prediction.to_csv(os.path.join(segment_dir, "prediction.csv"))

        sed_metrics = sed_eval.sound_event.EventBasedMetrics(
            event_label_list=labels,
            evaluate_onset=onset_tol is not None,
            evaluate_offset=offset_tol is not None or duration_tol is not None,
            # For sed_eval, onset_tolerance and offset_tolerance have to be the same
            # if both are evaluated
            t_collar=onset_tol if onset_tol is not None else offset_tol,
            percentage_of_length=duration_tol if duration_tol is not None else 0.0,
            event_matching_type="greedy",
        )
        truth_event_list = eventlist_from_series(truth)
        pred_event_list = eventlist_from_series(prediction)
        sed_metrics.evaluate(
            reference_event_list=truth_event_list, estimated_event_list=pred_event_list
        )
        expected_uaf = sed_metrics.results_class_wise_average_metrics()["f_measure"][
            "f_measure"
        ]
        result = {
            "n_labels": n_labels,
            "onset_tol": onset_tol,
            "offset_tol": offset_tol,
            "duration_tol": duration_tol,
            "uaf": expected_uaf,
        }
        for label in labels:
            for measure in ["Nref", "Nsys", "Ntp", "Nfn", "Nfp"]:
                result[f"{label}.{measure}"] = sed_metrics.class_wise[label][measure]
        result_metrics = pd.Series(result)
        result_metrics.to_csv(os.path.join(segment_dir, "result.csv"))


if __name__ == "__main__":
    main()
