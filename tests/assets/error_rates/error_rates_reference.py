# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "audeer",
#   "audformat",
#   "numpy",
#   "pandas",
#   "pyannote.metrics",
#   "typing_extensions",
# ]
# ///

import os
import random

import numpy as np
import pandas as pd
from pyannote.core import Annotation
from pyannote.core import Segment
from pyannote.metrics.diarization import GreedyDiarizationErrorRate
from pyannote.metrics.identification import IdentificationErrorRate

import audeer
import audformat


def generate_random_segments(
    max_segments_per_label, labels, file_duration, min_duration, max_duration
):
    starts = []
    ends = []
    label_values = []
    for label in labels:
        end = 0
        segment_counter = 0
        while (
            end + min_duration < file_duration
            and segment_counter < max_segments_per_label
        ):
            # Create non-overlapping segments per speaker
            start = np.random.uniform(
                low=end, high=max(end, file_duration - min_duration)
            )
            end = np.random.uniform(
                low=start + min_duration, high=min(file_duration, start + max_duration)
            )
            starts.append(start)
            ends.append(end)
            label_values.append(label)
            segment_counter += 1
    index = audformat.segmented_index(
        files=["f1.wav"] * len(starts),
        starts=starts,
        ends=ends,
    )
    segments = pd.Series(index=index, data=label_values)
    segments.name = "label"
    return segments


def annotation_from_series(series: pd.Series) -> Annotation:
    result = Annotation()
    for (file, start, end), label in series.items():
        result[Segment(start.total_seconds(), end.total_seconds())] = label
    return result


def generate_der(result_dir):
    test_cases = [
        (20, 1, 0.5, 0.75, 10.0),
        (10, 2, 4.0, 5.0, 60.0),
        (50, 5, 4.0, 5.0, 120.0),
        (100, 2, 0.5, 10.0, 600.0),
        (100, 10, 0.5, 5.0, 600.0),
    ]
    for i, (
        max_segments_per_label,
        n_labels,
        segment_duration_min,
        segment_duration_max,
        file_duration,
    ) in enumerate(test_cases):
        # Use different labels for truth and prediction
        # because the mapping needs to be determined by the metric
        truth_labels = [f"label_{i}" for i in range(n_labels)]
        pred_labels = [f"pred_label_{i}" for i in range(n_labels)]
        truth = generate_random_segments(
            max_segments_per_label,
            truth_labels,
            file_duration=file_duration,
            min_duration=segment_duration_min,
            max_duration=segment_duration_max,
        )
        prediction = generate_random_segments(
            max_segments_per_label,
            pred_labels,
            file_duration=file_duration,
            min_duration=segment_duration_min,
            max_duration=segment_duration_max,
        )
        segment_dir = os.path.join(result_dir, str(i))
        audeer.mkdir(segment_dir)
        truth.to_csv(os.path.join(segment_dir, "truth.csv"))
        prediction.to_csv(os.path.join(segment_dir, "prediction.csv"))

        truth_annotation = annotation_from_series(truth)
        pred_annotation = annotation_from_series(prediction)
        der_metric = GreedyDiarizationErrorRate()

        der = der_metric(truth_annotation, pred_annotation)

        result = {
            "der": der,
        }
        result_metrics = pd.Series(result)
        result_metrics.to_csv(os.path.join(segment_dir, "result.csv"))


def generate_ier(result_dir):
    test_cases = [
        (20, 1, 0.5, 0.75, 10.0),
        (10, 2, 4.0, 5.0, 60.0),
        (50, 5, 4.0, 5.0, 120.0),
        (100, 2, 0.5, 10.0, 600.0),
        (100, 10, 0.5, 5.0, 600.0),
    ]
    for i, (
        max_segments_per_label,
        n_labels,
        segment_duration_min,
        segment_duration_max,
        file_duration,
    ) in enumerate(test_cases):
        labels = [f"label_{i}" for i in range(n_labels)]
        truth = generate_random_segments(
            max_segments_per_label,
            labels,
            file_duration=file_duration,
            min_duration=segment_duration_min,
            max_duration=segment_duration_max,
        )
        prediction = generate_random_segments(
            max_segments_per_label,
            labels,
            file_duration=file_duration,
            min_duration=segment_duration_min,
            max_duration=segment_duration_max,
        )
        segment_dir = os.path.join(result_dir, str(i))
        audeer.mkdir(segment_dir)
        truth.to_csv(os.path.join(segment_dir, "truth.csv"))
        prediction.to_csv(os.path.join(segment_dir, "prediction.csv"))

        truth_annotation = annotation_from_series(truth)
        pred_annotation = annotation_from_series(prediction)
        ier_metric = IdentificationErrorRate()
        ier = ier_metric(truth_annotation, pred_annotation)
        result = {
            "ier": ier,
        }
        result_metrics = pd.Series(result)
        result_metrics.to_csv(os.path.join(segment_dir, "result.csv"))


def main():
    random.seed(1)
    np.random.seed(1)
    current_dir = audeer.script_dir()
    der_dir = os.path.join(current_dir, "der")
    generate_der(der_dir)
    ier_dir = os.path.join(current_dir, "ier")
    generate_ier(ier_dir)


if __name__ == "__main__":
    main()
