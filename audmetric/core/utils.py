from __future__ import annotations

from collections.abc import Callable
from collections.abc import Hashable
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np


if TYPE_CHECKING:
    import pandas as pd

# Index fields for segmented index
FILE = "file"
START = "start"
END = "end"


def assert_equal_length(
    truth: Sequence[object],
    prediction: Sequence[object],
):
    r"""Assert truth and prediction have equal length."""
    if len(truth) != len(prediction):
        raise ValueError(
            f"Truth and prediction differ in length: {len(truth)} != {len(prediction)}."
        )


def infer_labels(
    truth: Sequence[object],
    prediction: Sequence[object],
) -> list[object]:
    r"""Infer labels from truth and prediction.

    It gathers all labels that are present
    in the truth and prediction values.

    Args:
        truth: ground truth labels
        prediction: predicted labels

    Returns:
        labels in sorted order

    """
    return sorted(list(set(truth) | set(prediction)))


def is_segmented_index(series: pd.Series):
    """Check if index of series is conform to audformat segmented index."""
    return (
        len(series.index.names) == 3
        and series.index.names[0] == FILE
        and series.index.names[1] == START
        and series.index.names[2] == END
    )


def scores_per_subgroup_and_class(
    truth: Sequence[object],
    prediction: Sequence[object],
    protected_variable: Sequence[object],
    metric: Callable[
        [
            Sequence[object],
            Sequence[object],
            Sequence[str] | None,
            float,
        ],
        dict[str, float],
    ],
    labels: Sequence[object],
    subgroups: Sequence[object],
    zero_division: float,
) -> dict[Hashable, dict]:
    r"""Compute scores per class for each subgroup based on metric.

    Args:
        truth: ground truth classes
        prediction: predicted classes
        protected_variable: manifestations of protected variable such as
            subgroups "male" and "female" of variable "sex"
        metric: metric to measure performance
        labels: included labels in preferred ordering
        subgroups: included subgroups in preferred ordering
        zero_division: set the value to return when there is a zero division

    Returns:
        scores per class for each subgroup

    Raises:
        ValueError: if `subgroups` contains values not contained in
            `protected_variable`

    Examples:
        >>> import audmetric
        >>> scores_per_subgroup_and_class(
        ...     [1, 1],
        ...     [0, 1],
        ...     ["male", "female"],
        ...     audmetric.recall_per_class,
        ...     [0, 1],
        ...     ["male", "female"],
        ...     0.0,
        ... )
        {'male': {0: 0.0, 1: 0.0}, 'female': {0: 0.0, 1: 1.0}}
        >>> scores_per_subgroup_and_class(
        ...     [1, 1],
        ...     [0, 1],
        ...     ["male", "female"],
        ...     audmetric.precision_per_class,
        ...     [0, 1],
        ...     ["male", "female"],
        ...     zero_division=np.nan,
        ... )
        {'male': {0: 0.0, 1: nan}, 'female': {0: nan, 1: 1.0}}

    """  # noqa: E501
    if set(subgroups) - set(protected_variable):
        raise ValueError(
            f"`subgroups` contains manifestations of the protected "
            f"variable which are not contained in `protected_variable`: "
            f"{set(subgroups) - set(protected_variable)}"
        )

    truth = np.array(truth)
    prediction = np.array(prediction)
    protected_variable = np.array(protected_variable)

    score = {}
    for subgroup in subgroups:
        mask = protected_variable == subgroup
        score[subgroup] = metric(
            truth[mask],
            prediction[mask],
            labels,
            zero_division=zero_division,
        )
    return score
