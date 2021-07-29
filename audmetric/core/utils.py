import typing

import numpy as np


def assert_equal_length(
        truth: typing.Sequence[typing.Any],
        prediction: typing.Sequence[typing.Any],
):
    r"""Assert truth and prediction have equal length."""
    if len(truth) != len(prediction):
        raise ValueError(
            f'Truth and prediction differ in length: '
            f'{len(truth)} != {len(prediction)}.'
        )


def infer_labels(
        truth: typing.Sequence[typing.Any],
        prediction: typing.Sequence[typing.Any],
) -> typing.List[typing.Any]:
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
