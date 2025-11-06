import collections
from collections.abc import Callable
from collections.abc import Sequence
import math
import warnings

import numpy as np
import pandas as pd
from scipy.sparse import csr_array
from scipy.sparse.csgraph import maximum_bipartite_matching

import audeer
import audformat
from audformat.define import IndexField

from audmetric.core.utils import assert_equal_length
from audmetric.core.utils import infer_labels
from audmetric.core.utils import scores_per_subgroup_and_class


def accuracy(
    truth: Sequence[object],
    prediction: Sequence[object],
    labels: Sequence[str | int] = None,
) -> float:
    r"""Classification accuracy.

    .. math::

        \text{accuracy} = \frac{\text{number of correct predictions}}
                               {\text{number of total predictions}}

    Args:
        truth: ground truth values/classes
        prediction: predicted values/classes
        labels: included labels in preferred ordering.
            Sample is considered in computation if either prediction or
            ground truth (logical OR) is contained in labels.
            If no labels are supplied,
            they will be inferred from
            :math:`\{\text{prediction}, \text{truth}\}`
            and ordered alphabetically.

    Returns:
        accuracy of prediction :math:`\in [0, 1]`

    Raises:
        ValueError: if ``truth`` and ``prediction`` differ in length

    Examples:
        >>> accuracy([0, 0], [0, 1])
        0.5

    """
    assert_equal_length(truth, prediction)
    if labels is None:
        labels = infer_labels(truth, prediction)

    prediction = np.array(prediction)
    truth = np.array(truth)

    # keep where both prediction and truth contained in `labels`
    label_mask = np.nonzero(
        np.logical_or(np.isin(truth, labels), np.isin(prediction, labels))
    )
    truth = truth[label_mask]
    prediction = prediction[label_mask]

    if len(prediction) == 0:
        return np.nan
    else:
        return float(sum(prediction == truth) / len(prediction))


def concordance_cc(
    truth: Sequence[float],
    prediction: Sequence[float],
    *,
    ignore_nan: bool = False,
) -> float:
    r"""Concordance correlation coefficient.

    .. math::

        \rho_c = \frac{2\rho\sigma_\text{prediction}\sigma_\text{truth}}
                      {\sigma_\text{prediction}^2 + \sigma_\text{truth}^2 + (
                      \mu_\text{prediction}-\mu_\text{truth})^2}

    where :math:`\rho` is the Pearson correlation coefficient,
    :math:`\mu` the mean
    and :math:`\sigma^2` the variance.\ :footcite:`Lin1989`

    .. footbibliography::

    Args:
        truth: ground truth values
        prediction: predicted values
        ignore_nan: if ``True``
            all samples that contain ``NaN``
            in ``truth`` or ``prediction``
            are ignored

    Returns:
        concordance correlation coefficient :math:`\in [-1, 1]`

    Raises:
        ValueError: if ``truth`` and ``prediction`` differ in length

    Examples:
        >>> concordance_cc([0, 1, 2], [0, 1, 1])
        0.6666666666666665

    """
    assert_equal_length(truth, prediction)

    if not isinstance(truth, np.ndarray):
        truth = np.array(list(truth))
    if not isinstance(prediction, np.ndarray):
        prediction = np.array(list(prediction))

    if ignore_nan:
        mask = ~(np.isnan(truth) | np.isnan(prediction))
        truth = truth[mask]
        prediction = prediction[mask]

    if len(prediction) < 2:
        return np.nan

    length = prediction.size
    mean_y = np.mean(truth)
    mean_x = np.mean(prediction)
    a = prediction - mean_x
    b = truth - mean_y

    numerator = 2 * np.dot(a, b)
    denominator = np.dot(a, a) + np.dot(b, b) + length * (mean_x - mean_y) ** 2

    if denominator == 0:
        ccc = np.nan
    else:
        ccc = numerator / denominator

    return float(ccc)


def confusion_matrix(
    truth: Sequence[object],
    prediction: Sequence[object],
    labels: Sequence[object] = None,
    *,
    normalize: bool = False,
) -> list[list[int | float]]:
    r"""Confusion matrix.

    Args:
        truth: ground truth values/classes
        prediction: predicted values/classes
        labels: included labels in preferred ordering.
            If no labels are supplied,
            they will be inferred from
            :math:`\{\text{prediction}, \text{truth}\}`
            and ordered alphabetically.
        normalize: normalize confusion matrix over the rows

    Returns:
        confusion matrix

    Raises:
        ValueError: if ``truth`` and ``prediction`` differ in length

    Examples:
        >>> truth = [0, 1, 2]
        >>> prediction = [0, 2, 0]
        >>> confusion_matrix(truth, prediction)
        [[1, 0, 0], [0, 0, 1], [1, 0, 0]]

    """
    assert_equal_length(truth, prediction)
    if labels is None:
        labels = infer_labels(truth, prediction)

    truth = np.array(truth)
    prediction = np.array(prediction)

    matrix = []
    for row in labels:
        row_indices = np.where(truth == row)
        y_row = prediction[row_indices]
        row_matrix = []
        for column in labels:
            row_matrix += [len(np.where(y_row == column)[0])]
        matrix += [row_matrix]

    if normalize:
        for idx, row in enumerate(matrix):
            if np.sum(row) != 0:
                row_sum = float(np.sum(row))
                matrix[idx] = [x / row_sum for x in row]

    return matrix


def detection_error_tradeoff(
    truth: Sequence[bool | int],
    prediction: Sequence[bool | int | float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Detection error tradeoff for verification experiments.

    The `detection error tradeoff (DET)`_
    is a graph showing
    the false non-match rate (FNMR)
    against the false match rate (FMR).
    The FNMR indicates
    how often an enrolled speaker was missed.
    The FMR indicates
    how often an impostor was verified as the enrolled speaker.

    This function does not return a figure,
    but the FMR and FNMR,
    together with the corresponding verification thresholds
    at which a similarity value
    was regarded to belong to the enrolled speaker.

    ``truth`` may only contain entries like ``[1, 0, True, False...]``,
    whereas prediction values
    can also contain similarity scores, e.g. ``[0.8, 0.1, ...]``.

    The implementation was inspired by pyeer.eer_stats.calculate_roc but has
    been accelerated by using numpy-arrays instead of lists.

    .. _detection error tradeoff (DET): https://en.wikipedia.org/wiki/Detection_error_tradeoff
    .. _pyeer: https://github.com/manuelaguadomtz/pyeer

    Args:
        truth: ground truth classes
        prediction: predicted classes or similarity scores

    Returns:
        * false match rate (FMR)
        * false non-match rate (FNMR)
        * verification thresholds

    Raises:
        ValueError: if ``truth`` contains values
            different from ``1, 0, True, False``

    Examples:
        >>> truth = [1, 0]
        >>> prediction = [0.9, 0.1]
        >>> detection_error_tradeoff(truth, prediction)
        (array([1., 0.]), array([0., 0.]), array([0.1, 0.9]))

    """  # noqa: E501
    # Get mated scores
    # (genuine matching scores)
    # and non-mated scores
    # (impostor matching scores)
    gscores, iscores = _matching_scores(truth, prediction)

    gscores_number = len(gscores)
    iscores_number = len(iscores)

    # Labeling genuine scores as 1 and impostor scores as 0
    gscores = np.column_stack((gscores, np.ones(gscores_number, dtype=int)))
    iscores = np.column_stack((iscores, np.zeros(iscores_number, dtype=int)))

    # Stacking scores
    all_scores = np.concatenate([gscores, iscores])
    sorted_indices = np.argsort(all_scores[:, 0])
    scores = all_scores[sorted_indices]
    cumul = np.cumsum(scores[:, 1])

    # Grouping scores
    thresholds, u_indices = np.unique(scores[:, 0], return_index=True)

    # Calculating FNM and FM distributions
    fnm = cumul[u_indices] - scores[u_indices][:, 1]  # rejecting s < t
    fm = iscores_number - (u_indices - fnm)

    # Calculating FMR and FNMR
    fnmr = fnm / gscores_number
    fmr = fm / iscores_number

    return fmr, fnmr, thresholds


def edit_distance(
    truth: str | Sequence[int],
    prediction: str | Sequence[int],
) -> int:
    r"""Edit distance between two sequences of chars or ints.

    The implementation follows the `Wagner-Fischer algorithm`_.

    .. _Wagner-Fischer algorithm:
        https://en.wikipedia.org/wiki/Wagner%E2%80%93Fischer_algorithm

    Args:
        truth: ground truth sequence
        prediction: predicted sequence

    Returns:
        edit distance

    Examples:
        >>> truth = "lorem"
        >>> prediction = "lorm"
        >>> edit_distance(truth, prediction)
        1
        >>> truth = [0, 1, 2]
        >>> prediction = [0, 1]
        >>> edit_distance(truth, prediction)
        1

    """
    if truth == prediction:
        return 0

    elif len(prediction) == 0:
        return len(truth)

    elif len(truth) == 0:
        return len(prediction)

    m0 = [None] * (len(truth) + 1)
    m1 = [None] * (len(truth) + 1)

    for i in range(len(m0)):
        m0[i] = i

    for i in range(len(prediction)):
        m1[0] = i + 1
        for j in range(len(truth)):
            cost = 0 if prediction[i] == truth[j] else 1
            m1[j + 1] = min(
                m1[j] + 1,  # deletion
                m0[j + 1] + 1,  # insertion
                m0[j] + cost,
            )  # substitution

        for j in range(len(m0)):
            m0[j] = m1[j]

    return m1[len(truth)]


def equal_error_rate(
    truth: Sequence[bool | int],
    prediction: Sequence[bool | int | float],
) -> tuple[float, collections.namedtuple]:
    r"""Equal error rate for verification tasks.

    The equal error rate (EER) is the point
    where false non-match rate (FNMR)
    and the impostors or false match rate (FMR)
    are identical.
    The FNMR indicates
    how often an enrolled speaker was missed.
    The FMR indicates
    how often an impostor was verified as the enrolled speaker.

    In practice the score distribution is not continuous
    and an interval is returned instead.
    The EER value will be set as the midpoint
    of this interval::footcite:`Maio2002`

    .. math::

        \text{EER} = \frac{
            \min(\text{FNMR}[t], \text{FMR}[t])
            + \max(\text{FNMR}[t], \text{FMR}[t])
        }{2}

    with :math:`t = \text{argmin}(|\text{FNMR} - \text{FMR}|)`.

    ``truth`` may only contain entries like ``[1, 0, True, False...]``,
    whereas prediction values
    can also contain similarity scores, e.g. ``[0.8, 0.1, ...]``.

    The implementation is identical with the one provided
    by the pyeer_ package.

    .. footbibliography::

    .. _pyeer: https://github.com/manuelaguadomtz/pyeer

    Args:
        truth: ground truth classes
        prediction: predicted classes or similarity scores

    Returns:
        * equal error rate (EER)
        * namedtuple containing
          ``fmr``,
          ``fnmr``,
          ``thresholds``,
          ``threshold``
          whereas the last one corresponds to the threshold
          corresponding to the returned EER

    Raises:
        ValueError: if ``truth`` contains values
            different from ``1, 0, True, False``

    Examples:
        >>> truth = [0, 1, 0, 1, 0]
        >>> prediction = [0.2, 0.8, 0.4, 0.5, 0.5]
        >>> eer, stats = equal_error_rate(truth, prediction)
        >>> eer
        0.16666666666666666
        >>> stats.threshold
        0.5

    """
    Stats = collections.namedtuple(
        "stats",
        [
            "fmr",  # False match rates (FMR)
            "fnmr",  # False non-match rates (FNMR)
            "thresholds",  # Thresholds
            "threshold",  # verification threshold for EER
        ],
    )
    fmr, fnmr, thresholds = detection_error_tradeoff(truth, prediction)
    diff = fmr - fnmr
    # t1 and t2 are our time indices
    t2 = np.where(diff <= 0)[0]
    if len(t2) > 0:
        t2 = t2[0]
    else:
        warnings.warn(
            "The false match rate "
            "and false non-match rate curves "
            "do not intersect each other.",
            RuntimeWarning,
        )
        eer = 1.0
        threshold = float(thresholds[0])
        return eer, Stats(fmr, fnmr, thresholds, threshold)

    t1 = t2 - 1 if diff[t2] != 0 and t2 != 0 else t2
    if fmr[t1] + fnmr[t1] <= fmr[t2] + fnmr[t2]:
        eer = (fnmr[t1] + fmr[t1]) / 2.0
        threshold = thresholds[t1]
    else:  # pragma: nocover (couldn't find a test to trigger this)
        eer = (fnmr[t2] + fmr[t2]) / 2.0
        threshold = thresholds[t2]
    eer = float(eer)
    threshold = float(threshold)
    return eer, Stats(fmr, fnmr, thresholds, threshold)


def event_confusion_matrix(
    truth: pd.Series,
    prediction: pd.Series,
    labels: Sequence[object] | None = None,
    *,
    onset_tolerance: float | None = 0.0,
    offset_tolerance: float | None = 0.0,
    duration_tolerance: float | None = None,
    normalize: bool = False,
) -> list[list[int | float]]:
    r"""Event-based confusion.

    This metric compares not only the labels of prediction and ground truth,
    but also the time windows they occur in.

    Each event is considered to be correctly identified
    if the predicted label is the same as the ground truth label,
    and if the onset is within the given ``onset_tolerance`` (in seconds)
    and the offset is within the given ``offset_tolerance`` (in seconds).
    Additionally to the ``offset_tolerance``,
    one can also specify the ``duration_tolerance``,
    to ensure that the offset occurs
    within a certain proportion of the reference event duration.
    If a prediction fulfills the ``duration_tolerance``
    but not the ``offset_tolerance`` (or vice versa),
    it is still considered to be an overlapping segment.
    :footcite:`Mesaros2016`

    The resulting confusion matrix has one more row and and one more column
    than there are labels.
    The last row/column corresponds to the absence of any event.
    This allows to distinguish between segments that overlap but have differing labels,
    and false negatives that have no overlapping predicted segment
    as well as false positives that have no overlapping ground truth segment.

    .. footbibliography::

    Args:
        truth: ground truth labels with a segmented index conform to `audformat`_
        prediction: predicted labels with a segmented index conform to `audformat`_
        labels: included labels in preferred ordering.
            If no labels are supplied,
            they will be inferred from
            :math:`\{\text{prediction}, \text{truth}\}`
            and ordered alphabetically
        onset_tolerance: the onset tolerance in seconds.
            If the predicted segment's onset does not occur within this time window
            compared to the ground truth segment's onset,
            it is not considered correct
        offset_tolerance: the offset tolerance in seconds.
            If the predicted segment's offset does not occur within this time window
            compared to the ground truth segment's offset,
            it is not considered correct,
            unless the ``duration_tolerance`` is specified and fulfilled
        duration_tolerance: the duration tolerance as a measure of proportion
            of the ground truth segment's total duration.
            If the ``offset_tolerance`` is not fulfilled,
            and the predicted segment's offset does not occur within this time window
            compared to the ground truth segment's offset,
            it is not considered correct
        normalize: normalize confusion matrix over the rows

    Returns:
        event confusion matrix

    Raises:
        ValueError: if ``truth`` or ``prediction``
            do not have a segmented index conform to `audformat`_

    Examples:
        >>> truth = pd.Series(
        ...     index=audformat.segmented_index(
        ...         files=["f1.wav"] * 4,
        ...         starts=[0, 0.1, 0.2, 0.3],
        ...         ends=[0.1, 0.2, 0.3, 0.4],
        ...     ),
        ...     data=["a", "a", "b", "b"],
        ... )
        >>> prediction = pd.Series(
        ...     index=audformat.segmented_index(
        ...         files=["f1.wav"] * 4 + ["f2.wav"],
        ...         starts=[0, 0.09, 0.2, 0.31, 0.0],
        ...         ends=[0.1, 0.2, 0.3, 0.41, 1.0],
        ...     ),
        ...     data=["a", "b", "a", "b", "b"],
        ... )
        >>> event_confusion_matrix(
        ...     truth, prediction, onset_tolerance=0.02, offset_tolerance=0.02
        ... )
        [[1, 1, 0], [1, 1, 0], [0, 1, 0]]

    .. _audformat: https://audeering.github.io/audformat/data-format.html

    """
    if not audformat.is_segmented_index(truth) or not audformat.is_segmented_index(
        prediction
    ):
        raise ValueError(
            "For event-based metrics, the truth and prediction "
            "should be a pandas Series with a segmented index conform to audformat."
        )
    if labels is None:
        labels = infer_labels(truth.values, prediction.values)

    # Confusion matrix of event labels + "no event" label
    cm = [[0 for _ in range(len(labels) + 1)] for _ in range(len(labels) + 1)]

    # Code based on 'optimal' event matching
    # at https://github.com/TUT-ARG/sed_eval/blob/0cb1b6d11ceec4fe500cc9b31079c9d8666ed6eb/sed_eval/sound_event.py#L1108
    for file, file_truth in truth.groupby(level=IndexField.FILE):
        file_pred = prediction[
            prediction.index.get_level_values(IndexField.FILE) == file
        ]
        n_truth = len(file_truth)
        n_pred = len(file_pred)
        # Matrix storing whether there is an overlap
        # between each truth segment and each predicted segment
        overlap_matrix = np.ones((n_truth, n_pred), dtype=bool)
        # Matrix storing whether there is an overlap
        # AND the labels are the same
        hit_matrix = np.zeros((n_truth, n_pred), dtype=bool)
        for i, ((_, start_truth, end_truth), label_truth) in enumerate(
            # file_truth.sort_index().items()
            file_truth.items()
        ):
            start_truth = start_truth.total_seconds()
            end_truth = end_truth.total_seconds()
            for j, ((_, start_pred, end_pred), label_pred) in enumerate(
                # file_pred.sort_index().items()
                file_pred.items()
            ):
                start_pred = start_pred.total_seconds()
                end_pred = end_pred.total_seconds()
                # Condition 1: labels are the same
                label_match = label_truth == label_pred
                # Condition 2: segments overlap in onset/offset
                overlap_match = _segments_overlap(
                    start_truth,
                    end_truth,
                    start_pred,
                    end_pred,
                    onset_tolerance,
                    offset_tolerance,
                    duration_tolerance,
                )
                hit_matrix[i, j] = label_match & overlap_match
                overlap_matrix[i, j] = overlap_match
        _match_segments_and_accumulate(
            cm=cm,
            hit_matrix=hit_matrix,
            overlap_matrix=overlap_matrix,
            truth_labels=file_truth.to_list(),
            pred_labels=file_pred.to_list(),
            label_to_index={label: i for i, label in enumerate(labels)},
        )

    # Fill in remaining errors that have no confusions
    for i, label in enumerate(labels):
        n_label_truth = len(truth[truth == label])
        # Count any ground truth segments that have no overlapping prediction at all
        n_missed = n_label_truth - sum(cm[i][: len(labels)])
        cm[i][-1] += n_missed
        n_label_pred = len(prediction[prediction == label])
        # Count any predictions that have no overlap with ground truth segments
        n_extra = n_label_pred - sum([cm[j][i] for j in range(len(labels))])
        cm[-1][i] = n_extra

    if normalize:
        for idx, row in enumerate(cm):
            if np.sum(row) != 0:
                row_sum = float(np.sum(row))
                cm[idx] = [x / row_sum for x in row]
    return cm


def event_error_rate(
    truth: str | Sequence[str | Sequence[int]],
    prediction: (str | Sequence[str | Sequence[int]]),
) -> float:
    r"""Event error rate based on edit distance.

    The event error rate is computed by aggregating the mean edit
    distances of each (truth, prediction)-pair and averaging the
    aggregated score by the number of pairs.

    The mean edit distance of each (truth, prediction)-pair is computed
    as an average of the edit distance over the length of the longer sequence
    of the corresponding pair. By normalizing over the longer sequence the
    normalized distance is bound to [0, 1].

    Args:
        truth: ground truth classes
        prediction: predicted classes

    Returns:
        event error rate

    Raises:
        ValueError: if ``truth`` and ``prediction`` differ in length

    Examples:
        >>> event_error_rate([[0, 1]], [[0]])
        0.5
        >>> event_error_rate([[0, 1], [2]], [[0], [2]])
        0.25
        >>> event_error_rate(["lorem"], ["lorm"])
        0.2
        >>> event_error_rate(["lorem", "ipsum"], ["lorm", "ipsum"])
        0.1

    """
    truth = audeer.to_list(truth)
    prediction = audeer.to_list(prediction)

    assert_equal_length(truth, prediction)

    eer = 0.0

    for t, p in zip(truth, prediction):
        n = max(len(t), len(p))
        n = n if n > 1 else 1
        eer += edit_distance(t, p) / n

    num_samples = len(truth) if len(truth) > 1 else 1
    return eer / num_samples


def event_fscore_per_class(
    truth: pd.Series,
    prediction: pd.Series,
    labels: Sequence[object] | None = None,
    *,
    zero_division: float = 0.0,
    propagate_nans: bool = False,
    onset_tolerance: float | None = 0.0,
    offset_tolerance: float | None = 0.0,
    duration_tolerance: float | None = None,
) -> dict[str, float]:
    r"""Event-based F-score per class.

    .. math::

        \text{fscore}_k = \frac{\text{true positive}_k}
                 {\text{true positive}_k + \frac{1}{2}
                 (\text{false positive}_k + \text{false negative}_k)}

    This metric compares not only the labels of prediction and ground truth,
    but also the time windows they occur in.

    Each event is considered to be correctly identified
    if the predicted label is the same as the ground truth label,
    and if the onset is within the given ``onset_tolerance`` (in seconds)
    and the offset is within the given ``offset_tolerance`` (in seconds).
    Additionally to the ``offset_tolerance``,
    one can also specify the ``duration_tolerance``,
    to ensure that the offset occurs
    within a certain proportion of the reference event duration.
    If a prediction fulfills the ``duration_tolerance``
    but not the ``offset_tolerance`` (or vice versa),
    it is still considered to be an overlapping segment.
    :footcite:`Mesaros2016`

    .. footbibliography::

    Args:
        truth: ground truth values/classes
        prediction: predicted values/classes
        labels: included labels in preferred ordering.
            If no labels are supplied,
            they will be inferred from
            :math:`\{\text{prediction}, \text{truth}\}`
            and ordered alphabetically.
        zero_division: set the value to return when there is a zero division
        propagate_nans: whether to set the F-score to ``NaN``
            when recall or precision are ``NaN``.
            If ``False``, the F-score is only set to ``NaN``
            when both recall and precision are ``NaN``
        onset_tolerance: the onset tolerance in seconds.
            If the predicted segment's onset does not occur within this time window
            compared to the ground truth segment's onset,
            it is not considered correct
        offset_tolerance: the offset tolerance in seconds.
            If the predicted segment's offset does not occur within this time window
            compared to the ground truth segment's offset,
            it is not considered correct,
            unless the ``duration_tolerance`` is specified and fulfilled
        duration_tolerance: the duration tolerance as a measure of proportion
            of the ground truth segment's total duration.
            If the ``offset_tolerance`` is not fulfilled,
            and the predicted segment's offset does not occur within this time window
            compared to the ground truth segment's offset,
            it is not considered correct

    Returns:
        dictionary with label as key and F-score as value

    Raises:
        ValueError: if ``truth`` or ``prediction``
            do not have a segmented index conform to `audformat`_

    Examples:
        >>> truth = pd.Series(
        ...     index=audformat.segmented_index(
        ...         files=["f1.wav", "f1.wav"],
        ...         starts=[0.0, 0.1],
        ...         ends=[0.1, 0.2],
        ...     ),
        ...     data=["a", "b"],
        ... )
        >>> prediction = pd.Series(
        ...     index=audformat.segmented_index(
        ...         files=["f1.wav", "f1.wav"],
        ...         starts=[0, 0.09],
        ...         ends=[0.1, 0.2],
        ...     ),
        ...     data=["a", "a"],
        ... )
        >>> event_fscore_per_class(
        ...     truth, prediction, onset_tolerance=0.02, offset_tolerance=0.02
        ... )
        {'a': 0.6666666666666666, 'b': 0.0}

    .. _audformat: https://audeering.github.io/audformat/data-format.html

    """
    if labels is None:
        labels = infer_labels(truth, prediction)
    precision = event_precision_per_class(
        truth,
        prediction,
        labels,
        zero_division=zero_division,
        onset_tolerance=onset_tolerance,
        offset_tolerance=offset_tolerance,
        duration_tolerance=duration_tolerance,
    )
    recall = event_recall_per_class(
        truth,
        prediction,
        labels,
        zero_division=zero_division,
        onset_tolerance=onset_tolerance,
        offset_tolerance=offset_tolerance,
        duration_tolerance=duration_tolerance,
    )
    fscore = {}
    for label, p, r in zip(labels, precision.values(), recall.values()):
        if np.isnan(p) or np.isnan(r):
            if propagate_nans:
                fscore[label] = np.nan
            else:
                fscore[label] = 0.0
        elif p * r == 0:
            fscore[label] = 0.0
        else:
            fscore[label] = (2 * p * r) / (p + r)
    return fscore


def event_precision_per_class(
    truth: pd.Series,
    prediction: pd.Series,
    labels: Sequence[object] | None = None,
    *,
    zero_division: float = 0.0,
    onset_tolerance: float | None = 0.0,
    offset_tolerance: float | None = 0.0,
    duration_tolerance: float | None = None,
) -> dict[str, float]:
    r"""Event-based precision per class.

    .. math::

        \text{precision}_k = \frac{\text{true positive}_k}
                 {\text{true positive}_k + \text{false positive}_k}

    This metric compares not only the labels of prediction and ground truth,
    but also the time windows they occur in.

    Each event is considered to be correctly identified
    if the predicted label is the same as the ground truth label,
    and if the onset is within the given ``onset_tolerance`` (in seconds)
    and the offset is within the given ``offset_tolerance`` (in seconds).
    Additionally to the ``offset_tolerance``,
    one can also specify the ``duration_tolerance``,
    to ensure that the offset occurs
    within a certain proportion of the reference event duration.
    If a prediction fulfills the ``duration_tolerance``
    but not the ``offset_tolerance`` (or vice versa),
    it is still considered to be an overlapping segment.
    :footcite:`Mesaros2016`

    .. footbibliography::

    Args:
        truth: ground truth values/classes
        prediction: predicted values/classes
        labels: included labels in preferred ordering.
            If no labels are supplied,
            they will be inferred from
            :math:`\{\text{prediction}, \text{truth}\}`
            and ordered alphabetically.
        zero_division: set the value to return when there is a zero division
        onset_tolerance: the onset tolerance in seconds.
            If the predicted segment's onset does not occur within this time window
            compared to the ground truth segment's onset,
            it is not considered correct
        offset_tolerance: the offset tolerance in seconds.
            If the predicted segment's offset does not occur within this time window
            compared to the ground truth segment's offset,
            it is not considered correct,
            unless the ``duration_tolerance`` is specified and fulfilled
        duration_tolerance: the duration tolerance as a measure of proportion
            of the ground truth segment's total duration.
            If the ``offset_tolerance`` is not fulfilled,
            and the predicted segment's offset does not occur within this time window
            compared to the ground truth segment's offset,
            it is not considered correct

    Returns:
        dictionary with label as key and precision as value

    Raises:
        ValueError: if ``truth`` or ``prediction``
            do not have a segmented index conform to `audformat`_

    Examples:
        >>> truth = pd.Series(
        ...     index=audformat.segmented_index(
        ...         files=["f1.wav", "f1.wav"],
        ...         starts=[0.0, 0.1],
        ...         ends=[0.1, 0.2],
        ...     ),
        ...     data=["a", "b"],
        ... )
        >>> prediction = pd.Series(
        ...     index=audformat.segmented_index(
        ...         files=["f1.wav", "f1.wav"],
        ...         starts=[0.0, 0.09],
        ...         ends=[0.11, 0.2],
        ...     ),
        ...     data=["a", "a"],
        ... )
        >>> event_precision_per_class(
        ...     truth, prediction, onset_tolerance=0.02, offset_tolerance=0.02
        ... )
        {'a': 0.5, 'b': 0.0}

    .. _audformat: https://audeering.github.io/audformat/data-format.html

    """
    return _event_metric_per_class(
        truth,
        prediction,
        labels,
        zero_division,
        onset_tolerance,
        offset_tolerance,
        duration_tolerance,
        axis=0,
    )


def event_recall_per_class(
    truth: pd.Series,
    prediction: pd.Series,
    labels: Sequence[object] | None = None,
    *,
    zero_division: float = 0.0,
    onset_tolerance: float | None = 0.0,
    offset_tolerance: float | None = 0.0,
    duration_tolerance: float | None = None,
) -> dict[str, float]:
    r"""Event-based recall per class.

    .. math::

        \text{recall}_k = \frac{\text{true positive}_k}
                 {\text{true positive}_k + \text{false negative}_k}

    This metric compares not only the labels of prediction and ground truth,
    but also the time windows they occur in.

    Each event is considered to be correctly identified
    if the predicted label is the same as the ground truth label,
    and if the onset is within the given ``onset_tolerance`` (in seconds)
    and the offset is within the given ``offset_tolerance`` (in seconds).
    Additionally to the ``offset_tolerance``,
    one can also specify the ``duration_tolerance``,
    to ensure that the offset occurs
    within a certain proportion of the reference event duration.
    If a prediction fulfills the ``duration_tolerance``
    but not the ``offset_tolerance`` (or vice versa),
    it is still considered to be an overlapping segment.
    :footcite:`Mesaros2016`

    .. footbibliography::

    Args:
        truth: ground truth values/classes
        prediction: predicted values/classes
        labels: included labels in preferred ordering.
            If no labels are supplied,
            they will be inferred from
            :math:`\{\text{prediction}, \text{truth}\}`
            and ordered alphabetically.
        zero_division: set the value to return when there is a zero division
        onset_tolerance: the onset tolerance in seconds.
            If the predicted segment's onset does not occur within this time window
            compared to the ground truth segment's onset,
            it is not considered correct
        offset_tolerance: the offset tolerance in seconds.
            If the predicted segment's offset does not occur within this time window
            compared to the ground truth segment's offset,
            it is not considered correct,
            unless the ``duration_tolerance`` is specified and fulfilled
        duration_tolerance: the duration tolerance as a measure of proportion
            of the ground truth segment's total duration.
            If the ``offset_tolerance`` is not fulfilled,
            and the predicted segment's offset does not occur within this time window
            compared to the ground truth segment's offset,
            it is not considered correct

    Returns:
        dictionary with label as key and recall as value

    Raises:
        ValueError: if ``truth`` or ``prediction``
            do not have a segmented index conform to `audformat`_

    Examples:
        >>> truth = pd.Series(
        ...     index=audformat.segmented_index(
        ...         files=["f1.wav", "f1.wav"],
        ...         starts=[0.0, 0.1],
        ...         ends=[0.1, 0.2],
        ...     ),
        ...     data=["a", "b"],
        ... )
        >>> prediction = pd.Series(
        ...     index=audformat.segmented_index(
        ...         files=["f1.wav", "f1.wav"],
        ...         starts=[0.0, 0.09],
        ...         ends=[0.11, 0.2],
        ...     ),
        ...     data=["a", "a"],
        ... )
        >>> event_recall_per_class(
        ...     truth, prediction, onset_tolerance=0.02, offset_tolerance=0.02
        ... )
        {'a': 1.0, 'b': 0.0}

    .. _audformat: https://audeering.github.io/audformat/data-format.html

    """
    return _event_metric_per_class(
        truth,
        prediction,
        labels,
        zero_division,
        onset_tolerance,
        offset_tolerance,
        duration_tolerance,
        axis=1,
    )


def event_unweighted_average_fscore(
    truth: pd.Series,
    prediction: pd.Series,
    labels: Sequence[object] | None = None,
    *,
    zero_division: float = 0.0,
    propagate_nans: bool = False,
    onset_tolerance: float | None = 0.0,
    offset_tolerance: float | None = 0.0,
    duration_tolerance: float | None = None,
) -> float:
    r"""Event-based unweighted average F-score.

    .. math::

        \text{UAF} = \frac{1}{K} \sum^K_{k=1}
            \frac{\text{true positive}_k}
                 {\text{true positive}_k + \frac{1}{2}
                 (\text{false positive}_k + \text{false negative}_k)}

    This metric compares not only the labels of prediction and ground truth,
    but also the time windows they occur in.

    Each event is considered to be correctly identified
    if the predicted label is the same as the ground truth label,
    and if the onset is within the given ``onset_tolerance`` (in seconds)
    and the offset is within the given ``offset_tolerance`` (in seconds).
    Additionally to the ``offset_tolerance``,
    one can also specify the ``duration_tolerance``,
    to ensure that the offset occurs
    within a certain proportion of the reference event duration.
    If a prediction fulfills the ``duration_tolerance``
    but not the ``offset_tolerance`` (or vice versa),
    it is still considered to be an overlapping segment.
    :footcite:`Mesaros2016`

    .. footbibliography::

    Args:
        truth: ground truth values/classes
        prediction: predicted values/classes
        labels: included labels in preferred ordering.
            If no labels are supplied,
            they will be inferred from
            :math:`\{\text{prediction}, \text{truth}\}`
            and ordered alphabetically.
        zero_division: set the value to return when there is a zero division
        propagate_nans: whether to set the F-score to ``NaN``
            when recall or precision are ``NaN``.
            If ``False``, the F-score is only set to ``NaN``
            when both recall and precision are ``NaN``
        onset_tolerance: the onset tolerance in seconds.
            If the predicted segment's onset does not occur within this time window
            compared to the ground truth segment's onset,
            it is not considered correct
        offset_tolerance: the offset tolerance in seconds.
            If the predicted segment's offset does not occur within this time window
            compared to the ground truth segment's offset,
            it is not considered correct,
            unless the ``duration_tolerance`` is specified and fulfilled
        duration_tolerance: the duration tolerance as a measure of proportion
            of the ground truth segment's total duration.
            If the ``offset_tolerance`` is not fulfilled,
            and the predicted segment's offset does not occur within this time window
            compared to the ground truth segment's offset,
            it is not considered correct

    Returns:
        event-based unweighted average f-score

    Examples:
        >>> truth = pd.Series(
        ...     index=audformat.segmented_index(
        ...         files=["f1.wav", "f1.wav"],
        ...         starts=[0.0, 0.1],
        ...         ends=[0.1, 0.2],
        ...     ),
        ...     data=["a", "b"],
        ... )
        >>> prediction = pd.Series(
        ...     index=audformat.segmented_index(
        ...         files=["f1.wav", "f1.wav"],
        ...         starts=[0, 0.09],
        ...         ends=[0.1, 0.2],
        ...     ),
        ...     data=["a", "a"],
        ... )
        >>> event_unweighted_average_fscore(
        ...     truth, prediction, onset_tolerance=0.02, offset_tolerance=0.02
        ... )
        0.3333333333333333

    .. _audformat: https://audeering.github.io/audformat/data-format.html

    """
    fscore = event_fscore_per_class(
        truth,
        prediction,
        labels,
        zero_division=zero_division,
        propagate_nans=propagate_nans,
        onset_tolerance=onset_tolerance,
        offset_tolerance=offset_tolerance,
        duration_tolerance=duration_tolerance,
    )
    fscore = np.array(list(fscore.values()))
    return float(np.nanmean(fscore))


def fscore_per_class(
    truth: Sequence[object],
    prediction: Sequence[object],
    labels: Sequence[object] = None,
    *,
    zero_division: float = 0,
) -> dict[str, float]:
    r"""F-score per class.

    .. math::

        \text{fscore}_k = \frac{\text{true positive}_k}
                 {\text{true positive}_k + \frac{1}{2}
                 (\text{false positive}_k + \text{false negative}_k)}

    Args:
        truth: ground truth values/classes
        prediction: predicted values/classes
        labels: included labels in preferred ordering.
            If no labels are supplied,
            they will be inferred from
            :math:`\{\text{prediction}, \text{truth}\}`
            and ordered alphabetically.
        zero_division: set the value to return when there is a zero division

    Returns:
        dictionary with label as key and F-score as value

    Examples:
        >>> fscore_per_class([0, 0], [0, 1])
        {0: 0.6666666666666666, 1: 0.0}

    """
    if labels is None:
        labels = infer_labels(truth, prediction)

    precision = precision_per_class(
        truth,
        prediction,
        labels,
        zero_division=zero_division,
    )
    recall = recall_per_class(
        truth,
        prediction,
        labels,
        zero_division=zero_division,
    )
    fscore = {}
    for label, p, r in zip(labels, precision.values(), recall.values()):
        if p * r == 0:
            fscore[label] = 0.0
        elif (p == 0.0 and np.isnan(r)) or (r == 0.0 and np.isnan(p)):
            fscore[label] = 0.0
        else:
            fscore[label] = (2 * p * r) / (p + r)
    return fscore


def linkability(
    truth: (bool | int | Sequence[bool | int]),
    prediction: (bool | int | float | Sequence[bool | int | float]),
    omega: float = 1.0,
    nbins: int = None,
) -> float:
    r"""Linkability for verification tasks.

    Let :math:`s` be the provided prediction score
    for the similarity of the tested sample.
    The clipped local linkability metric is then defined as:

    .. math::

        \text{max}(0, p(\text{mated} | s) - p(\text{non-mated} | s))

    The higher the value,
    the more likely
    that an attacker can link two mated samples.
    The global linkability metric :math:`D_\text{sys}`
    is the mean value
    over all local scores,\ :footcite:`GomezBarrero2017`
    and in the range :math:`0` and :math:`1`.

    Implementation is based on
    `code from M. Maouche`_,
    which is licensed under LGPL.

    .. footbibliography::

    .. _code from M. Maouche: https://gitlab.inria.fr/magnet/anonymization_metrics

    Args:
        truth: ground truth classes
        prediction: predicted classes or similarity scores
        omega: prior ratio
            :math:`\frac{p(\text{mated})}{p(\text{non-mated})}`
        nbins: number of bins
            of the histograms
            that estimate the distributions
            of mated and non-mated scores.
            If ``None`` it is set to
            :math:`\min(\frac{\text{len}(\text{mated})}{10}, 100)`

    Returns:
        global linkability :math:`D_\text{sys}`

    Raises:
        ValueError: if ``truth`` contains values
            different from ``1``, ``0``, ``True``, ``False``

    Examples:
        >>> np.random.seed(1)
        >>> samples = 10000
        >>> truth = [1, 0] * int(samples / 2)
        >>> prediction = []
        >>> for _ in range(int(samples / 2)):
        ...     prediction.extend(
        ...         [np.random.uniform(0, 0.2), np.random.uniform(0.8, 1.0)]
        ...     )
        >>> linkability(truth, prediction)
        0.9747999999999999
        >>> truth = [1, 0, 0, 0] * int(samples / 4)
        >>> prediction = [np.random.uniform(0, 1) for _ in range(samples)]
        >>> linkability(truth, prediction, omega=1 / 3)
        0.0

    """  # noqa: E501
    mated_scores, non_mated_scores = _matching_scores(truth, prediction)

    # Limiting the number of bins
    # (100 maximum or lower if few scores available)
    if nbins is None:
        nbins = min(int(len(mated_scores) / 10), 100)

    # Define range of scores to compute D
    bin_edges = np.linspace(
        min([min(mated_scores), min(non_mated_scores)]),
        max([max(mated_scores), max(non_mated_scores)]),
        num=nbins + 1,
        endpoint=True,
    )
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2

    # Compute score distributions using normalized histograms
    y1 = np.histogram(mated_scores, bins=bin_edges, density=True)[0]
    y2 = np.histogram(non_mated_scores, bins=bin_edges, density=True)[0]
    # LR = P[s|mated ]/P[s|non-mated]
    lr = np.divide(y1, y2, out=np.ones_like(y1), where=y2 != 0)
    d = 2 * (omega * lr / (1 + omega * lr)) - 1
    # Def of D
    d[omega * lr <= 1] = 0
    # Taking care of inf/NaN
    mask = [True if y2[i] == 0 and y1[i] != 0 else False for i in range(len(y1))]
    d[mask] = 1
    # Global measure using trapz numerical integration
    d_sys = np.trapezoid(x=bin_centers, y=d * y1)

    return float(d_sys)


def mean_absolute_error(
    truth: Sequence[float],
    prediction: Sequence[float],
) -> float:
    r"""Mean absolute error.

    .. math::

        \text{MAE} = \frac{1}{n} \sum^n_{i=1}
            |\text{prediction} - \text{truth}|

    Args:
        truth: ground truth values
        prediction: predicted values

    Returns:
        mean absolute error

    Raises:
        ValueError: if ``truth`` and ``prediction`` differ in length

    Examples:
        >>> mean_absolute_error([0, 0], [0, 1])
        0.5

    """
    assert_equal_length(truth, prediction)

    prediction = np.array(prediction)
    truth = np.array(truth)

    return float(np.abs(truth - prediction).mean(axis=0))


def mean_squared_error(
    truth: Sequence[float],
    prediction: Sequence[float],
) -> float:
    r"""Mean squared error.

    .. math::

        \text{MSE} = \frac{1}{n} \sum^n_{i=1}
            (\text{prediction} - \text{truth})^2

    Args:
        truth: ground truth values
        prediction: predicted values

    Returns:
        mean squared error

    Raises:
        ValueError: if ``truth`` and ``prediction`` differ in length

    Examples:
        >>> mean_squared_error([0, 0], [0, 1])
        0.5

    """
    assert_equal_length(truth, prediction)

    prediction = np.array(prediction)
    truth = np.array(truth)

    return float(np.square(truth - prediction).mean(axis=0))


def pearson_cc(
    truth: Sequence[float],
    prediction: Sequence[float],
) -> float:
    r"""Pearson correlation coefficient.

    .. math::

        \rho = \frac{\text{cov}(\text{prediction}, \text{truth})}{
        \sigma_\text{prediction}\sigma_\text{truth}}

    where :math:`\sigma` is the standard deviation,
    and :math:`\text{cov}` is the covariance.

    Args:
        truth: ground truth values
        prediction: predicted values

    Returns:
        pearson correlation coefficient :math:`\in [-1, 1]`

    Raises:
        ValueError: if ``truth`` and ``prediction`` differ in length

    Examples:
        >>> pearson_cc([0, 1, 2], [0, 1, 1])
        0.8660254037844385

    """
    assert_equal_length(truth, prediction)

    if not isinstance(truth, np.ndarray):
        truth = np.array(list(truth))
    if not isinstance(prediction, np.ndarray):
        prediction = np.array(list(prediction))

    if len(prediction) < 2 or prediction.std() == 0:
        return np.nan
    else:
        return float(np.corrcoef(prediction, truth)[0][1])


def precision_per_class(
    truth: Sequence[object],
    prediction: Sequence[object],
    labels: Sequence[object] = None,
    *,
    zero_division: float = 0,
) -> dict[str, float]:
    r"""Precision per class.

    .. math::

        \text{precision}_k = \frac{\text{true positive}_k}
                 {\text{true positive}_k + \text{false positive}_k}

    Args:
        truth: ground truth values/classes
        prediction: predicted values/classes
        labels: included labels in preferred ordering.
            If no labels are supplied,
            they will be inferred from
            :math:`\{\text{prediction}, \text{truth}\}`
            and ordered alphabetically.
        zero_division: set the value to return when there is a zero division

    Returns:
        dictionary with label as key and precision as value

    Examples:
        >>> precision_per_class([0, 0], [0, 1])
        {0: 1.0, 1: 0.0}

    """
    if labels is None:
        labels = infer_labels(truth, prediction)

    matrix = np.array(confusion_matrix(truth, prediction, labels))
    total = matrix.sum(axis=0)
    old_settings = np.seterr(invalid="ignore")
    precision = matrix.diagonal() / total
    np.seterr(**old_settings)
    precision[np.isnan(precision)] = zero_division

    return {label: float(r) for label, r in zip(labels, precision)}


def recall_per_class(
    truth: Sequence[object],
    prediction: Sequence[object],
    labels: Sequence[object] = None,
    *,
    zero_division: float = 0,
) -> dict[str, float]:
    r"""Recall per class.

    .. math::

        \text{recall}_k = \frac{\text{true positive}_k}
                 {\text{true positive}_k + \text{false negative}_k}

    Args:
        truth: ground truth values/classes
        prediction: predicted values/classes
        labels: included labels in preferred ordering.
            If no labels are supplied,
            they will be inferred from
            :math:`\{\text{prediction}, \text{truth}\}`
            and ordered alphabetically.
        zero_division: set the value to return when there is a zero division

    Returns:
        dictionary with label as key and recall as value

    Examples:
        >>> recall_per_class([0, 0], [0, 1])
        {0: 0.5, 1: 0.0}

    """
    if labels is None:
        labels = infer_labels(truth, prediction)

    matrix = np.array(confusion_matrix(truth, prediction, labels))
    total = matrix.sum(axis=1)
    old_settings = np.seterr(invalid="ignore")
    recall = matrix.diagonal() / total
    np.seterr(**old_settings)
    recall[np.isnan(recall)] = zero_division

    return {label: float(r) for label, r in zip(labels, recall)}


def unweighted_average_bias(
    truth: Sequence[object],
    prediction: Sequence[object],
    protected_variable: Sequence[object],
    labels: Sequence[object] = None,
    *,
    subgroups: Sequence[object] = None,
    metric: Callable[
        [
            Sequence[object],
            Sequence[object],
            Sequence[str] | None,
        ],
        dict[str, float],
    ] = fscore_per_class,
    reduction: Callable[
        [
            Sequence[float],
        ],
        float,
    ] = np.std,
) -> float:
    r"""Unweighted average bias of protected variable.

    The bias is measured in terms of *equalized odds* which requires
    the classifier to have identical performance for all classes independent
    of a protected variable such as race. The performance of the classifier
    for its different classes can be assessed with standard metrics
    such as *recall* or *precision*. The difference in performance, denoted
    as score divergence, can be computed in different ways, as well.
    For two subgroups the (absolute) difference serves as a standard choice.
    For more than two subgroups the score divergence could be estimated by
    the standard deviation of the scores.

    Note:
        If for a class less than two subgroups exhibit a performance score,
        the corresponding class is ignored in the bias computation.
        This occurs if there is no class sample for a subgroup,
        e.g. no negative (class label) female (subgroup of sex).

    Args:
        truth: ground truth classes
        prediction: predicted classes
        protected_variable: manifestations of protected variable such as
            subgroups "male" and "female" of variable "sex"
        labels: included labels in preferred ordering.
            The bias is computed only on the specified labels.
            If no labels are supplied,
            they will be inferred from
            :math:`\{\text{prediction}, \text{truth}\}`
            and ordered alphabetically.
        subgroups: included subgroups in preferred ordering.
            The direction of the bias is determined by the ordering of the
            subgroups.
            Besides, the bias is computed only on the specified subgroups.
            If no subgroups are supplied, they will be inferred from
            :math:`\text{protected\_variable}` and ordered alphanumerically.
        metric: metric which equalized odds are measured with.
            Typical choices are: :func:`audmetric.recall_per_class`,
            :func:`audmetric.precision_per_class` or
            :func:`audmetric.fscore_per_class`
        reduction: specifies the reduction operation to measure the divergence
            between the scores of the subgroups of the protected variable
            for each class. Typical choices are:
            difference or absolute difference between scores for two subgroups
            and standard deviation of scores for more than two subgroups.

    Returns:
        unweighted average bias

    Raises:
        ValueError: if ``truth``, ``prediction`` and ``protected_variable``
            have different lengths
        ValueError: if ``subgroups`` contains values not contained in
            ``protected_variable``

    Examples:
        >>> unweighted_average_bias([1, 1], [1, 0], ["male", "female"])
        0.5
        >>> unweighted_average_bias(
        ...     [1, 1],
        ...     [1, 0],
        ...     ["male", "female"],
        ...     subgroups=["female", "male"],
        ...     reduction=lambda x: x[0] - x[1],
        ... )
        -1.0
        >>> unweighted_average_bias(
        ...     [0, 1], [1, 0], ["male", "female"], metric=recall_per_class
        ... )
        nan
        >>> unweighted_average_bias(
        ...     [0, 0, 0, 0],
        ...     [1, 1, 0, 0],
        ...     ["a", "b", "c", "d"],
        ...     metric=recall_per_class,
        ... )
        0.5

    """  # noqa: E501
    if labels is None:
        labels = infer_labels(truth, prediction)

    if not len(truth) == len(prediction) == len(protected_variable):
        raise ValueError(
            f"'truth', 'prediction' and 'protected_variable' should have "
            f"same lengths, but received '{len(truth)}', '{len(prediction)}' "
            f"and '{len(protected_variable)}'"
        )

    if subgroups is None:
        subgroups = sorted(set(protected_variable))

    scores = scores_per_subgroup_and_class(
        truth=truth,
        prediction=prediction,
        protected_variable=protected_variable,
        labels=labels,
        subgroups=subgroups,
        metric=metric,
        zero_division=np.nan,
    )

    bias = 0.0
    denominator = 0

    for label in labels:
        scores_subgroup = [
            scores[subgroup][label]
            for subgroup in subgroups
            if label in scores[subgroup] and not np.isnan(scores[subgroup][label])
        ]
        # compute score divergence only where more than 1 score per class
        if len(scores_subgroup) > 1:
            bias += reduction(scores_subgroup)
            denominator += 1

    if denominator == 0:
        return np.nan

    return float(bias / denominator)


def unweighted_average_fscore(
    truth: Sequence[object],
    prediction: Sequence[object],
    labels: Sequence[object] = None,
    *,
    zero_division: float = 0,
) -> float:
    r"""Unweighted average F-score.

    .. math::

        \text{UAF} = \frac{1}{K} \sum^K_{k=1}
            \frac{\text{true positive}_k}
                 {\text{true positive}_k + \frac{1}{2}
                 (\text{false positive}_k + \text{false negative}_k)}

    Args:
        truth: ground truth values/classes
        prediction: predicted values/classes
        labels: included labels in preferred ordering.
            If no labels are supplied,
            they will be inferred from
            :math:`\{\text{prediction}, \text{truth}\}`
            and ordered alphabetically.
        zero_division: set the value to return when there is a zero division

    Returns:
        unweighted average f-score

    Examples:
        >>> unweighted_average_fscore([0, 0], [0, 1])
        0.3333333333333333

    """
    fscore = fscore_per_class(
        truth,
        prediction,
        labels,
        zero_division=zero_division,
    )
    fscore = np.array(list(fscore.values()))
    return float(fscore.mean())


def unweighted_average_precision(
    truth: Sequence[object],
    prediction: Sequence[object],
    labels: Sequence[object] = None,
    *,
    zero_division: float = 0,
) -> float:
    r"""Unweighted average precision.

    .. math::

        \text{UAP} = \frac{1}{K} \sum^K_{k=1}
            \frac{\text{true positive}_k}
                 {\text{true positive}_k + \text{false positive}_k}

    Args:
        truth: ground truth values/classes
        prediction: predicted values/classes
        labels: included labels in preferred ordering.
            If no labels are supplied,
            they will be inferred from
            :math:`\{\text{prediction}, \text{truth}\}`
            and ordered alphabetically.
        zero_division: set the value to return when there is a zero division

    Returns:
        unweighted average precision

    Examples:
        >>> unweighted_average_precision([0, 0], [0, 1])
        0.5

    """
    precision = precision_per_class(
        truth,
        prediction,
        labels,
        zero_division=zero_division,
    )
    precision = np.array(list(precision.values()))
    return float(precision.mean())


def unweighted_average_recall(
    truth: Sequence[object],
    prediction: Sequence[object],
    labels: Sequence[object] = None,
    *,
    zero_division: float = 0,
) -> float:
    r"""Unweighted average recall.

    .. math::

        \text{UAR} = \frac{1}{K} \sum^K_{k=1}
            \frac{\text{true positive}_k}
                 {\text{true positive}_k + \text{false negative}_k}

    Args:
        truth: ground truth values/classes
        prediction: predicted values/classes
        labels: included labels in preferred ordering.
            If no labels are supplied,
            they will be inferred from
            :math:`\{\text{prediction}, \text{truth}\}`
            and ordered alphabetically.
        zero_division: set the value to return when there is a zero division

    Returns:
        unweighted average recall

    Examples:
        >>> unweighted_average_recall([0, 0], [0, 1])
        0.25

    """
    recall = recall_per_class(
        truth,
        prediction,
        labels,
        zero_division=zero_division,
    )
    recall = np.array(list(recall.values()))
    return float(recall.mean())


def weighted_confusion_error(
    truth: Sequence[object],
    prediction: Sequence[object],
    weights: Sequence[Sequence[int | float]],
    labels: Sequence[object] = None,
) -> float:
    r"""Weighted confusion error.

    Computes the normalized confusion matrix, applies given weights to each
    cell and sums them up. Weights are expected as positive numbers and
    will be normalized by the sum of all weights. The higher the weight,
    the more costly will be the error. A weight of 0 means that the cell
    is not taken into account for the error, this is usually the case for the
    diagonal as it holds correctly classified samples.

    Args:
        truth: ground truth values/classes
        prediction: predicted values/classes
        weights: weights applied to the confusion matrix.
            Expected as a list of lists in the following form
            (r=row, c=column):
            ``[[w_r0_c0, ..., w_r0_cN], ..., [w_rN_c0, ..., w_rN_cN]]``
        labels: included labels in preferred ordering.
            If no labels are supplied,
            they will be inferred from
            :math:`\{\text{prediction}, \text{truth}\}`
            and ordered alphabetically.

    Returns:
        weighted confusion error

    Examples:
        >>> truth = [0, 1, 2]
        >>> prediction = [0, 2, 0]
        >>> # penalize only errors > 1
        >>> weights = [[0, 0, 1], [0, 0, 0], [1, 0, 0]]
        >>> weighted_confusion_error(truth, prediction, weights)
        0.5

    """
    weights = weights / np.sum(weights)
    cm = confusion_matrix(truth, prediction, labels, normalize=True)
    cm = np.array(cm)

    if not cm.shape == weights.shape:
        raise ValueError(
            "Shape of weights "
            f"{weights.shape} "
            "does not match shape of confusion matrix "
            f"{cm.shape}."
        )

    weighted_cm = cm * weights
    return float(np.sum(weighted_cm))


def word_error_rate(
    truth: Sequence[Sequence[str]],
    prediction: Sequence[Sequence[str]],
    *,
    norm: str = "truth",
) -> float:
    r"""Word error rate based on edit distance.

    The word error rate is computed
    by aggregating the normalized edit distances
    of each (truth, prediction)-pair
    and averaging the aggregated score
    by the number of pairs.

    The normalized edit distance
    of each (truth, prediction)-pair is computed
    as the edit distance divided by a normalization factor n.
    This represents the average editing cost per sequence item.
    The value of n depends on the ``norm`` parameter.

    If ``norm`` is ``"truth"``,
    n is set to the reference (truth) length,
    following the Wikipedia formulation.
    Here, n is the number of words in the reference.
    This means WER can be greater than 1
    if the prediction sequence is longer than the reference:

        .. math::

            n = \text{len}(t)

    If ``norm`` is ``"longest"``,
    n is set to the maximum length between truth and prediction:

        .. math::

            n = \max(\text{len}(t), \text{len}(p))

    Args:
        truth: ground truth strings
        prediction: predicted strings
        norm: normalization method, either "truth" or "longest".
            "truth" normalizes by truth length,
            "longest" normalizes by max length of truth and prediction

    Returns:
        word error rate

    Raises:
        ValueError: if ``truth`` and ``prediction`` differ in length
        ValueError: if ``norm`` is not one of ``"truth"``, ``"longest"``

    Examples:
        >>> truth = [["lorem", "ipsum"], ["north", "wind", "and", "sun"]]
        >>> prediction = [["lorm", "ipsum"], ["north", "wind"]]
        >>> word_error_rate(truth, prediction)
        0.5
        >>> truth = [["hello", "world"]]
        >>> prediction = [["xyz", "moon", "abc"]]
        >>> word_error_rate(truth, prediction)
        1.5
        >>> word_error_rate(truth, prediction, norm="longest")
        1.0

    """
    assert_equal_length(truth, prediction)

    if norm not in ["truth", "longest"]:
        raise ValueError(f"'norm' must be one of 'truth', 'longest', got '{norm}'")

    wer = 0.0

    for t, p in zip(truth, prediction):
        # map words to ints
        unique_words = set(t).union(set(p))
        map = {k: v for k, v in zip(unique_words, range(len(unique_words)))}
        t = [map[i] for i in t]
        p = [map[i] for i in p]

        if norm == "longest":
            n = max(len(t), len(p))
        else:
            n = len(t)

        n = n if n > 1 else 1

        wer += edit_distance(t, p) / n

    num_samples = len(truth) if len(truth) > 1 else 1

    return float(wer / num_samples)


def _matching_scores(
    truth: (bool | int | Sequence[bool | int]),
    prediction: (bool | int | float | Sequence[bool | int | float]),
) -> tuple[np.ndarray, np.ndarray]:
    r"""Mated and non-mated scores for verification tasks.

    For verification task,
    predictions are usually separated
    in all predictions belonging
    to the matching examples,
    and all other predictions.
    The first are called mated scores
    or genuine matching scores,
    the second non-mated scores
    or impostor matching scores.

    For example,
    in a speaker verification task
    the mated scores are all similarity values
    that belong to the matching speaker.

    Args:
        truth: ground truth classes
        prediction: predicted classes or similarity scores

    Returns:
        * mated scores
        * non-mated scores

    Raises:
        ValueError: if ``truth`` contains values
            different from ``1, 0, True, False``

    Examples:
        >>> truth = [1, 0]
        >>> prediction = [0.9, 0.1]
        >>> _matching_scores(truth, prediction)
        (array([0.9]), array([0.1]))

    """
    truth = np.array(truth)

    allowed_truth_values = {1, 0, True, False}
    if not set(truth).issubset(allowed_truth_values):
        raise ValueError(
            "'truth' is only allowed to contain "
            "[1, 0, True, False], "
            "yours contains:\n"
            f"[{', '.join([str(t) for t in set(truth)])}]"
        )

    truth = truth.astype(bool)
    prediction = np.array(prediction).astype(np.float64)

    # Predictions for all matching examples
    # (truth is 1 or True)
    # In literature these are called
    # "genuine matching scores"
    # or "mated scores"
    mated_scores = prediction[truth]
    # Predictions for all non-matching examples
    # (truth is 0 or False)
    # In literature these are called
    # "impostor matching scores"
    # or "non-mated scores"
    non_mated_scores = prediction[~truth]

    return mated_scores, non_mated_scores


def _event_metric_per_class(
    truth: pd.Series,
    prediction: pd.Series,
    labels: Sequence[object] | None,
    zero_division: float,
    onset_tolerance: float | None,
    offset_tolerance: float | None,
    duration_tolerance: float | None,
    axis: int,  # 0=precision, 1=recall
) -> dict[str, float]:
    if labels is None:
        labels = infer_labels(truth, prediction)
    cm = np.array(
        event_confusion_matrix(
            truth,
            prediction,
            labels,
            onset_tolerance=onset_tolerance,
            offset_tolerance=offset_tolerance,
            duration_tolerance=duration_tolerance,
        )
    )
    totals = cm.sum(axis=axis)
    vals = cm.diagonal() / totals
    vals = np.nan_to_num(vals, nan=zero_division)
    # The event based confusion matrix also has a row/column
    # for the "no event" class (aka the absence of a segment)
    # but we only return the recall/precision per class
    return {lab: float(vals[i]) for i, lab in enumerate(labels)}


def _match_segments_and_accumulate(
    cm: list[list[int]],
    hit_matrix: np.ndarray,
    overlap_matrix: np.ndarray,
    truth_labels: list,
    pred_labels: list,
    label_to_index: dict,
):
    # First match correct‐label hits
    graph = csr_array(hit_matrix)
    matches = maximum_bipartite_matching(graph)
    for p_i, t_i in enumerate(matches):
        if t_i >= 0:
            idx = label_to_index[truth_labels[t_i]]
            cm[idx][idx] += 1
            overlap_matrix[t_i, :] = False
            overlap_matrix[:, p_i] = False

    # Then match leftover overlaps with wrong labels
    graph2 = csr_array(overlap_matrix)
    for p_i, t_i in enumerate(maximum_bipartite_matching(graph2)):
        if t_i >= 0:
            i = label_to_index[truth_labels[t_i]]
            j = label_to_index[pred_labels[p_i]]
            cm[i][j] += 1


def _segments_overlap(
    start_t, end_t, start_p, end_p, onset_tol, offset_tol, duration_tol
) -> bool:
    if onset_tol is not None:
        if math.fabs(start_t - start_p) > onset_tol:
            return False
    # Compute the effective offset tolerance
    eff_off = offset_tol or 0.0
    if duration_tol is not None:
        eff_off = max(eff_off, duration_tol * (end_t - start_t))
    if math.fabs(end_t - end_p) > eff_off:
        return False
    return True
