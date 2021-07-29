from collections import namedtuple
import operator
import typing
import warnings

import numpy as np

import audeer

from audmetric.core.utils import (
    assert_equal_length,
    infer_labels,
    scores_per_subgroup_and_class,
)


def accuracy(
        truth: typing.Sequence[typing.Any],
        prediction: typing.Sequence[typing.Any],
        labels: typing.Sequence[typing.Union[str, int]] = None
) -> float:
    r"""Classification accuracy.

    .. math::

        \text{accuracy} = \frac{\text{number of correct predictions}}
                               {\text{number of incorrect predictions}}

    Args:
        truth: ground truth values/classes
        prediction: predicted values/classes
        labels: included labels in preferred ordering.
            Sample is considered in computation if either prediction or
            ground truth (logical OR) is contained in labels.
            If no labels are supplied,
            they will inferred from :math:`\{\text{prediction}, \text{truth}\}`
            and ordered alphabetically.

    Returns:
        accuracy of prediction :math:`\in [0, 1]`

    Raises:
        ValueError: if ``truth`` and ``prediction`` differ in length

    Example:
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
        np.logical_or(
            np.isin(truth, labels),
            np.isin(prediction, labels)
        )
    )
    truth = truth[label_mask]
    prediction = prediction[label_mask]

    if len(prediction) == 0:
        return np.nan
    else:
        return float(sum(prediction == truth) / len(prediction))


def concordance_cc(
        truth: typing.Sequence[float],
        prediction: typing.Sequence[float],
) -> float:
    r"""Concordance correlation coefficient.

    .. math::

        \rho_c = \frac{2\rho\sigma_\text{prediction}\sigma_\text{truth}}
                      {\sigma_\text{prediction}^2 + \sigma_\text{truth}^2 + (
                      \mu_\text{prediction}-\mu_\text{truth})^2}

    where :math:`\rho` is the Pearson correlation coefficient,
    :math:`\mu` the mean and :math:`\sigma^2` the variance.

    Args:
        truth: ground truth values
        prediction: predicted values

    Returns:
        concordance correlation coefficient :math:`\in [-1, 1]`

    Raises:
        ValueError: if ``truth`` and ``prediction`` differ in length

    Example:
        >>> concordance_cc([0, 1, 2], [0, 1, 1])
        0.6666666666666666

    """
    assert_equal_length(truth, prediction)

    prediction = np.array(prediction)
    truth = np.array(truth)

    if len(prediction) < 2:
        return np.NaN

    r = pearson_cc(prediction, truth)
    x_mean = prediction.mean()
    y_mean = truth.mean()
    x_std = prediction.std()
    y_std = truth.std()
    denominator = (
        x_std * x_std
        + y_std * y_std
        + (x_mean - y_mean) * (x_mean - y_mean)
    )
    if denominator == 0:
        ccc = np.nan
    else:
        ccc = 2 * r * x_std * y_std / denominator

    return float(ccc)


def confusion_matrix(
        truth: typing.Sequence[typing.Any],
        prediction: typing.Sequence[typing.Any],
        labels: typing.Sequence[typing.Any] = None,
        *,
        normalize: bool = False,
) -> typing.List[typing.List[typing.Union[int, float]]]:
    r"""Confusion matrix.

    Args:
        truth: ground truth values/classes
        prediction: predicted values/classes
        labels: included labels in preferred ordering.
            If no labels are supplied,
            they will inferred from :math:`\{\text{prediction}, \text{truth}\}`
            and ordered alphabetically.
        normalize: normalize confusion matrix over the rows

    Returns:
        confusion matrix

    Raises:
        ValueError: if ``truth`` and ``prediction`` differ in length

    Example:
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
    truth: typing.Union[
        typing.Union[bool, int],
        typing.Sequence[typing.Union[bool, int]]
    ],
    prediction: typing.Union[
        typing.Union[bool, int, float],
        typing.Sequence[typing.Union[bool, int, float]]
    ],
) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
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

    The implementation is identical with the one provided
    by the pyeer_ package.

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

    Example:
        >>> truth = [1, 0]
        >>> prediction = [0.9, 0.1]
        >>> detection_error_tradeoff(truth, prediction)
        (array([1., 0.]), array([0., 0.]), array([0.1, 0.9]))

    """  # noqa: E501
    truth = np.array(truth)

    allowed_truth_values = set([1, 0, True, False])
    if not set(truth).issubset(allowed_truth_values):
        raise ValueError(
            "'truth' is only allowed to contain "
            "[1, 0, True, False], "
            'yours contains:\n'
            f"[{', '.join([str(t) for t in set(truth)])}]"
        )

    truth = truth.astype(bool)
    prediction = np.array(prediction).astype(np.float64)

    # Genuine matching scores
    gscores = prediction[truth]
    # Impostor matching scores
    iscores = prediction[~truth]

    gscores_number = len(gscores)
    iscores_number = len(iscores)

    # Labeling genuine scores as 1 and impostor scores as 0
    gscores = list(zip(gscores, [1] * gscores_number))
    iscores = list(zip(iscores, [0] * iscores_number))

    # Stacking scores
    scores = np.array(sorted(gscores + iscores, key=operator.itemgetter(0)))
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
    truth: typing.Union[str, typing.Sequence[int]],
    prediction: typing.Union[str, typing.Sequence[int]]
) -> int:
    r"""Edit distance between two strings of characters or sequences of ints.

    The implementation follows the `Wagner-Fischer algorithm`_.

    .. _Wagner-Fischer algorithm:
        https://en.wikipedia.org/wiki/Wagner%E2%80%93Fischer_algorithm

    Args:
        truth: ground truth sequence
        prediction: predicted sequence

    Returns:
        edit distance

    Example:
        >>> truth = 'lorem'
        >>> prediction = 'lorm'
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
            m1[j + 1] = min(m1[j] + 1,       # deletion
                            m0[j + 1] + 1,   # insertion
                            m0[j] + cost)    # substitution

        for j in range(len(m0)):
            m0[j] = m1[j]

    return m1[len(truth)]


def equal_error_rate(
    truth: typing.Union[
        typing.Union[bool, int],
        typing.Sequence[typing.Union[bool, int]]
    ],
    prediction: typing.Union[
        typing.Union[bool, int, float],
        typing.Sequence[typing.Union[bool, int, float]]
    ],
) -> typing.Tuple[float, namedtuple]:
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

    Example:
        >>> truth = [0, 1, 0, 1, 0]
        >>> prediction = [0.2, 0.8, 0.4, 0.5, 0.5]
        >>> eer, stats = equal_error_rate(truth, prediction)
        >>> eer
        0.16666666666666666
        >>> stats.threshold
        0.5

    """
    Stats = namedtuple(
        'stats',
        [
            'fmr',  # False match rates (FMR)
            'fnmr',  # False non-match rates (FNMR)
            'thresholds',  # Thresholds
            'threshold',  # verification threshold for EER
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
            'The false match rate '
            'and false non-match rate curves '
            'do not intersect each other.',
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


def event_error_rate(
    truth: typing.Union[
        str, typing.Sequence[typing.Union[str, typing.Sequence[int]]]
    ],
    prediction: typing.Union[
        str, typing.Sequence[typing.Union[str, typing.Sequence[int]]]
    ],
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

    Example:
        >>> event_error_rate([[0, 1]], [[0]])
        0.5
        >>> event_error_rate([[0, 1], [2]], [[0], [2]])
        0.25
        >>> event_error_rate(['lorem'], ['lorm'])
        0.2
        >>> event_error_rate(['lorem', 'ipsum'], ['lorm', 'ipsum'])
        0.1

    """
    truth = audeer.to_list(truth)
    prediction = audeer.to_list(prediction)

    assert_equal_length(truth, prediction)

    eer = 0.

    for t, p in zip(truth, prediction):
        n = max(len(t), len(p))
        n = n if n > 1 else 1
        eer += edit_distance(t, p) / n

    num_samples = len(truth) if len(truth) > 1 else 1
    return eer / num_samples


def fscore_per_class(
        truth: typing.Sequence[typing.Any],
        prediction: typing.Sequence[typing.Any],
        labels: typing.Sequence[typing.Any] = None,
        *,
        zero_division: float = 0,
) -> typing.Dict[str, float]:
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
            they will inferred from :math:`\{\text{prediction}, \text{truth}\}`
            and ordered alphabetically.
        zero_division: set the value to return when there is a zero division

    Returns:
        dictionary with label as key and F-score as value

    Example:
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


def mean_absolute_error(
        truth: typing.Sequence[float],
        prediction: typing.Sequence[float],
) -> float:
    r"""Mean absolute error.

    .. math::

        \text{MSE} = \frac{1}{n} \sum^n_{i=1}
            |\text{prediction} - \text{truth}|

    Args:
        truth: ground truth values
        prediction: predicted values

    Returns:
        mean absolute error

    Raises:
        ValueError: if ``truth`` and ``prediction`` differ in length

    Example:
        >>> mean_absolute_error([0, 0], [0, 1])
        0.5

    """
    assert_equal_length(truth, prediction)

    prediction = np.array(prediction)
    truth = np.array(truth)

    return float(np.abs(truth - prediction).mean(axis=0))


def mean_squared_error(
        truth: typing.Sequence[float],
        prediction: typing.Sequence[float],
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

    Example:
        >>> mean_squared_error([0, 0], [0, 1])
        0.5

    """
    assert_equal_length(truth, prediction)

    prediction = np.array(prediction)
    truth = np.array(truth)

    return float(np.square(truth - prediction).mean(axis=0))


def pearson_cc(
        truth: typing.Sequence[float],
        prediction: typing.Sequence[float],
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

    Example:
        >>> pearson_cc([0, 1, 2], [0, 1, 1])
        0.8660254037844385

    """
    assert_equal_length(truth, prediction)

    prediction = np.array(prediction)
    truth = np.array(truth)

    if len(prediction) < 2 or prediction.std() == 0:
        return np.NaN
    else:
        return float(np.corrcoef(prediction, truth)[0][1])


def precision_per_class(
        truth: typing.Sequence[typing.Any],
        prediction: typing.Sequence[typing.Any],
        labels: typing.Sequence[typing.Any] = None,
        *,
        zero_division: float = 0,
) -> typing.Dict[str, float]:
    r"""Precision per class.

    .. math::

        \text{precision}_k = \frac{\text{true positive}_k}
                 {\text{true positive}_k + \text{false positive}_k}

    Args:
        truth: ground truth values/classes
        prediction: predicted values/classes
        labels: included labels in preferred ordering.
            If no labels are supplied,
            they will inferred from :math:`\{\text{prediction}, \text{truth}\}`
            and ordered alphabetically.
        zero_division: set the value to return when there is a zero division

    Returns:
        dictionary with label as key and precision as value

    Example:
        >>> precision_per_class([0, 0], [0, 1])
        {0: 1.0, 1: 0.0}

    """
    if labels is None:
        labels = infer_labels(truth, prediction)

    matrix = np.array(confusion_matrix(truth, prediction, labels))
    total = matrix.sum(axis=0)
    old_settings = np.seterr(invalid='ignore')
    recall = matrix.diagonal() / total
    np.seterr(**old_settings)
    recall[np.isnan(recall)] = zero_division

    return {label: float(r) for label, r in zip(labels, recall)}


def recall_per_class(
        truth: typing.Sequence[typing.Any],
        prediction: typing.Sequence[typing.Any],
        labels: typing.Sequence[typing.Any] = None,
        *,
        zero_division: float = 0,
) -> typing.Dict[str, float]:
    r"""Recall per class.

    .. math::

        \text{recall}_k = \frac{\text{true positive}_k}
                 {\text{true positive}_k + \text{false negative}_k}

    Args:
        truth: ground truth values/classes
        prediction: predicted values/classes
        labels: included labels in preferred ordering.
            If no labels are supplied,
            they will inferred from :math:`\{\text{prediction}, \text{truth}\}`
            and ordered alphabetically.
        zero_division: set the value to return when there is a zero division

    Returns:
        dictionary with label as key and recall as value

    Example:
        >>> recall_per_class([0, 0], [0, 1])
        {0: 0.5, 1: 0.0}

    """
    if labels is None:
        labels = infer_labels(truth, prediction)

    matrix = np.array(confusion_matrix(truth, prediction, labels))
    total = matrix.sum(axis=1)
    old_settings = np.seterr(invalid='ignore')
    recall = matrix.diagonal() / total
    np.seterr(**old_settings)
    recall[np.isnan(recall)] = zero_division

    return {label: float(r) for label, r in zip(labels, recall)}


def unweighted_average_bias(
        truth: typing.Sequence[typing.Any],
        prediction: typing.Sequence[typing.Any],
        protected_variable: typing.Sequence[typing.Any],
        labels: typing.Sequence[typing.Any] = None,
        *,
        subgroups: typing.Sequence[typing.Any] = None,
        metric: typing.Callable[
            [
                typing.Sequence[typing.Any],
                typing.Sequence[typing.Any],
                typing.Optional[typing.Sequence[str]],
            ],
            typing.Dict[str, float]
        ] = fscore_per_class,
        reduction: typing.Callable[
            [
                typing.Sequence[float],
            ],
            float,
        ] = np.std,
) -> float:
    r"""Compute unweighted average bias with respect to a protected variable.

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
            they will inferred from :math:`\{\text{prediction}, \text{truth}\}`
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

    Example:
        >>> unweighted_average_bias([1, 1], [1, 0], ['male', 'female'])
        0.5
        >>> unweighted_average_bias(
        ...     [1, 1], [1, 0], ['male', 'female'],
        ...     subgroups=['female', 'male'],
        ...     reduction=lambda x: x[0] - x[1],
        ... )
        -1.0
        >>> unweighted_average_bias(
        ...     [0, 1], [1, 0], ['male', 'female'],
        ...     metric=recall_per_class
        ... )
        nan
        >>> unweighted_average_bias(
        ...     [0, 0, 0, 0], [1, 1, 0, 0], ['a', 'b', 'c', 'd'],
        ...     metric=recall_per_class,
        ... )
        0.5

    """
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

    bias = 0.
    denominator = 0

    for label in labels:
        scores_subgroup = [
            scores[subgroup][label] for subgroup in subgroups
            if label in scores[subgroup]
            and not np.isnan(scores[subgroup][label])
        ]
        # compute score divergence only where more than 1 score per class
        if len(scores_subgroup) > 1:
            bias += reduction(scores_subgroup)
            denominator += 1

    if denominator == 0:
        return np.nan

    return bias / denominator


def unweighted_average_fscore(
        truth: typing.Sequence[typing.Any],
        prediction: typing.Sequence[typing.Any],
        labels: typing.Sequence[typing.Any] = None,
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
            they will inferred from :math:`\{\text{prediction}, \text{truth}\}`
            and ordered alphabetically.
        zero_division: set the value to return when there is a zero division

    Returns:
        unweighted average precision

    Example:
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
        truth: typing.Sequence[typing.Any],
        prediction: typing.Sequence[typing.Any],
        labels: typing.Sequence[typing.Any] = None,
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
            they will inferred from :math:`\{\text{prediction}, \text{truth}\}`
            and ordered alphabetically.
        zero_division: set the value to return when there is a zero division

    Returns:
        unweighted average precision

    Example:
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
        truth: typing.Sequence[typing.Any],
        prediction: typing.Sequence[typing.Any],
        labels: typing.Sequence[typing.Any] = None,
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
            they will inferred from :math:`\{\text{prediction}, \text{truth}\}`
            and ordered alphabetically.
        zero_division: set the value to return when there is a zero division

    Returns:
        unweighted average recall

    Example:
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
    truth: typing.Sequence[typing.Any],
    prediction: typing.Sequence[typing.Any],
    weights: typing.Sequence[typing.Sequence[typing.Union[int, float]]],
    labels: typing.Sequence[typing.Any] = None,
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
            they will inferred from :math:`\{\text{prediction}, \text{truth}\}`
            and ordered alphabetically.

    Returns:
        weighted confusion error

    Example:

        >>> truth = [0, 1, 2]
        >>> prediction = [0, 2, 0]
        >>> # penalize only errors > 1
        >>> weights = [[0, 0 , 1], [0, 0, 0], [1, 0, 0]]
        >>> weighted_confusion_error(truth, prediction, weights)
        0.5

    """
    weights = weights / np.sum(weights)
    cm = confusion_matrix(truth, prediction, labels, normalize=True)
    cm = np.array(cm)

    if not cm.shape == weights.shape:
        raise ValueError(
            'Shape of weights '
            f'{weights.shape} '
            'does not match shape of confusion matrix '
            f'{cm.shape}.'
        )

    weighted_cm = cm * weights
    return float(np.sum(weighted_cm))


def word_error_rate(
    truth: typing.Sequence[typing.Sequence[str]],
    prediction: typing.Sequence[typing.Sequence[str]]
) -> float:
    r"""Word error rate based on edit distance.

    Args:
        truth: ground truth strings
        prediction: predicted strings

    Returns:
        word error rate

    Raises:
        ValueError: if ``truth`` and ``prediction`` differ in length

    Example:
        >>> truth = [['lorem', 'ipsum'], ['north', 'wind', 'and', 'sun']]
        >>> prediction = [['lorm', 'ipsum'], ['north', 'wind']]
        >>> word_error_rate(truth, prediction)
        0.5

    """
    assert_equal_length(truth, prediction)

    wer = 0.

    for t, p in zip(truth, prediction):
        # map words to ints
        unique_words = set(t).union(set(p))
        map = {k: v for k, v in zip(unique_words, range(len(unique_words)))}
        t = [map[i] for i in t]
        p = [map[i] for i in p]

        n = max(len(t), len(p))
        n = n if n > 1 else 1
        wer += edit_distance(t, p) / n

    num_samples = len(truth) if len(truth) > 1 else 1
    return wer / num_samples
