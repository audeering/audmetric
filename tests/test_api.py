import numpy as np
import pyeer.eer_info
import pytest
import sklearn.metrics

import audmetric


@pytest.mark.parametrize('truth,prediction,labels,to_string', [
    (
        np.random.randint(0, 10, size=5),
        np.random.randint(0, 10, size=5),
        None,
        False,
    ),
    (
        np.random.randint(0, 10, size=1),
        np.random.randint(0, 10, size=1),
        list(range(1, 10)),
        False,
    ),
    (
        np.random.randint(0, 10, size=10),
        np.random.randint(0, 10, size=10),
        list(range(1, 10)),
        False,
    ),
    (
        np.random.randint(0, 10, size=10),
        np.random.randint(0, 10, size=10),
        None,
        True,
    ),
    (
        np.array([]),
        np.array([]),
        None,
        False,
    ),
    (
        np.zeros(10),
        np.zeros(10),
        None,
        False,
    ),
    (
        np.arange(10),
        np.arange(10),
        list(range(1, 10)),
        False,
    ),
    (
        np.arange(10),
        np.arange(1, 11),
        list(range(1, 10)),
        False,
    ),
    (
        np.arange(5),
        np.array([1, 2, 3, 4, 6]),
        list(range(5)),
        False
    )
])
def test_accuracy(truth, prediction, labels, to_string):
    if to_string:
        truth = [str(w) for w in truth]
        prediction = [str(w) for w in prediction]

    if len(prediction) == 0:
        accuracy = np.NaN
    else:
        if labels:
            mask = np.nonzero(
                np.logical_and(
                    np.isin(truth, labels),
                    np.isin(prediction, labels)
                )
            )
            truth = truth[mask]
            prediction = prediction[mask]
        accuracy = sklearn.metrics.accuracy_score(truth, prediction)

    np.testing.assert_almost_equal(
        audmetric.accuracy(truth, prediction, labels=labels),
        accuracy,
    )


@pytest.mark.parametrize(
    'truth, prediction, expected_eer, expected_threshold',
    [
        (
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            0,
            1,
        ),
        (
            [True, True, False, False],
            [1, 1, 0, 0],
            0,
            1,
        ),
        (
            [True, True, False, False],
            [True, True, False, False],
            0,
            1,
        ),
        (
            [1, 1, 0, 0],
            [0.9, 0.9, 0.1, 0.1],
            0,
            0.9,
        ),
        (
            [1, 1, 0, 0],
            [1, 0.1, 0.1, 0],
            0.25,
            0.1,
        ),
        (
            [1, 1, 0, 0],
            [0.8, 0.7, 0.4, 0.1],
            0,
            0.7,
        ),
        (
            [1, 1, 0, 0, 0],
            [0.8, 0.7, 0.4, 0.1, 0.1],
            0,
            0.7,
        ),
        # Non integer truth not allowed
        pytest.param(
            [0.9, 0.9, 0.1, 0.1],
            [0.9, 0.9, 0.1, 0.1],
            0,
            1,
            marks=pytest.mark.xfail(raises=ValueError),
        ),
    ]
)
def test_equal_error_rate(truth, prediction, expected_eer, expected_threshold):
    eer, stats = audmetric.equal_error_rate(truth, prediction)
    # Check expected results
    assert type(eer) == float
    assert type(stats.threshold) == float
    assert eer == expected_eer
    assert stats.threshold == expected_threshold
    # Compare to pyeer package
    truth = np.array(truth)
    prediction = np.array(prediction)
    pyeer_stats = pyeer.eer_info.get_eer_stats(
        prediction[truth],
        prediction[~truth],
    )
    assert eer == pyeer_stats.eer
    assert stats.threshold == pyeer_stats.eer_th


def test_equal_error_rate_warnings():

    # No imposter scores (division by 0)
    truth = np.array([1, 1])
    prediction = np.array([1, 1])
    warning = 'invalid value encountered in true_divide'
    with pytest.warns(RuntimeWarning, match=warning):
        eer, stats = audmetric.equal_error_rate(truth, prediction)
        pyeer_stats = pyeer.eer_info.get_eer_stats(
            prediction[truth],
            prediction[~truth],
        )
        assert eer == pyeer_stats.eer
        assert stats.threshold == pyeer_stats.eer_th

    # Curves to not overlap
    truth = np.array([1, 1, 0])
    prediction = np.array([.5, .5, .5])
    warning = (
        r'false match rate and false non-match rate curves '
        r'do not intersect each other'
    )
    with pytest.warns(RuntimeWarning, match=warning):
        eer, stats = audmetric.equal_error_rate(truth, prediction)
        pyeer_stats = pyeer.eer_info.get_eer_stats(
            prediction[truth],
            prediction[~truth],
        )
        assert eer == pyeer_stats.eer
        assert stats.threshold == pyeer_stats.eer_th


@pytest.mark.parametrize(
    'truth,prediction,eer', [
        ([], [], 0),
        ([[]], [[]], 0),
        ([[None]], [[]], 1.),
        ([[None]], [[1]], 1.),
        ([[None]], [[1, 2]], 1.),
        ([[0], []], [[1], []], 0.5),
        ([[0, 1]], [[0]], 0.5),
        ([[0]], [[0, 1]], 0.5),
        ([[0, 1], [2]], [[0], [2]], 0.25),
        pytest.param(
            [[0, 1]], [[0], [2]], 0.,
            marks=pytest.mark.xfail(raises=ValueError)
        ),
        ('lorem', 'lorm', 0.2),
        (['lorem'], ['lorm'], 0.2),
        (['lorem', 'ipsum'], ['lorm', 'ipsum'], 0.1),
        pytest.param(
            ['lorem', 'ipsum'], ['lorm'], 0.,
            marks=pytest.mark.xfail(raises=ValueError),
        )
    ]
)
def test_event_error_rate(truth, prediction, eer):
    np.testing.assert_equal(
        audmetric.event_error_rate(truth, prediction),
        eer
    )


@pytest.mark.parametrize('truth,prediction', [
    (
        np.random.randint(0, 10, size=5),
        np.random.randint(0, 10, size=5),
    ),
    (
        np.random.randint(0, 10, size=1),
        np.random.randint(0, 10, size=1),
    ),
    (
        np.random.randint(0, 10, size=10),
        np.random.randint(0, 10, size=10),
    ),
    (
        np.random.randint(0, 2, size=100),
        np.random.randint(0, 2, size=100),
    ),
    (
        np.array([]),
        np.array([]),
    ),
    (
        np.zeros(10),
        np.zeros(10),
    ),
])
def test_concordancecc(truth, prediction):
    if len(prediction) < 2:
        ccc = np.NaN
    else:
        denominator = (
            prediction.std() ** 2
            + truth.std() ** 2
            + (prediction.mean() - truth.mean()) ** 2
        )
        if denominator == 0:
            ccc = np.NaN
        else:
            r = np.corrcoef(prediction, truth)[0][1]
            ccc = 2 * r * prediction.std() * truth.std() / denominator
    np.testing.assert_almost_equal(
        audmetric.concordance_cc(truth, prediction),
        ccc,
    )


@pytest.mark.parametrize('class_range,num_elements,to_string,percentage', [
    ([0, 10], 5, False, False),
    ([0, 10], 1, False, False),
    ([0, 10], 10, False, False),
    ([0, 2], 100, False, False),
    ([0, 10], 10, True, False),
    ([0, 10], 100, True, True),
    ([0, 10], 10, True, True),
])
def test_confusion_matrix(class_range, num_elements, to_string, percentage):

    t = np.random.randint(class_range[0], class_range[1], size=num_elements)
    p = np.random.randint(class_range[0], class_range[1], size=num_elements)

    if to_string:
        t = [str(w) for w in t]
        p = [str(w) for w in p]

    cm = audmetric.confusion_matrix(t, p, normalize=percentage)

    if percentage:
        cm_sklearn = sklearn.metrics.confusion_matrix(t, p, normalize='true')
    else:
        cm_sklearn = sklearn.metrics.confusion_matrix(t, p)

    np.testing.assert_almost_equal(cm, cm_sklearn)


@pytest.mark.parametrize(
    'truth,prediction,edit_distance', [
        ('lorem', 'lorem', 0),
        ('lorem', '', 5),
        ('', 'lorem', 5),
        ('lorem', 'lorm', 1),
        ('lorem', 'lorrem', 1),
        ('lorem', 'lorom', 1),
        ('lorem', 'morel', 2),
        ([], [0], 1),
        ([0], [], 1),
        ([0, 1, 2], [0, 1], 1),
        ([0, 1, 2], [0, 1, 1], 1),
        ([None], [], 1),
        ([None], [1], 1),
        ([None], [1, 2], 2)
    ]
)
def test_edit_distance(truth, prediction, edit_distance):
    np.testing.assert_equal(
        audmetric.edit_distance(truth, prediction),
        edit_distance
    )


@pytest.mark.parametrize('value_range,num_elements', [
    ([0, 10], 5),
    ([0, 10], 1),
    ([0, 10], 10),
    ([0, 2], 100),
])
def test_mean_absolute_error(value_range, num_elements):
    t = np.random.randint(value_range[0], value_range[1], size=num_elements)
    p = np.random.randint(value_range[0], value_range[1], size=num_elements)

    np.testing.assert_almost_equal(
        audmetric.mean_absolute_error(t, p),
        sklearn.metrics.mean_absolute_error(t, p),
    )


@pytest.mark.parametrize('value_range,num_elements', [
    ([0, 10], 5),
    ([0, 10], 1),
    ([0, 10], 10),
    ([0, 2], 100),
])
def test_mean_squared_error(value_range, num_elements):
    t = np.random.randint(value_range[0], value_range[1], size=num_elements)
    p = np.random.randint(value_range[0], value_range[1], size=num_elements)

    np.testing.assert_almost_equal(
        audmetric.mean_squared_error(t, p),
        sklearn.metrics.mean_squared_error(t, p),
    )


@pytest.mark.parametrize('truth,prediction', [
    (
        np.random.randint(0, 10, size=5),
        np.random.randint(0, 10, size=5),
    ),
    (
        np.random.randint(0, 10, size=1),
        np.random.randint(0, 10, size=1),
    ),
    (
        np.random.randint(0, 10, size=10),
        np.random.randint(0, 10, size=10),
    ),
    (
        np.random.randint(0, 2, size=100),
        np.random.randint(0, 2, size=100),
    ),
    (
        np.array([]),
        np.array([]),
    ),
    (
        np.zeros(10),
        np.zeros(10),
    ),
])
def test_pearsoncc(truth, prediction):
    if len(prediction) < 2 or prediction.std() == 0:
        pcc = np.NaN
    else:
        pcc = np.corrcoef(truth, prediction)[0][1]
    np.testing.assert_almost_equal(
        audmetric.pearson_cc(truth, prediction),
        pcc,
    )


@pytest.mark.parametrize(
    'truth, prediction, labels, zero_division',
    [
        (
            ['a'],
            ['a'],
            None,
            0,
        ),
        (
            ['a'],
            ['b'],
            None,
            0,
        ),
        (
            ['a'],
            ['b'],
            ['a', 'b'],
            0,
        ),
        (
            ['a'],
            ['b'],
            ['a', 'b'],
            1,
        ),
        (
            ['a', 'b'],
            ['b', 'a'],
            None,
            0,
        ),
        (
            np.random.randint(0, 10, 5),
            np.random.randint(0, 10, 5),
            None,
            0,
        ),
        (
            np.random.randint(0, 10, 100),
            np.random.randint(0, 10, 100),
            None,
            0,
        )
    ]
)
def test_recall_precision_fscore(truth, prediction, labels, zero_division):

    for metric, sklearn_metric in (
        (
            audmetric.unweighted_average_recall,
            sklearn.metrics.recall_score,
        ),
        (
            audmetric.unweighted_average_precision,
            sklearn.metrics.precision_score,
        ),
        (
            audmetric.unweighted_average_fscore,
            sklearn.metrics.f1_score,
        )
    ):
        result = metric(
            truth,
            prediction,
            labels,
            zero_division=zero_division,
        )
        expected = sklearn_metric(
            truth,
            prediction,
            average='macro',
            zero_division=zero_division,
        ),
        np.testing.assert_almost_equal(
            result,
            expected,
        )


@pytest.mark.parametrize(
    'truth,prediction,protected_variable,metric,labels,subgroups,'
    'zero_division,expected',
    [
        pytest.param(
            [], [], [], None, [], [0], None, {},
            marks=pytest.mark.xfail(raises=ValueError)
        ),
        (
            [], [], [], audmetric.recall_per_class, [], [], 0., {}
        ),
        (
            [1], [0], [0], audmetric.recall_per_class, [0, 1],
            [0], 0., {0: {0: 0.0, 1: 0.0}}
        ),
        (
            [1], [0], [0], audmetric.precision_per_class, [0, 1],
            [0], 0., {0: {0: 0.0, 1: 0.0}}
        ),
        (
            [1, 1], [0, 1], [0, 1], audmetric.recall_per_class, [0, 1],
            [0, 1], np.nan, {0: {0: np.nan, 1: 0.0}, 1: {0: np.nan, 1: 1.0}}
        ),
        (
            [1, 1], [0, 1], [0, 1], audmetric.recall_per_class, [1],
            [0, 1], np.nan, {0: {1: np.nan}, 1: {1: 1.0}}
        ),
        (
            [1, 1], [0, 1], [0, 1], audmetric.precision_per_class, [0, 1],
            [0, 1], np.nan, {0: {0: 0.0, 1: np.nan}, 1: {0: np.nan, 1: 1.0}}
        )
    ]
)
def test_scores_per_subgroup_and_class(
        truth, prediction, protected_variable, metric, labels, subgroups,
        zero_division, expected):
    np.testing.assert_equal(
        audmetric.core.utils.scores_per_subgroup_and_class(
            truth, prediction, protected_variable, metric,
            labels=labels,
            subgroups=subgroups,
            zero_division=zero_division
        ), expected
    )


@pytest.mark.parametrize(
    'truth,prediction,protected_variable,labels,subgroups,metric,reduction,'
    'expected',
    [
        (
            [],
            [],
            [],
            None,
            None,
            audmetric.fscore_per_class,
            lambda x: abs(x[0] - x[1]),
            np.nan,
        ),
        (
            [0],
            [0],
            [0],
            None,
            None,
            audmetric.recall_per_class,
            lambda x: abs(x[0] - x[1]),
            np.nan,
        ),
        pytest.param(
            [0, 0],
            [0],
            [0],
            None,
            None,
            audmetric.recall_per_class,
            lambda x: abs(x[0] - x[1]),
            None,
            marks=pytest.mark.xfail(raises=ValueError)
        ),
        pytest.param(
            [0, 0],
            [0, 0],
            [0, 0],
            None,
            [0, 1],
            audmetric.recall_per_class,
            lambda x: abs(x[0] - x[1]),
            None,
            marks=pytest.mark.xfail(raises=ValueError)
        ),
        (
            [0, 0],
            [0, 0],
            [0, 1],
            None,
            None,
            audmetric.recall_per_class,
            lambda x: abs(x[0] - x[1]),
            0.0
        ),
        (
            [0, 1],
            [0, 0],
            [0, 1],
            None,
            None,
            audmetric.recall_per_class,
            lambda x: abs(x[0] - x[1]),
            np.nan
        ),
        (
            [0, 1],
            [0, 0],
            [0, 1],
            None,
            None,
            audmetric.precision_per_class,
            lambda x: abs(x[0] - x[1]),
            1.0
        ),
        (
            [0, 1],
            [0, 0],
            [0, 1],
            [0, 1, 2, 3],
            None,
            audmetric.precision_per_class,
            lambda x: abs(x[0] - x[1]),
            1.0
        ),
        (
            [1, 1],
            [0, 1],
            [0, 1],
            None,
            None,
            audmetric.recall_per_class,
            lambda x: abs(x[0] - x[1]),
            1.0
        ),
        (
            [1, 1],
            [0, 1],
            [0, 1],
            None,
            None,
            audmetric.recall_per_class,
            lambda x: x[0] - x[1],
            -1.0
        ),
        (
            [1, 1],
            [0, 1],
            [0, 1],
            None,
            [1, 0],
            audmetric.recall_per_class,
            lambda x: x[0] - x[1],
            1.0
        ),
        (
            [1, 1],
            [0, 1],
            [0, 1],
            None,
            [1, 0],
            audmetric.fscore_per_class,
            lambda x: abs(x[0] - x[1]),
            1.0
        ),
        (
            [1, 1],
            [0, 1],
            [0, 1],
            None,
            None,
            audmetric.recall_per_class,
            lambda x: abs(x[0] - x[1]),
            1.0
        ),
        (
            [1, 1, 2],
            [0, 1, 2],
            [0, 1, 1],
            None,
            None,
            audmetric.recall_per_class,
            lambda x: abs(x[0] - x[1]),
            1.0
        ),
        (
            [1, 1, 2, 2],
            [0, 1, 2, 2],
            [0, 1, 1, 0],
            None,
            None,
            audmetric.recall_per_class,
            lambda x: abs(x[0] - x[1]),
            0.5
        ),
        (
            [1, 1, 2, 2],
            [0, 1, 2, 2],
            [0, 1, 1, 0],
            None,
            None,
            audmetric.recall_per_class,
            lambda x: x[0] - x[1],
            -0.5
        ),
        (
            [0, 0, 0, 0],
            [1, 1, 0, 0],
            [0, 1, 2, 3],
            None,
            None,
            audmetric.recall_per_class,
            np.std,
            0.5
        )
    ]
)
def test_unweighted_average_bias(
        truth, prediction, protected_variable, labels, subgroups, metric,
        reduction, expected
):
    np.testing.assert_equal(
        audmetric.unweighted_average_bias(
            truth,
            prediction,
            protected_variable,
            labels=labels,
            subgroups=subgroups,
            metric=metric,
            reduction=reduction
        ),
        expected
    )


def test_recall_precision_fscore_nan():

    truth = ['a', 'b']
    prediction = ['a', 'a']
    labels = ['a', 'b', 'c']

    for metric in (
            audmetric.unweighted_average_recall,
            audmetric.unweighted_average_precision,
            audmetric.unweighted_average_fscore,
    ):
        result = metric(
            truth,
            prediction,
            labels,
            zero_division=np.nan,
        )
        np.testing.assert_equal(result, np.nan)

    result = audmetric.recall_per_class(
        truth,
        prediction,
        labels,
        zero_division=np.nan,
    )
    np.testing.assert_equal(result, {'a': 1.0, 'b': 0.0, 'c': np.nan})

    result = audmetric.precision_per_class(
        truth,
        prediction,
        labels,
        zero_division=np.nan,
    )
    np.testing.assert_equal(result, {'a': 0.5, 'b': np.nan, 'c': np.nan})

    result = audmetric.fscore_per_class(
        truth,
        prediction,
        labels,
        zero_division=np.nan,
    )
    np.testing.assert_equal(result, {'a': 2 / 3, 'b': 0.0, 'c': np.nan})


@pytest.mark.parametrize(
    'truth, prediction, labels, zero_division, expected',
    [
        (
            ['a'],
            ['a'],
            ['a', 'b'],
            np.nan,
            np.nan,
        ),
        (
            ['a'],
            ['a'],
            ['a', 'b'],
            0,
            0.5,
        ),
        (
            ['a'],
            ['a'],
            ['a', 'b'],
            1,
            1,
        ),
    ]
)
def test_unweighted_average_zero_division(truth, prediction, labels,
                                          zero_division, expected):

    for metric in (
        audmetric.unweighted_average_recall,
        audmetric.unweighted_average_precision,
        audmetric.unweighted_average_fscore,
    ):
        result = metric(
            truth,
            prediction,
            labels,
            zero_division=zero_division,
        )
        np.testing.assert_equal(result, expected)


@pytest.mark.parametrize('weights,num_elements,to_string', [
    (
        [
            [0, 1, 2],
            [1, 0, 1],
            [2, 1, 0],
        ],
        5,
        False,
    ),
    (
        [
            [0, 1, 2],
            [1, 0, 1],
            [2, 1, 0],
        ],
        5,
        True,
    ),
    pytest.param(  # shape of cm and weights do not match
        [
            [0, 1, 2],
            [1, 0, 1],
        ],
        5,
        False,
        marks=pytest.mark.xfail(raises=ValueError),
    ),
])
def test_weighted_confusion_error(weights, num_elements, to_string):

    n = len(weights)
    t = np.zeros(num_elements, dtype=int)
    t[:n] = range(n)
    p = np.random.randint(0, len(weights), size=num_elements)

    if to_string:
        t = [str(w) for w in t]
        p = [str(w) for w in p]

    wce = audmetric.weighted_confusion_error(t, p, weights)
    cm = audmetric.confusion_matrix(t, p, normalize=True)
    weights = weights / np.sum(weights)
    np.testing.assert_equal(
        wce,
        float(np.sum(cm * weights)),
    )


@pytest.mark.parametrize(
    'truth,prediction,wer', [
        ([[]], [[]], 0),
        ([[None]], [[]], 1.),
        ([[None]], [['lorem']], 1.),
        ([[None]], [['lorem', 'ipsum']], 1.),
        ([['lorem']], [[]], 1),
        ([[]], [['lorem']], 1),
        ([['lorem', 'ipsum']], [['lorem']], 0.5),
        ([['lorem']], [['lorem', 'ipsum']], 0.5),
        ([['lorem']], [['lorem']], 0),
        ([['lorem', 'ipsum']], [['lorm', 'ipsum']], 0.5),
        (
            [['lorem', 'ipsum'], ['north', 'wind', 'and', 'sun']],
            [['lorm', 'ipsum'], ['north', 'wind']],
            0.5
        ),
        pytest.param(
            [['lorem'], []], [[]], 0.,
            marks=pytest.mark.xfail(raises=ValueError),
        )
    ]
)
def test_word_error_rate(truth, prediction, wer):
    np.testing.assert_equal(
        audmetric.word_error_rate(truth, prediction),
        wer
    )
