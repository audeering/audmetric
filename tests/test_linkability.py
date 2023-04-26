import numpy as np
import pytest

import audeer
import audmetric


# The expected values were manually calculated using code from
# https://gitlab.inria.fr/magnet/anonymization_metrics.
# See tests/assests/linkability/README.md
# for instructions
#
# To ensure reproducibility of the signals
# we create our own signal generator functions
#
def truth(samples=1000):
    r"""Generate truth values for linkability test.

    The truth values are in the form:
    ``[0, 1, 0, 1, ..., 0, 1]``

    """
    return [0, 1] * samples


def prediction(range1, range2, samples=1000, random_state=1):
    r"""Generate prediction values for linkability test.

    The values are returned as pairs
    drawn from uniform distribution
    given by ``range1`` and ``range2``, e.g.
    ``[0.12, 0.84, 0.07, 0.81, ..., 0.13, 0.84]``

    """
    np.random.seed(random_state)
    return audeer.flatten_list(
        [
            [np.random.uniform(*range1), np.random.uniform(*range2)]
            for _ in range(samples)
        ]
    )


@pytest.mark.parametrize(
    'truth, prediction, omega, expected',
    [
        (
            # Distribution 1: ##________
            # Distribution 2: ________##
            # Guessing: 0.5
            truth(),
            prediction((0.0, 0.2), (0.8, 1.0)),
            1,
            0.9710,
        ),
        (
            # Distribution 1: ##________
            # Distribution 2: ________##
            # Guessing: 0.1
            truth(),
            prediction((0.0, 0.2), (0.8, 1.0)),
            1 / 9.0,
            0.9710,
        ),
        (
            # Distribution 1: _##_______
            # Distribution 2: _______##_
            # Guessing: 0.5
            truth(),
            prediction((0.1, 0.3), (0.7, 0.9)),
            1,
            0.9775,
        ),
        (
            # Distribution 1: __##______
            # Distribution 2: ______##__
            # Guessing: 0.5
            truth(),
            prediction((0.2, 0.4), (0.6, 0.8)),
            1,
            0.9840,
        ),
        (
            # Distribution 1: ___##_____
            # Distribution 2: _____##___
            # Guessing: 0.5
            truth(),
            prediction((0.3, 0.5), (0.5, 0.7)),
            1,
            0.9905,
        ),
        (
            # Distribution 1: ____##____
            # Distribution 2: _____##___
            # Guessing: 0.5
            truth(),
            prediction((0.4, 0.6), (0.5, 0.7)),
            1,
            0.5634,
        ),
        (
            # Distribution 1: ___##_____
            # Distribution 2: ____##____
            # Guessing: 0.5
            truth(),
            prediction((0.3, 0.5), (0.4, 0.6)),
            1,
            0.5625,
        ),
        (
            # Distribution 1: ____##____
            # Distribution 2: ____##____
            # Guessing: 0.5
            truth(),
            prediction((0.4, 0.6), (0.4, 0.6)),
            1,
            0.1371,
        ),
        (
            # Distribution 1: _____##___
            # Distribution 2: ___##_____
            # Guessing: 0.5
            truth(),
            prediction((0.5, 0.7), (0.3, 0.5)),
            1,
            0.9870,
        ),
        (
            # Distribution 1: ___###____
            # Distribution 2: _____###__
            # Guessing: 0.5
            truth(),
            prediction((0.3, 0.6), (0.5, 0.8)),
            1,
            0.7074,
        ),
        (
            # Distribution 1: __####____
            # Distribution 2: _____####_
            # Guessing: 0.5
            truth(),
            prediction((0.2, 0.6), (0.5, 0.9)),
            1,
            0.7655,
        ),
        (
            # Distribution 1: _#####____
            # Distribution 2: _____#####
            # Guessing: 0.5
            truth(),
            prediction((0.1, 0.6), (0.5, 1.0)),
            1,
            0.8092,
        ),
        (
            # Distribution 1: _#####____
            # Distribution 2: _____#####
            # Guessing: 0.5
            truth(),
            prediction((0.4, 0.6), (0.5, 0.7)),
            1,
            0.5634,
        ),
        (
            # Distribution 1: ###########
            # Distribution 2: ###########
            # Guessing: 0.5
            truth(),
            prediction((0.0, 1.0), (0.0, 1.0)),
            1.0,
            0.1374,
        ),
        (
            # Distribution 1: ###########
            # Distribution 2: ###########
            # Guessing: 0.33
            truth(),
            prediction((0.0, 1.0), (0.0, 1.0)),
            1 / 2.0,
            0.1374,
        ),
        (
            # Distribution 1: ###########
            # Distribution 2: ###########
            # Guessing: 0.66
            truth(),
            prediction((0.0, 1.0), (0.0, 1.0)),
            2.0 / 1,
            0.1374,
        ),
        (
            # Distribution 1: ##________
            # Distribution 2: ________##
            # Guessing: 0.5
            # Only 50 samples
            truth(samples=50),
            prediction((0.0, 0.2), (0.8, 1.0), samples=50),
            1,
            1,
        ),
    ]
)
def test_linkability(truth, prediction, omega, expected):
    linkability = audmetric.linkability(truth, prediction, omega=omega)
    np.testing.assert_allclose(linkability, expected, rtol=0.15)
