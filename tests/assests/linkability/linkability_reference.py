import subprocess
import tempfile

import numpy as np

import audeer
import audmetric


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


test_cases = [
    # truth, prediction, omega
    (
        truth(),
        prediction((0.0, 0.2), (0.8, 1.0)),
        1,
    ),
    (
        truth(),
        prediction((0.0, 0.2), (0.8, 1.0)),
        1 / 9.0,
    ),
    (
        truth(),
        prediction((0.1, 0.3), (0.7, 0.9)),
        1,
    ),
    (
        truth(),
        prediction((0.2, 0.4), (0.6, 0.8)),
        1,
    ),
    (
        truth(),
        prediction((0.3, 0.5), (0.5, 0.7)),
        1,
    ),
    (
        truth(),
        prediction((0.4, 0.6), (0.5, 0.7)),
        1,
    ),
    (
        truth(),
        prediction((0.3, 0.5), (0.4, 0.6)),
        1,
    ),
    (
        truth(),
        prediction((0.4, 0.6), (0.4, 0.6)),
        1,
    ),
    (
        truth(),
        prediction((0.5, 0.7), (0.3, 0.5)),
        1,
    ),
    (
        truth(),
        prediction((0.3, 0.6), (0.5, 0.8)),
        1,
    ),
    (
        truth(),
        prediction((0.2, 0.6), (0.5, 0.9)),
        1,
    ),
    (
        truth(),
        prediction((0.1, 0.6), (0.5, 1.0)),
        1,
    ),
    (
        truth(),
        prediction((0.4, 0.6), (0.5, 0.7)),
        1,
    ),
    (
        truth(),
        prediction((0.0, 1.0), (0.0, 1.0)),
        1.0,
    ),
    (
        truth(),
        prediction((0.0, 1.0), (0.0, 1.0)),
        1 / 2.0,
    ),
    (
        truth(),
        prediction((0.0, 1.0), (0.0, 1.0)),
        2.0 / 1,
    ),
]

results = []
for n, (truth, prediction, omega) in enumerate(test_cases):
    mated_scores, nonmated_scores = audmetric.core.api._matching_scores(
        truth,
        prediction,
    )
    with tempfile.TemporaryDirectory() as tmp:
        score_file = audeer.path(tmp, 'score')
        with open(score_file, 'w') as fp:
            for score in mated_scores:
                fp.write(f'{score} 1\n')
            for score in nonmated_scores:
                fp.write(f'{score} 0\n')
        shell_command = [
            'python',
            'anonymization_metrics/compute_metrics.py',
            '-s',
            f'{score_file}',
            '--omega',
            f'{omega}',
        ]
        out = subprocess.check_output(
            shell_command,
            stderr=subprocess.STDOUT
        )
        # The above Python command returns a string containing:
        # id,matedMean,nonMatedMean,matedStd,nonMatedStd,linkability,cllr,cmin,eer
        # which means we can access the linkability by -4 from the end
        linkability = out.split()[0].decode("utf-8").split(',')[-4]
        results.append((omega, linkability))

# Store results
with open('results.txt', 'w') as fp:
    fp.write('testcase,omega,linkability\n')
    for n, (omega, linkability) in enumerate(results):
        fp.write(f'{n},{omega},{linkability}\n')
