=========
audmetric
=========

|tests| |coverage| |docs| |python-versions| |license|

**audmetric** includes several equations
to estimate the performance of a machine learning prediction algorithm.

Some of the metrics are also available in sklearn_,
but we wanted to have a package
which depends only on numpy_.
For those metrics
we included tests that the results are identical to sklearn_.


.. _numpy: https://numpy.org/
.. _sklearn: https://scikit-learn.org/stable/


.. badges images and links:
.. |tests| image:: https://github.com/audeering/audmetric/workflows/Test/badge.svg
    :target: https://github.com/audeering/audmetric/actions?query=workflow%3ATest
    :alt: Test status
.. |coverage| image:: https://codecov.io/gh/audeering/audmetric/branch/master/graph/badge.svg?token=wOMLYzFnDO
    :target: https://codecov.io/gh/audeering/audmetric/
    :alt: code coverage
.. |docs| image:: https://img.shields.io/pypi/v/audmetric?label=docs
    :target: https://audeering.github.io/audmetric/
    :alt: audmetric's documentation
.. |license| image:: https://img.shields.io/badge/license-MIT-green.svg
    :target: https://github.com/audeering/audmetric/blob/master/LICENSE
    :alt: audmetric's MIT license
.. |python-versions| image:: https://img.shields.io/pypi/pyversions/audmetric.svg
    :target: https://pypi.org/project/audmetric/
    :alt: audmetric's supported Python versions
