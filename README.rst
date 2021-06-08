=========
audmetric
=========

|tests| |license|

**audmetric** includes several equations
to estimate the performance of a machine learning prediction algorithm.

Some of the metrics are also available in sklearn_,
but we wanted to have a package
which depends only on :mod:`numpy`.
For those metrics
we included tests that the results are identical to sklearn_.


.. _sklearn: https://scikit-learn.org/stable/


.. badges images and links:
.. |tests| image:: https://github.com/audeering/audmetric/workflows/Test/badge.svg
    :target: https://github.com/audeering/audmetric/actions?query=workflow%3ATest
    :alt: Test status
.. |license| image:: https://img.shields.io/badge/license-MIT-green.svg
    :target: https://github.com/audeering/audmetric/blob/master/LICENSE
    :alt: audmetric's MIT license
