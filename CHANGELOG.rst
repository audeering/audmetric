Changelog
=========

All notable changes to this project will be documented in this file.

The format is based on `Keep a Changelog`_,
and this project adheres to `Semantic Versioning`_.


Version 1.3.1 (2026/01/07)
--------------------------

* Fixed: ``audmetric.diarization_error_rate()`` for categorical dtype


Version 1.3.0 (2025/12/12)
--------------------------

* Added: ``audmetric.diarization_error_rate()``
* Added: ``audmetric.identification_error_rate()``
* Added: event-based metrics ``audmetric.event_confusion_matrix()``,
  ``audmetric.event_fscore_per_class()``, ``audmetric.event_precision_per_class()``,
  ``audmetric.event_recall_per_class()``, ``audmetric.event_unweighted_average_fscore()``


Version 1.2.3 (2025/11/03)
--------------------------

* Added: support for Python 3.13
* Added: support for Python 3.14
* Removed: support for Python 3.8
* Removed: support for Python 3.9


Version 1.2.2 (2024/06/18)
--------------------------

* Added: support for ``numpy`` 2.0


Version 1.2.1 (2024/02/28)
--------------------------

* Added: ``ignore_nan`` argument
  to ``audmetric.concordance_cc()``
* Added: support for Python 3.12
* Changed: speedup ``audmetric.detection_error_tradeoff()``
* Fixed: avoid deprecation warning for ``pkg_resources``
* Removed: support for Python 3.7


Version 1.2.0 (2023/05/08)
--------------------------

* Added: ``audmetric.linkability()``
* Changed: speedup ``audmetric.concordance_cc()``
  and ``audmetric.pearson_cc()``
  when providing ``truth``
  and/or ``prediction``
  as numpy arrays


Version 1.1.6 (2023/01/03)
--------------------------

* Fixed: require ``sphinx-audeering-theme>=1.2.1``
  to enforce correct theme
  in published docs


Version 1.1.5 (2023/01/03)
--------------------------

* Added: support for Python 3.10
* Added: support for Python 3.11
* Changed: split API documentation into sub-pages
  for each function


Version 1.1.4 (2022/07/05)
--------------------------

* Fixed: accuracy formula in docstring


Version 1.1.3 (2022/02/16)
--------------------------

* Added: reference for CCC formula
* Fixed: support pandas series with datatype ``Int64``


Version 1.1.2 (2022/01/11)
--------------------------

* Fixed: typo in docstring of ``audmetric.mean_absolute_error()``


Version 1.1.1 (2022/01/03)
--------------------------

* Added: Python 3.9 support
* Removed: Python 3.6 support


Version 1.1.0 (2021/07/29)
--------------------------

* Added: ``audmetric.utils.infer_labels()``
* Added: ``audmetric.equal_error_rate()``
* Added: ``audmetric.detection_error_tradeoff()``


Version 1.0.1 (2021/06/10)
--------------------------

* Fixed: broken package due to missing ``__init_.py`` file


Version 1.0.0 (2021/06/09)
--------------------------

* Added: initial public release


.. _Keep a Changelog: https://keepachangelog.com/en/1.0.0/
.. _Semantic Versioning: https://semver.org/spec/v2.0.0.html
