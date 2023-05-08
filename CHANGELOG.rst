Changelog
=========

All notable changes to this project will be documented in this file.

The format is based on `Keep a Changelog`_,
and this project adheres to `Semantic Versioning`_.


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
