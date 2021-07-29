from audmetric import utils
from audmetric.core.api import (
    accuracy,
    concordance_cc,
    confusion_matrix,
    detection_error_tradeoff,
    edit_distance,
    equal_error_rate,
    event_error_rate,
    fscore_per_class,
    mean_absolute_error,
    mean_squared_error,
    pearson_cc,
    precision_per_class,
    recall_per_class,
    unweighted_average_bias,
    unweighted_average_fscore,
    unweighted_average_recall,
    unweighted_average_precision,
    weighted_confusion_error,
    word_error_rate
)


__all__ = []


# Dynamically get the version of the installed module
try:
    import pkg_resources
    __version__ = pkg_resources.get_distribution(__name__).version
except Exception:  # pragma: no cover
    pkg_resources = None  # pragma: no cover
finally:
    del pkg_resources
