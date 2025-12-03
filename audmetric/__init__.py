from audmetric import utils
from audmetric.core.api import accuracy
from audmetric.core.api import concordance_cc
from audmetric.core.api import confusion_matrix
from audmetric.core.api import detection_error_tradeoff
from audmetric.core.api import diarization_error_rate
from audmetric.core.api import edit_distance
from audmetric.core.api import equal_error_rate
from audmetric.core.api import event_confusion_matrix
from audmetric.core.api import event_error_rate
from audmetric.core.api import event_fscore_per_class
from audmetric.core.api import event_precision_per_class
from audmetric.core.api import event_recall_per_class
from audmetric.core.api import event_unweighted_average_fscore
from audmetric.core.api import fscore_per_class
from audmetric.core.api import identification_error_rate
from audmetric.core.api import linkability
from audmetric.core.api import mean_absolute_error
from audmetric.core.api import mean_squared_error
from audmetric.core.api import pearson_cc
from audmetric.core.api import precision_per_class
from audmetric.core.api import recall_per_class
from audmetric.core.api import unweighted_average_bias
from audmetric.core.api import unweighted_average_fscore
from audmetric.core.api import unweighted_average_precision
from audmetric.core.api import unweighted_average_recall
from audmetric.core.api import weighted_confusion_error
from audmetric.core.api import word_error_rate


__all__ = []


# Dynamically get the version of the installed module
try:
    import importlib.metadata

    __version__ = importlib.metadata.version(__name__)
except Exception:  # pragma: no cover
    importlib = None  # pragma: no cover
finally:
    del importlib
