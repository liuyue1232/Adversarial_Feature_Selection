"""
.. module:: MetricFNRatFPR
   :synopsis: Performance Metric: False Negative Rate @ False Positive Rate

.. moduleauthor:: Marco Melis <marco.melis@unica.it>

"""
from secml.array import CArray
from secml.ml.peval.metrics import CMetric
from secml.ml.peval.metrics import CRoc


class CMetricFNRatFPR(CMetric):
    """Performance evaluation metric: False Negative Rate @ False Positive Rate.

    The metric uses:
     - y_true (true ground labels)
     - score (estimated target values)

    Parameters
    ----------
    fpr : float
        Desired False Positive Rate  in the interval [0,1]. Default 0.01 (1%)

    Attributes
    ----------
    class_type : 'fnr-at-fpr'

    Notes
    -----
    This implementation is restricted to the binary classification task.

    Examples
    --------
    >>> from secml.ml.peval.metrics import CMetricFNRatFPR
    >>> from secml.array import CArray

    >>> peval = CMetricFNRatFPR(fpr=0.5)
    >>> peval.performance_score(CArray([0, 1, 0, 0]), score=CArray([0, 0, 0, 0]))
    0.5

    """
    __class_type = 'fnr-at-fpr'
    best_value = 1.0

    def __init__(self, fpr=0.01):
        self.fpr = float(fpr)

    def _performance_score(self, y_true, score):
        """Computes the False Negative Rate at given False Positive Rate.

        Parameters
        ----------
        y_true : CArray
            Flat array with true binary labels in range
            {0, 1} or {-1, 1} for each pattern.
        score : CArray
            Flat array with target scores for each pattern, can either be
            probability estimates of the positive class or confidence values.

        Returns
        -------
        metric : float
            Returns metric value as float.

        Notes
        -----
        This implementation is restricted to the binary classification task.

        """
        return 1 - CArray(self.fpr).interp(
            *CRoc().compute(y_true, score)[0:2]).item()
