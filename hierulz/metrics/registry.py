from .accuracy import Accuracy
from .hamming_loss import HammingLoss
from .hf_beta_score import hFBetaScore
from .leaf2leaf_metric import Leaf2LeafMetric
from .mistake_severity import MistakeSeverity
from .node2leaf_metric import Node2LeafMetric

METRIC_REGISTRY = {
    'accuracy': Accuracy,
    'hamming': HammingLoss,
    'hfbeta': hFBetaScore,
    'mistake_severity': MistakeSeverity,
    'node2leaf': Node2LeafMetric,
    'leaf2leaf': Leaf2LeafMetric,
}

def get_metric(metric_name, **kwargs):
    """Get metric class from registry."""
    if metric_name not in METRIC_REGISTRY:
        raise ValueError(f"Unknown metric: {metric_name}")
    return METRIC_REGISTRY[metric_name](**kwargs)