import json
from pathlib import Path
from dataclasses import dataclass
from typing import Callable, Dict, Type

from .accuracy import Accuracy
from .hamming_loss import HammingLoss
from .hf_beta_score import hFBetaScore
from .leaf2leaf_metric import Leaf2LeafMetric
from .mistake_severity import MistakeSeverity
from .node2leaf_metric import Node2LeafMetric

from hierulz.hierarchy import load_hierarchy


@dataclass(frozen=True)
class MetricInfo:
    constructor: Type
    config_path: Path


# Unified metric registry
METRIC_REGISTRY: Dict[str, MetricInfo] = {
    'Accuracy': MetricInfo(Accuracy, Path('configs/metrics/accuracy.json')),
    'Hamming Loss': MetricInfo(HammingLoss, Path('configs/metrics/hamming.json')),
    'hF_ß': MetricInfo(hFBetaScore, Path('configs/metrics/hfbeta.json')),
    'Mistake Severity': MetricInfo(MistakeSeverity, Path('configs/metrics/mistake_severity.json')),
    'Wu-Palmer': MetricInfo(Node2LeafMetric, Path('configs/metrics/wu_palmer.json')),
    'Zhao': MetricInfo(Leaf2LeafMetric, Path('configs/metrics/zhao.json')),
}


def load_metric(metric_name: str, dataset_name: str, **kwargs) -> Callable:
    """
    Returns an instance of the specified metric, initialized with kwargs.
    """
    hierarchy = load_hierarchy(dataset_name)

    if metric_name not in METRIC_REGISTRY:
        raise ValueError(f"Unknown metric: '{metric_name}'. Available metrics: {list(METRIC_REGISTRY.keys())}")
    
    return METRIC_REGISTRY[metric_name].constructor(hierarchy, **kwargs)


def get_metric_config(metric_name: str, dataset_name : str) -> dict:
    """
    Loads and returns the JSON configuration for the specified metric.
    """
    if metric_name not in METRIC_REGISTRY:
        raise ValueError(f"Unknown metric: '{metric_name}'. Available metrics: {list(METRIC_REGISTRY.keys())}")
    
    config_path = METRIC_REGISTRY[metric_name].config_path

    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found for metric '{metric_name}' at: {config_path}")
    
    with open(config_path, 'r') as f:
        return json.load(f)[dataset_name]