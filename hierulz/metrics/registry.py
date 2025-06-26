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
    'Accuracy': MetricInfo(Accuracy, Path('configs/metrics/interface/accuracy.json')),
    'Hamming Loss': MetricInfo(HammingLoss, Path('configs/metrics/interface/hamming.json')),
    'hF_ß': MetricInfo(hFBetaScore, Path('configs/metrics/interface/hf1.json')),
    'Mistake Severity': MetricInfo(MistakeSeverity, Path('configs/metrics/interface/mistake_severity.json')),
    'Wu-Palmer': MetricInfo(Node2LeafMetric, Path('configs/metrics/interface/wu_palmer.json')),
    'Zhao': MetricInfo(Node2LeafMetric, Path('configs/metrics/interface/zhao.json')),
}


def load_metric(metric_name: str, dataset_name: str, interface=False) -> Callable:
    """
    Loads metric config and returns an instance of the specified metric.
    """
    if metric_name in METRIC_REGISTRY:
        config_path = METRIC_REGISTRY[metric_name].config_path
    else:
        base_dir = 'interface' if interface else 'experiments'
        config_path = Path(f'configs/metrics/{base_dir}/{metric_name}.json')

    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found for metric '{metric_name}' at: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)[dataset_name]

    hierarchy = load_hierarchy(dataset_name)
    
    return METRIC_REGISTRY[metric_name].constructor(hierarchy, **config.get('kwargs', {}))
