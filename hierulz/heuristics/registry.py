import json
from pathlib import Path
from dataclasses import dataclass
from typing import Callable, Dict, Type

from .confidence_threshold import ConfidenceThreshold
from .crm_bm import CRM_BM
from .expected_information import ExpectedInformation
from .hie import HiE
from .information_threshold import InformationThreshold
from .plurality import Plurality
from .top_down import TopDown
from hierulz.metrics import Accuracy

from hierulz.hierarchy import load_hierarchy

@dataclass(frozen=True)
class HeuristicInfo:
    constructor: Type
    config_path: Path

HEURISTIC_REGISTRY: Dict[str, HeuristicInfo] = {
    'Thresholding 0.5': HeuristicInfo(ConfidenceThreshold, Path('configs/heuristics/confidence_threshold.json')),
    '(Karthik et al., 2021)': HeuristicInfo(CRM_BM, Path('configs/heuristics/crm_bm.json')),
    'Exp. Information': HeuristicInfo(ExpectedInformation, Path('configs/heuristics/expected_information.json')),
    'Hie-Self (Jain et al., 2023)': HeuristicInfo(HiE, Path('configs/heuristics/hie.json')),
    'information_threshold': HeuristicInfo(InformationThreshold, Path('configs/heuristics/information_threshold.json')),
    'Plurality': HeuristicInfo(Plurality, Path('configs/heuristics/plurality.json')),
    'Top-down argmax': HeuristicInfo(TopDown, Path('configs/heuristics/top_down.json')),
    'Argmax leaves': HeuristicInfo(Accuracy, Path('configs/heuristics/argmax_leaves.json'))
}

def load_heuristic(heuristic_name: str, dataset_name: str, **kwargs) -> Callable:
    """
    Returns an instance of the specified heuristic, initialized with kwargs.
    """
    hierarchy = load_hierarchy(dataset_name)

    if heuristic_name not in HEURISTIC_REGISTRY:
        raise ValueError(f"Unknown heuristic: '{heuristic_name}'. Available heuristics: {list(HEURISTIC_REGISTRY.keys())}")
    
    return HEURISTIC_REGISTRY[heuristic_name].constructor(hierarchy, **kwargs)