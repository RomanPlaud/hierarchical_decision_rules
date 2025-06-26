import json
from pathlib import Path
from dataclasses import dataclass
from typing import Callable, Type, Tuple

from torchvision.models import (
    alexnet, AlexNet_Weights,
    convnext_tiny, ConvNeXt_Tiny_Weights,
    densenet121, DenseNet121_Weights,
    efficientnet_v2_s, EfficientNet_V2_S_Weights,
    inception_v3, Inception_V3_Weights,
    resnet18, ResNet18_Weights,
    swin_v2_t, Swin_V2_T_Weights,
    vgg11, VGG11_Weights,
    vit_b_16, ViT_B_16_Weights,
)


@dataclass(frozen=True)
class ModelInfo:
    constructor: Callable
    weights: Type
    config_path: Path


# Unified model registry
MODEL_REGISTRY = {
    'alexnet': ModelInfo(alexnet, AlexNet_Weights, Path('configs/models/tieredimagenet/alexnet.json')),
    'convnext_tiny': ModelInfo(convnext_tiny, ConvNeXt_Tiny_Weights, Path('configs/models/tieredimagenet/convnext_tiny.json')),
    'densenet121': ModelInfo(densenet121, DenseNet121_Weights, Path('configs/models/tieredimagenet/densenet121.json')),
    'efficientnet_v2_s': ModelInfo(efficientnet_v2_s, EfficientNet_V2_S_Weights, Path('configs/models/tieredimagenet/efficientnet_v2_s.json')),
    'inception_v3': ModelInfo(inception_v3, Inception_V3_Weights, Path('configs/models/tieredimagenet/inception_v3.json')),
    'resnet18': ModelInfo(resnet18, ResNet18_Weights, Path('configs/models/tieredimagenet/resnet18.json')),
    'swin_v2_t': ModelInfo(swin_v2_t, Swin_V2_T_Weights, Path('configs/models/tieredimagenet/swin_v2_t.json')),
    'vgg11': ModelInfo(vgg11, VGG11_Weights, Path('configs/models/tieredimagenet/vgg11.json')),
    'vit_b_16': ModelInfo(vit_b_16, ViT_B_16_Weights, Path('configs/models/tieredimagenet/vit_b_16.json')),
}


def get_pretrained_model(model_name: str) -> Tuple[Callable, Type]:
    """
    Returns the model constructor and weight class for the specified model name.
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{model_name}'. Available models: {list(MODEL_REGISTRY.keys())}")
    
    model_info = MODEL_REGISTRY[model_name]
    return model_info.constructor, model_info.weights


def get_model_config(model_name: str) -> dict:
    """
    Loads and returns the JSON configuration for the specified model name.
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{model_name}'. Available models: {list(MODEL_REGISTRY.keys())}")
    
    config_path = MODEL_REGISTRY[model_name].config_path

    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found at: {config_path}")
    
    with open(config_path, 'r') as f:
        return json.load(f)
