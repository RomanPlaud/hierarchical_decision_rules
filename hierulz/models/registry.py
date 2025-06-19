import json 
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

# Dictionary of supported models
MODEL_REGISTRY = {
    'alexnet': (alexnet, AlexNet_Weights),
    'convnext_tiny': (convnext_tiny, ConvNeXt_Tiny_Weights),
    'densenet121': (densenet121, DenseNet121_Weights),
    'efficientnet_v2_s': (efficientnet_v2_s, EfficientNet_V2_S_Weights),
    'inception_v3': (inception_v3, Inception_V3_Weights),
    'resnet18': (resnet18, ResNet18_Weights),
    'swin_v2_t': (swin_v2_t, Swin_V2_T_Weights),
    'vgg11': (vgg11, VGG11_Weights),
    'vit_b_16': (vit_b_16, ViT_B_16_Weights),
}
MODEL_REGISTRY_configs = {
    'alexnet': 'configs/models/tieredimagenet/alexnet.json',
    'convnext_tiny': 'configs/models/tieredimagenet/convnext_tiny.json',
    'densenet121': 'configs/models/tieredimagenet/densenet121.json',
    'efficientnet_v2_s': 'configs/models/tieredimagenet/efficientnet_v2_s.json',
    'inception_v3': 'configs/models/tieredimagenet/inception_v3.json',
    'resnet18': 'configs/models/tieredimagenet/resnet18.json',
    'swin_v2_t': 'configs/models/tieredimagenet/swin_v2_t.json',
    'vgg11': 'configs/models/tieredimagenet/vgg11.json',
    'vit_b_16': 'configs/models/tieredimagenet/vit_b_16.json',
}


def get_pretrained_model(model_name: str):
    """
    Return a model constructor and weights given a model name.
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{model_name}'. Available: {list(MODEL_REGISTRY)}")

    constructor, weights_cls = MODEL_REGISTRY[model_name]
    return constructor, weights_cls

def get_model_config(model_name: str):
    """
    Return the path to the model config file given a model name.
    """
    if model_name not in MODEL_REGISTRY_configs:
        raise ValueError(f"Unknown model '{model_name}'. Available: {list(MODEL_REGISTRY_configs)}")
    else : 
        path_config = MODEL_REGISTRY_configs[model_name]
        # open the config file and return it
        with open(path_config, 'r') as f:
            config = json.load(f)
        return config