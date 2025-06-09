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


def get_pretrained_model(model_name: str):
    """
    Return a model constructor and weights given a model name.
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{model_name}'. Available: {list(MODEL_REGISTRY)}")

    constructor, weights_cls = MODEL_REGISTRY[model_name]
    weights = weights_cls.IMAGENET1K_V1
    model = constructor(weights=weights)
    return model
