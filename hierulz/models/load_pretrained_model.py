#     model = load_model(args.model, dataset_name=args.dataset, pretrained=args.pretrained, device=device)
from hierulz.models.model import HierarchicalModel
from hierulz.models.registry import get_pretrained_model

def load_pretrained_model(model_name):
    """
    Load a pretrained model from pytorch based on the name

    Args:
        model_name (str): Name of the model to load. Must be one of the supported models in the registry.

    Returns:
        model: The loaded model. (nn.Module)
    """
    model_constructor, weights_cls = get_pretrained_model(model_name)
    weights = weights_cls.IMAGENET1K_V1
    model = model_constructor(weights=weights)
    
    return model