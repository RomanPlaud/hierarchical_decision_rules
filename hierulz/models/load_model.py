from .load_pretrained_model import load_pretrained_model
# from .load_finetuned_model import load_finetuned_model
from .model import HierarchicalModel


def load_model(config_model):
    """
    Load a model based on the configuration provided.

    Args:
        config_model (dict): Configuration dictionary containing model parameters.

    Returns:
        Model: The loaded model.
    """
    if config_model['pretrained']:
        model = load_pretrained_model(config_model['model_name'])
        model = HierarchicalModel(model)
        model.prune_classifier(config_model[config_model['idx_mapping']])
    else:
        # model = load_finetuned_model(config_model['model_name'], config_model['dataset_name'])
        # model = HierarchicalModel(model)
        raise ValueError("Not supported yet")

    return model