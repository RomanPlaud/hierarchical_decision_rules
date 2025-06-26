import pickle as pkl

from .load_pretrained_model import load_pretrained_model
from .load_finetuned_model import load_finetuned_model
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
        path_idx_mapping = config_model.get('idx_mapping', None)
        dict_idx_mapping = pkl.load(open(path_idx_mapping, 'rb')) if path_idx_mapping else None
        if dict_idx_mapping is not None:
            model.prune_classifier(dict_idx_mapping)
        else: 
            print("No idx_mapping provided, using the full classifier layer.")
    else:
        model = load_finetuned_model(config_model, config_model['dataset_name'])
        model = HierarchicalModel(model)
        raise ValueError("Not supported yet")

    return model