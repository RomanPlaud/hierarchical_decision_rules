import torch
import torch.nn as nn
import torch.nn.functional as F


class HierarchicalModel(nn.Module):
    """
    A class to wrap a base model for hierarchical tasks, allowing post hoc
    modification of its classifier layer (e.g., pruning outputs).
    """

    def __init__(self, model):
        """
        Initialize the HierarchicalModel.

        Args:
            model (nn.Module): The base model to wrap. It should contain a classifier layer.
        """
        super(HierarchicalModel, self).__init__()
        self.model = model
        self.classifier_layer = self._find_last_linear_layer()
        

    def _find_last_linear_layer(self) -> nn.Linear:
        """
        Find the last nn.Linear layer in the model.
        Returns:
            nn.Linear: The last linear layer used for classification.
        """
        linear_layers = [m for m in self.model.modules() if isinstance(m, nn.Linear)]
        return linear_layers[-1] if linear_layers else None

    def prune_classifier(self, idx_mapping: dict):
        """
        Prune the output classifier layer to keep only weights corresponding to a subset of classes.

        Args:
            idx_mapping (dict): A mapping from class keys to indices to retain.
                                E.g., {'cat': 5, 'dog': 12, ...}
        """
        if self.classifier_layer is None:
            raise RuntimeError("Classifier layer not found or already pruned.")

        indices = [idx_mapping[key] for key in idx_mapping]
        indices = torch.tensor(indices, dtype=torch.long)

        # Get original weights and bias
        old_weight = self.classifier_layer.weight.data
        old_bias = self.classifier_layer.bias.data

        # Prune
        new_weight = old_weight[indices]
        new_bias = old_bias[indices]

        # Replace parameters
        self.classifier_layer.weight = nn.Parameter(new_weight)
        self.classifier_layer.bias = nn.Parameter(new_bias)
        self.classifier_layer.out_features = len(indices)

        print(f"Classifier layer pruned to {len(indices)} classes.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward pass through the model.
        Args:
            x (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: Output logits.
        """
        logits = self.model(x)
        # apply softmax to logits
        probas = F.softmax(logits, dim=1)
        return probas

