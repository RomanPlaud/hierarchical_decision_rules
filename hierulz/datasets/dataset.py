from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torchvision import transforms

class HierarchicalDataset(Dataset):
    """
    A general-purpose dataset wrapper for hierarchical image datasets.
    Applies optional Gaussian blur preprocessing to each image.

    Args:
        root (str): Path to the dataset root directory.
        transform (callable, optional): Optional transform to be applied on an image.
        blurr_level (float, optional): Sigma value for Gaussian blur. If None, no blur is applied.
    """
    def __init__(self, root, transform=None, blurr_level=None, kernel_size=61):
        self.transform = transform
        self.dataset = ImageFolder(root, transform=transform)
        if blurr_level==0 or blurr_level is None:
            self.blurr = None
        else:
            self.blurr = (
                transforms.GaussianBlur(kernel_size=kernel_size, sigma=blurr_level)
                if blurr_level is not None else None
            )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        if self.blurr:
            img = self.blurr(img)
        return img, label
