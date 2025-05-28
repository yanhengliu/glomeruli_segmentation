import os

def collate_fn(batch):
    """
    Custom collate function that transforms a batch of (image, target)
    pairs into a tuple of lists: (images_list, targets_list).
    """
    return tuple(zip(*batch))

def mkdir(path):
    """
    Create a directory (and any necessary parent directories) if it does not already exist.
    """
    os.makedirs(path, exist_ok=True)
