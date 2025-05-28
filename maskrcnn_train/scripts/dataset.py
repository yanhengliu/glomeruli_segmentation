import os
from pathlib import Path
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.ops import masks_to_boxes
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F
from skimage import measure

# assume this file lives in …/glomeruli_segmentation/maskrcnn_train/scripts
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASETS_DIR = PROJECT_ROOT / "datasets"


class GlomeruliDataset(Dataset):
    """Load images and binary masks, convert them into targets for Mask-R-CNN."""
    def __init__(self, root_dir: str, transforms=None):
        """
        Args:
            root_dir: either a split name ('train','val','test') or a full path.
            transforms: optional torchvision v2 transform that will receive (image, target).
        """
        # support passing just the split name
        if root_dir in ("train", "val", "test"):
            self.root_dir = DATASETS_DIR / root_dir
        else:
            self.root_dir = Path(root_dir)

        self.transforms = transforms
        self.images_dir = self.root_dir / "images"
        self.masks_dir  = self.root_dir / "masks"

        self.image_files = sorted(os.listdir(self.images_dir))
        self.mask_files  = sorted(os.listdir(self.masks_dir))
        assert len(self.image_files) == len(self.mask_files), "image / mask count mismatch"

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx: int):
        # read image and mask as tensors
        img_path = self.images_dir / self.image_files[idx]
        msk_path = self.masks_dir  / self.mask_files[idx]
        image   = read_image(str(img_path))  # uint8, shape [C, H, W]
        mask    = read_image(str(msk_path))  # uint8, shape [1, H, W]

        # label connected foreground regions as separate instances
        labelled = measure.label(mask.numpy().squeeze(), background=0)
        labelled = torch.as_tensor(labelled, dtype=torch.int64)  # [H, W]
        obj_ids = torch.unique(labelled)
        obj_ids = obj_ids[obj_ids != 0]                          # drop background id 0
        masks = (labelled[None] == obj_ids[:, None, None]).to(torch.uint8)  # [N, H, W]

        # boxes, areas, labels, iscrowd
        boxes   = masks_to_boxes(masks)                            # [N, 4]
        labels  = torch.ones((boxes.shape[0],), dtype=torch.int64)
        areas   = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros_like(labels)

        # remove degenerate boxes with zero width or height
        valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        boxes   = boxes[valid]
        masks   = masks[valid]
        labels  = labels[valid]
        areas   = areas[valid]
        iscrowd = iscrowd[valid]

        # wrap into tv_tensors
        image_tv  = tv_tensors.Image(image)
        boxes_tv  = tv_tensors.BoundingBoxes(
                        boxes,
                        format="XYXY",
                        canvas_size=F.get_size(image_tv)
                    )
        masks_tv  = tv_tensors.Mask(masks)

        target = {
            "boxes":     boxes_tv,
            "labels":    labels,
            "masks":     masks_tv,
            "image_id":  torch.tensor([idx]),
            "area":      areas,
            "iscrowd":   iscrowd,
        }

        # optional transforms
        if self.transforms is not None:
            image_tv, target = self.transforms(image_tv, target)

        return image_tv, target
