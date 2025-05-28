import os
import datetime
from pathlib import Path
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as T
from PIL import Image
from torch.utils.tensorboard import SummaryWriter  # Import SummaryWriter

from scripts.dataset import GlomeruliDataset
from scripts.model import get_instance_segmentation_model
from scripts.engine import train_one_epoch, evaluate
from scripts import utils

# ------------------------------------------------------------------
# Base directories (all relative to project root)
# ------------------------------------------------------------------
# project root: …/glomeruli_segmentation
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASETS_DIR = PROJECT_ROOT / "datasets"
OUTPUT_BASE  = PROJECT_ROOT / "maskrcnn_train" / "outputs"

# Create a new run folder with timestamp
run_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
run_dir  = OUTPUT_BASE / run_time

# Simplified subfolders under this run
log_dir   = run_dir / "logs"
model_dir = run_dir / "models"
pred_dir  = run_dir / "predictions"

# Make all necessary directories
for d in (log_dir, model_dir, pred_dir):
    utils.mkdir(str(d))

# Initialize TensorBoard SummaryWriter
# Event files will be saved in the 'log_dir'
writer = SummaryWriter(log_dir=str(log_dir))

# Device selection
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using device: {device}")

# ------------------------------------------------------------------
# Transforms
# ------------------------------------------------------------------
def get_transform(train: bool):
    transforms = []
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    transforms.append(T.ToDtype(torch.float32, scale=True))
    transforms.append(T.ToPureTensor())
    return T.Compose(transforms)

# ------------------------------------------------------------------
# Data loaders
# ------------------------------------------------------------------
train_ds = GlomeruliDataset("train", transforms=get_transform(True))
val_ds   = GlomeruliDataset("val",   transforms=get_transform(False))
test_ds  = GlomeruliDataset("test",  transforms=get_transform(False))

train_loader = DataLoader(
    train_ds, batch_size=8, shuffle=True,  collate_fn=utils.collate_fn
)
val_loader   = DataLoader(
    val_ds,   batch_size=8, shuffle=False, collate_fn=utils.collate_fn
)
test_loader  = DataLoader(
    test_ds,  batch_size=1, shuffle=False, collate_fn=utils.collate_fn
)

# ------------------------------------------------------------------
# Build model
# ------------------------------------------------------------------
model = get_instance_segmentation_model(num_classes=2, pretrained=True)
model.to(device)

# Try to add model graph to TensorBoard (optional)
example_images, _ = next(iter(train_loader))
example_images_on_device = [img.to(device) for img in example_images]
if example_images_on_device:
    try:
        writer.add_graph(model, example_images_on_device)
    except Exception as e:
        print(f"Could not add model graph to TensorBoard: {e}")

# ------------------------------------------------------------------
# Training loop
# ------------------------------------------------------------------
NUM_EPOCHS = 250  # Epochs set to 250

# Optimizer: Adam with lr=0.001
optimizer    = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
lr_scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=30, gamma=0.1
)

training_log_file = open(str(log_dir / "training_log.txt"), "a")

print("Starting training with Adam optimizer...")
for epoch in range(1, NUM_EPOCHS + 1):
    avg_loss   = train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=20)
    current_lr = optimizer.param_groups[0]['lr']
    lr_scheduler.step()

    miou = evaluate(model, val_loader, device)

    # Log to text file
    log_message = (
        f"Epoch {epoch}/{NUM_EPOCHS}: "
        f"Train Loss = {avg_loss:.4f}, "
        f"Val mIoU = {miou:.4f}, "
        f"LR = {current_lr:.6f}\n"
    )
    training_log_file.write(log_message)
    training_log_file.flush()
    print(log_message.strip())

    # Log scalars to TensorBoard
    writer.add_scalar('Loss/train_epoch_avg', avg_loss, epoch)
    writer.add_scalar('mIoU/validation',    miou,     epoch)
    writer.add_scalar('Learning_Rate',      current_lr, epoch)

    # Save checkpoint
    ckpt_name = f"epoch_{epoch}.pth"
    ckpt_path = model_dir / ckpt_name
    torch.save({
        'epoch':                 epoch,
        'model_state_dict':      model.state_dict(),
        'optimizer_state_dict':  optimizer.state_dict(),
        'scheduler_state_dict':  lr_scheduler.state_dict(),
        'loss':                  avg_loss,
        'miou':                  miou
    }, str(ckpt_path))
    print(f"Saved checkpoint: {ckpt_path}")

training_log_file.close()
writer.close()
print("Training finished.")

# # ------------------------
# # Generate comparison images: GT vs. Pred
# # This part remains largely the same, using the final trained model.
# # You might want to load the best model checkpoint here instead of the last one.
# # ------------------------
# print("Generating comparison images using the last epoch model...")
# # To load a specific checkpoint:
# # checkpoint = torch.load(os.path.join(model_dir, "epoch_100.pth")) # or your best model
# # model.load_state_dict(checkpoint['model_state_dict'])
# model.eval()

# saved_images_count = 0
# MAX_COMPARISON_IMAGES = 5

# with torch.no_grad():
#     for i, batch_data in enumerate(test_loader):
#         if saved_images_count >= MAX_COMPARISON_IMAGES:
#             break

#         images, targets = batch_data
        
#         if not isinstance(images, list): images = [images]
#         if not isinstance(targets, list): targets = [targets]

#         img_tensor = images[0].to(device)
#         target_dict = targets[0]

#         masks_tv = target_dict["masks"]
#         gt_data = masks_tv.data if hasattr(masks_tv, "data") else masks_tv
        
#         if gt_data.shape[0] == 0:
#             print(f"Skipping image {i} due to no GT masks.")
#             continue

#         gt_union_mask = torch.any(gt_data.bool(), dim=0).cpu().numpy().astype(np.uint8) * 255
#         output = model([img_tensor])[0]
        
#         pred_masks_tensor = output.get("masks")
#         pred_union_mask = np.zeros_like(gt_union_mask, dtype=np.uint8)

#         if isinstance(pred_masks_tensor, torch.Tensor) and pred_masks_tensor.numel() > 0:
#             pred_binary_masks = (pred_masks_tensor[:, 0].cpu().numpy() >= 0.5)
#             if pred_binary_masks.ndim == 3 and pred_binary_masks.shape[0] > 0:
#                  pred_union_mask = (pred_binary_masks.sum(axis=0) > 0).astype(np.uint8) * 255
#             elif pred_binary_masks.ndim == 2 :
#                  pred_union_mask = (pred_binary_masks > 0).astype(np.uint8) * 255

#         if gt_union_mask.shape[0] != pred_union_mask.shape[0] or gt_union_mask.shape[1] != pred_union_mask.shape[1]:
#             print(f"Skipping image {i} due to shape mismatch for concatenation. GT: {gt_union_mask.shape}, Pred: {pred_union_mask.shape}")
#             # Consider resizing pred_union_mask to gt_union_mask.shape if necessary
#             # from skimage.transform import resize
#             # pred_union_mask = resize(pred_union_mask, gt_union_mask.shape, order=0, preserve_range=True, anti_aliasing=False).astype(np.uint8)
#             continue
            
#         comparison_image_np = np.concatenate([gt_union_mask, pred_union_mask], axis=1)
#         comparison_image_pil = Image.fromarray(comparison_image_np, mode='L')
        
#         comparison_image_path = os.path.join(pred_dir, f"compare_test_image_{i}.png")
#         comparison_image_pil.save(comparison_image_path)
#         print(f"Saved comparison image: {comparison_image_path}")
#         saved_images_count += 1

# print(f"Finished generating {saved_images_count} comparison images.")