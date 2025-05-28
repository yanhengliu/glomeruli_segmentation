import torch
import time
from tqdm import tqdm


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=50):
    """
    Train the model for a single epoch and print running loss.
    """
    model.train()
    running_loss = 0.0
    total_batches = len(data_loader)
    start_time = time.time()

    for batch_idx, (images, targets) in enumerate(tqdm(data_loader, total=total_batches, desc=f"Epoch {epoch}")):
        # move data to device
        images = [img.to(device) for img in images]
        targets = [{k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                    for k, v in t.items()} for t in targets]

        # forward + backward
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        running_loss += losses.item()
        if (batch_idx + 1) % print_freq == 0:
            avg = running_loss / (batch_idx + 1)
            print(f"Epoch [{epoch}] Batch [{batch_idx+1}/{total_batches}] avg_loss: {avg:.4f}", flush=True)

    epoch_loss = running_loss / total_batches
    elapsed = time.time() - start_time
    print(f"Epoch [{epoch}] finished. Average Loss: {epoch_loss:.4f}, Time: {elapsed:.2f}s", flush=True)
    return epoch_loss


def evaluate(model, data_loader, device):
    """
    Simple evaluator: merge all masks in one image and compute union IoU.
    Samples with 0 GT instances are ignored.
    """
    model.eval()
    total_iou = 0.0
    counted = 0

    with torch.no_grad():
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            targets = [{k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                        for k, v in t.items()} for t in targets]

            outputs = model(images)

            for out, tgt in zip(outputs, targets):
                # ground-truth masks tensor, shape [N_gt, H, W]
                gt = tgt["masks"]
                if hasattr(gt, "data"):          # tv_tensors.Mask
                    gt = gt.data
                if gt.shape[0] == 0:
                    # no GT in this image, skip it
                    continue

                # build union mask for GT
                gt_union = torch.zeros_like(gt[0], dtype=torch.bool)
                for gm in gt:
                    gt_union |= gm.bool()

                # build union mask for predictions
                pred = out.get("masks")
                pred_union = torch.zeros_like(gt_union, dtype=torch.bool)
                if isinstance(pred, torch.Tensor) and pred.numel() > 0:
                    # pred shape [N_pred, 1, H, W]
                    for pm in pred[:, 0]:
                        pred_union |= (pm >= 0.5)

                inter = (pred_union & gt_union).sum().item()
                union = (pred_union | gt_union).sum().item()
                if union > 0:
                    total_iou += inter / union
                    counted += 1

    mean_iou = total_iou / counted if counted else 0.0
    print(f"Validation dataset: mean IoU = {mean_iou:.4f}")
    return mean_iou