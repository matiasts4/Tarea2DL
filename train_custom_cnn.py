"""
train_custom_cnn.py
-------------------
Entrenamiento desde cero del detector convolucional propio (CustomCNNDetector).
- Pesos inicializados aleatoriamente con Kaiming He (sin preentrenamiento).
- Utiliza FjordFishDataset y el evaluador oficial COCO de models.utils.
- Asignación multiescala por área en FPN (P3: objetos pequeños, P4: medianos, P5: grandes).
- Coherencia exacta entre entrenamiento (SmoothL1 + BCEWithLogits) e inferencia.
"""

import os
import time
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.custom_detector import CustomCNNDetector
from models.utils import (
    NUM_CLASSES,
    set_seed,
    FjordFishDataset,
    evaluate_detections_coco,
)

SEED = 42
IMG_SIZE = 512
BATCH_SIZE = 16
NUM_EPOCHS = 20
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def collate_fn(batch):
    imgs = torch.stack([item[0] for item in batch])
    boxes = [item[1] for item in batch]
    labels = [item[2] for item in batch]
    return imgs, boxes, labels


def compute_loss(outputs, targets_boxes, targets_labels, device=DEVICE):
    """
    Cálculo de pérdida multiescala con asignación por escala en la pirámide FPN:
      - Nivel P3 (stride 8):  peces pequeños (escala sqrt(w*h) < 0.12)
      - Nivel P4 (stride 16): peces medianos (0.12 <= sqrt(w*h) < 0.30)
      - Nivel P5 (stride 32): peces grandes  (sqrt(w*h) >= 0.30)
    """
    total_reg_loss = 0.0
    total_obj_loss = 0.0
    total_cls_loss = 0.0

    bce_loss = nn.BCEWithLogitsLoss(reduction='none')
    smooth_l1 = nn.SmoothL1Loss(reduction='none')
    scale_ranges = [(0.0, 0.12), (0.12, 0.30), (0.30, 2.0)]

    for lvl_idx, (reg, obj, cls) in enumerate(outputs):
        B, _, H, W = reg.shape
        s_min, s_max = scale_ranges[lvl_idx]

        obj_target = torch.zeros((B, 1, H, W), device=device)
        reg_mask = torch.zeros((B, 1, H, W), dtype=torch.bool, device=device)
        reg_target = torch.zeros((B, 4, H, W), device=device)
        cls_target = torch.zeros((B, NUM_CLASSES, H, W), device=device)

        for b in range(B):
            b_boxes = targets_boxes[b].to(device)
            b_lbls = targets_labels[b].to(device)
            if len(b_boxes) == 0:
                continue

            for box, lbl in zip(b_boxes, b_lbls):
                cx, cy, bw, bh = box[0].item(), box[1].item(), box[2].item(), box[3].item()
                scale = (bw * bh) ** 0.5
                if not (s_min <= scale < s_max):
                    continue

                gx = min(int(cx * W), W - 1)
                gy = min(int(cy * H), H - 1)

                obj_target[b, 0, gy, gx] = 1.0
                reg_mask[b, 0, gy, gx] = True
                reg_target[b, :, gy, gx] = torch.tensor(
                    [cx * W - gx, cy * H - gy, bw, bh], device=device
                )
                cls_target[b, lbl, gy, gx] = 1.0

        pos_weight = torch.tensor([5.0], device=device)
        bce_weighted = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        obj_loss = bce_weighted(obj, obj_target)

        if reg_mask.sum() > 0:
            reg_pred = torch.cat([
                torch.clamp(reg[:, 0:2], 0.0, 1.0),
                torch.clamp(reg[:, 2:4], 0.01, 1.0)
            ], dim=1)
            reg_diff = smooth_l1(reg_pred, reg_target)
            reg_loss = reg_diff[reg_mask.repeat(1, 4, 1, 1)].mean()

            cls_diff = bce_loss(cls, cls_target)
            cls_loss = cls_diff[reg_mask.repeat(1, NUM_CLASSES, 1, 1)].mean()
        else:
            reg_loss = torch.tensor(0.0, device=device)
            cls_loss = torch.tensor(0.0, device=device)

        total_reg_loss += reg_loss
        total_obj_loss += obj_loss
        total_cls_loss += cls_loss

    return 2.0 * total_reg_loss + 1.0 * total_obj_loss + 1.0 * total_cls_loss


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    running_loss = 0.0
    for imgs, boxes, labels in loader:
        imgs = imgs.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = compute_loss(outputs, boxes, labels, device=device)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
    return running_loss / len(loader.dataset)


@torch.no_grad()
def evaluate_val(model, loader, device):
    """Evalúa pérdida y métricas reales COCO sobre el conjunto de validación."""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    for imgs, boxes_list, labels_list in loader:
        imgs = imgs.to(device)
        outputs = model(imgs)
        loss = compute_loss(outputs, boxes_list, labels_list, device=device)
        running_loss += loss.item() * imgs.size(0)

        preds = model.predict(imgs, conf_thresh=0.10, iou_thresh=0.45)
        for b in range(imgs.size(0)):
            all_preds.append({
                'boxes': preds[b]['boxes'].cpu().numpy(),
                'scores': preds[b]['scores'].cpu().numpy(),
                'labels': preds[b]['labels'].cpu().numpy()
            })
            gt_norm = boxes_list[b].numpy()
            if len(gt_norm) > 0:
                x1 = (gt_norm[:, 0] - gt_norm[:, 2] / 2.0).clip(0.0, 1.0)
                y1 = (gt_norm[:, 1] - gt_norm[:, 3] / 2.0).clip(0.0, 1.0)
                x2 = (gt_norm[:, 0] + gt_norm[:, 2] / 2.0).clip(0.0, 1.0)
                y2 = (gt_norm[:, 1] + gt_norm[:, 3] / 2.0).clip(0.0, 1.0)
                gt_xyxy = torch.tensor([x1, y1, x2, y2]).T.numpy()
            else:
                gt_xyxy = gt_norm
            all_targets.append({
                'boxes': gt_xyxy,
                'labels': labels_list[b].numpy()
            })

    val_loss = running_loss / len(loader.dataset)
    metrics = evaluate_detections_coco(all_preds, all_targets, num_classes=NUM_CLASSES, op_conf_thresh=0.25)
    return val_loss, metrics['Precision'], metrics['Recall'], metrics['F1-Score'], metrics['mAP@50']


def main():
    set_seed(SEED)
    print("=== Entrenamiento: Modelo 1 - Custom CNN Detector (Desde Cero) ===")
    print(f"Dispositivo: {DEVICE}")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_dir_img = os.path.join(base_dir, 'data', 'splits', 'images', 'train')
    train_dir_lbl = os.path.join(base_dir, 'data', 'splits', 'labels', 'train')
    val_dir_img = os.path.join(base_dir, 'data', 'splits', 'images', 'val')
    val_dir_lbl = os.path.join(base_dir, 'data', 'splits', 'labels', 'val')

    train_dataset = FjordFishDataset(train_dir_img, train_dir_lbl, img_size=IMG_SIZE, augment=True, mode='custom_cnn')
    val_dataset = FjordFishDataset(val_dir_img, val_dir_lbl, img_size=IMG_SIZE, augment=False, mode='custom_cnn')

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, collate_fn=collate_fn, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, collate_fn=collate_fn, pin_memory=True)

    model = CustomCNNDetector(num_classes=NUM_CLASSES, base_channels=32).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parámetros del modelo propio: {total_params:,} ({total_params/1e6:.2f}M)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-5)

    out_model_dir = os.path.join(base_dir, 'results', 'models')
    out_metric_dir = os.path.join(base_dir, 'results', 'metrics')
    os.makedirs(out_model_dir, exist_ok=True)
    os.makedirs(out_metric_dir, exist_ok=True)

    best_val_loss = float('inf')
    history = []

    start_total_time = time.time()
    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, DEVICE)
        val_loss, val_prec, val_rec, val_f1, val_map50 = evaluate_val(model, val_loader, DEVICE)
        scheduler.step()
        epoch_time = time.time() - t0

        print(
            f"Epoch [{epoch:02d}/{NUM_EPOCHS:02d}] ({epoch_time:.1f}s) - "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"mAP@50: {val_map50:.4f} | P: {val_prec:.4f} | R: {val_rec:.4f} | F1: {val_f1:.4f}"
        )

        history.append({
            'epoch': epoch,
            'train_loss': float(train_loss),
            'val_loss': float(val_loss),
            'map50': float(val_map50),
            'precision': float(val_prec),
            'recall': float(val_rec),
            'f1': float(val_f1),
            'epoch_time': float(epoch_time)
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(out_model_dir, 'custom_cnn_best.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'params': total_params
            }, save_path)

    total_train_time = time.time() - start_total_time
    history_file = os.path.join(out_metric_dir, 'custom_cnn_history.json')
    with open(history_file, 'w', encoding='utf-8') as f:
        json.dump({
            'model_name': 'Custom CNN Detector (Desde Cero)',
            'total_params': total_params,
            'total_time_seconds': total_train_time,
            'epochs': history
        }, f, indent=2)
    print(f"Historial guardado en: {history_file}")


if __name__ == '__main__':
    main()
