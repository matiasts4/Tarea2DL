"""
train_faster_rcnn_transfer.py
-----------------------------
Entrenamiento de arquitectura Faster R-CNN (ResNet-50-FPN v2) preentrenada en COCO.
- Inicialización con pesos oficiales FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1.
- Adaptación de la cabeza FastRCNNPredictor para 29 clases (28 marinas + 1 fondo).
- Fase 1 (Backbone Congelado): Se congelan las capas de ResNet-50 y se entrenan RPN, FPN y ROI Head.
- Fase 2 (Fine-Tuning): Se descongelan layer3 y layer4 con tasa de aprendizaje reducida (lr=1e-4).
- Durante la evaluación de pérdida en validación, congela las capas BatchNorm2d en modo .eval()
  para impedir que los lotes de validación contaminen running_mean y running_var.
"""

import os
import time
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from models.utils import NUM_CLASSES, set_seed, FjordFishDataset

SEED = 42
IMG_SIZE = 512
BATCH_SIZE = 4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
FASTER_RCNN_CLASSES = NUM_CLASSES + 1  # 28 clases marinas + 1 clase fondo (__background__)
PHASE1_EPOCHS = 5
PHASE2_EPOCHS = 5


def collate_fn(batch):
    return tuple(zip(*batch))


def set_backbone_freeze(model, freeze_stages=True):
    """Congela o descongela capas del backbone ResNet-50."""
    body = model.backbone.body
    for layer in [body.conv1, body.bn1, body.layer1, body.layer2]:
        for p in layer.parameters():
            p.requires_grad = False

    for layer in [body.layer3, body.layer4]:
        for p in layer.parameters():
            p.requires_grad = not freeze_stages


def train_epoch(model, loader, optimizer, scaler, device):
    model.train()
    running_loss = 0.0
    for batch_idx, (images, targets) in enumerate(loader, 1):
        images = [im.to(device) for im in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()
        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        scaler.scale(losses).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()

        running_loss += losses.item() * len(images)
        if batch_idx % 150 == 0 or batch_idx == len(loader):
            print(f"  [Batch {batch_idx:03d}/{len(loader):03d}] Batch Loss: {losses.item():.4f}", flush=True)

    return running_loss / len(loader.dataset)


@torch.no_grad()
def evaluate_epoch(model, loader, device):
    """
    Evalúa la pérdida de validación de Faster R-CNN sin contaminar estadísticas BatchNorm.
    Torchvision requiere model.train() para retornar loss_dict, pero colocamos explícitamente
    todos los módulos BatchNorm2d en modo .eval() para congelar running_mean y running_var.
    """
    model.train()
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()

    running_loss = 0.0
    for images, targets in loader:
        images = [im.to(device) for im in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
        running_loss += losses.item() * len(images)

    model.eval()
    return running_loss / len(loader.dataset)


def main():
    set_seed(SEED)
    print("=== Entrenamiento: Modelo 4 - Faster R-CNN v2 (Transfer Learning COCO + Fine-Tuning) ===", flush=True)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_dir_img = os.path.join(base_dir, 'data', 'splits', 'images', 'train')
    train_dir_lbl = os.path.join(base_dir, 'data', 'splits', 'labels', 'train')
    val_dir_img = os.path.join(base_dir, 'data', 'splits', 'images', 'val')
    val_dir_lbl = os.path.join(base_dir, 'data', 'splits', 'labels', 'val')

    train_dataset = FjordFishDataset(train_dir_img, train_dir_lbl, img_size=IMG_SIZE, augment=True, mode='faster_rcnn')
    val_dataset = FjordFishDataset(val_dir_img, val_dir_lbl, img_size=IMG_SIZE, augment=False, mode='faster_rcnn')

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, collate_fn=collate_fn, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, collate_fn=collate_fn, pin_memory=True)

    weights = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
    model = fasterrcnn_resnet50_fpn_v2(weights=weights)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=FASTER_RCNN_CLASSES)
    model.to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    out_model_dir = os.path.join(base_dir, 'results', 'models')
    out_metric_dir = os.path.join(base_dir, 'results', 'metrics')
    os.makedirs(out_model_dir, exist_ok=True)
    os.makedirs(out_metric_dir, exist_ok=True)

    scaler = torch.amp.GradScaler('cuda', enabled=(DEVICE.type == 'cuda'))

    # FASE 1: Backbone congelado
    set_backbone_freeze(model, freeze_stages=True)
    trainable_params_p1 = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params_p1 = total_params - trainable_params_p1

    optimizer_p1 = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=1e-3, momentum=0.9, weight_decay=1e-4)
    scheduler_p1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_p1, T_max=PHASE1_EPOCHS, eta_min=1e-5)

    history = []
    best_val_loss = float('inf')
    start_total_time = time.time()

    for epoch in range(1, PHASE1_EPOCHS + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer_p1, scaler, DEVICE)
        val_loss = evaluate_epoch(model, val_loader, DEVICE)
        scheduler_p1.step()
        ep_time = time.time() - t0
        print(f"Fase 1 - Epoch [{epoch:02d}/{PHASE1_EPOCHS:02d}] ({ep_time:.1f}s) - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}", flush=True)

        history.append({'phase': 1, 'epoch': epoch, 'train_loss': float(train_loss), 'val_loss': float(val_loss), 'epoch_time': float(ep_time)})
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(out_model_dir, 'faster_rcnn_best.pt'))

    # FASE 2: Fine-tuning layer3 y layer4
    set_backbone_freeze(model, freeze_stages=False)
    trainable_params_p2 = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params_p2 = total_params - trainable_params_p2

    optimizer_p2 = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=1e-4, momentum=0.9, weight_decay=1e-4)
    scheduler_p2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_p2, T_max=PHASE2_EPOCHS, eta_min=1e-6)

    for epoch in range(PHASE1_EPOCHS + 1, PHASE1_EPOCHS + PHASE2_EPOCHS + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer_p2, scaler, DEVICE)
        val_loss = evaluate_epoch(model, val_loader, DEVICE)
        scheduler_p2.step()
        ep_time = time.time() - t0
        print(f"Fase 2 - Epoch [{epoch:02d}/{PHASE1_EPOCHS + PHASE2_EPOCHS:02d}] ({ep_time:.1f}s) - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}", flush=True)

        history.append({'phase': 2, 'epoch': epoch, 'train_loss': float(train_loss), 'val_loss': float(val_loss), 'epoch_time': float(ep_time)})
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(out_model_dir, 'faster_rcnn_best.pt'))

    total_time = time.time() - start_total_time
    history_file = os.path.join(out_metric_dir, 'faster_rcnn_history.json')
    with open(history_file, 'w', encoding='utf-8') as f:
        json.dump({
            'model_name': 'Faster R-CNN v2 ResNet-50 FPN (Transfer Learning COCO)',
            'total_params': total_params,
            'phase1_frozen_params': frozen_params_p1,
            'phase1_trainable_params': trainable_params_p1,
            'phase2_frozen_params': frozen_params_p2,
            'phase2_trainable_params': trainable_params_p2,
            'total_time_seconds': total_time,
            'best_val_loss': best_val_loss,
            'epochs': history
        }, f, indent=2)


if __name__ == '__main__':
    main()
