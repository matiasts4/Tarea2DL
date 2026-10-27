"""
models/utils.py
---------------
Módulo centralizado de utilidades, carga de datos, arquitectura de Dataset PyTorch,
cálculo riguroso de métricas de detección (COCO mAP@50, mAP@50:95 con interpolación
de curva Precisión-Recall de 101 puntos, Precisión, Recall, F1) y estilo gráfico.

Centraliza la lógica compartida para eliminar duplicación de código entre:
  - prepare_dataset.py
  - eda_dataset.py
  - train_custom_cnn.py
  - train_faster_rcnn_transfer.py
  - evaluate_all_models.py
  - evaluate_degradation.py
"""

import os
import glob
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF

# ==============================================================================
# 1. CONSTANTES Y TAXONOMÍA CANÓNICA FJORDFISH (28 ESPECIES)
# ==============================================================================
NUM_CLASSES = 28

CLASS_NAMES = {
    0: 'ballan',
    1: 'cuckoo_female',
    2: 'cuckoo_male',
    3: 'corkwing_female',
    4: 'corkwing_male',
    5: 'goldsinny',
    6: 'rock_cook',
    7: 'labrids_unknown',
    8: 'cod',
    9: 'cod_yoy',
    10: 'haddock',
    11: 'pollack',
    12: 'poor_cod',
    13: 'saithe',
    14: 'whiting',
    15: 'gadines_unknown',
    16: 'spiny_dogfish',
    17: 'sharks_unknown',
    18: 'brown_crab',
    19: 'green_crab',
    20: 'hermit',
    21: 'lobster',
    22: 'decapods_unknown',
    23: 'ling',
    24: 'two_spot_goby',
    25: 'black_goby',
    26: 'sand_goby',
    27: 'unknown'
}


def set_seed(seed=42):
    """Fija semillas deterministas para reproducibilidad estricta."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def apply_white_plot_style():
    """Configura el estilo visual estándar de matplotlib con fondo blanco puro."""
    plt.style.use('default')
    plt.rcParams.update({
        'figure.facecolor': '#FFFFFF',
        'axes.facecolor': '#FFFFFF',
        'axes.edgecolor': '#D1D5DB',
        'axes.labelcolor': '#111827',
        'text.color': '#111827',
        'xtick.color': '#1F2937',
        'ytick.color': '#1F2937',
        'grid.color': '#E5E7EB',
        'grid.linestyle': '--',
        'grid.alpha': 0.8,
        'font.sans-serif': ['DejaVu Sans', 'Arial', 'Segoe UI']
    })


# ==============================================================================
# 2. CARGA UNIFICADA DE ANOTACIONES Y DATASET PYTORCH
# ==============================================================================
def load_yolo_labels(lbl_path, xyxy=True, scale=1.0, label_offset=0):
    """
    Lee un archivo de etiquetas en formato YOLO (class_id cx cy w h).

    Args:
        lbl_path: Ruta al archivo .txt.
        xyxy: Si True, convierte a [x1, y1, x2, y2]. Si False, mantiene [cx, cy, w, h].
        scale: Factor de escala espacial (1.0 para coordenadas normalizadas [0,1],
               o img_size para coordenadas en píxeles).
        label_offset: Desplazamiento de índice de clase (0 para YOLO/CustomCNN,
                      +1 para Faster R-CNN donde 0 es __background__).
    Returns:
        boxes (np.ndarray [M, 4]), labels (np.ndarray [M])
    """
    boxes = []
    labels = []
    if lbl_path and os.path.exists(lbl_path):
        with open(lbl_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cid = int(parts[0]) + label_offset
                    cx = float(parts[1]) * scale
                    cy = float(parts[2]) * scale
                    w = float(parts[3]) * scale
                    h = float(parts[4]) * scale
                    if xyxy:
                        x1 = max(0.0, cx - w / 2.0)
                        y1 = max(0.0, cy - h / 2.0)
                        x2 = min(float(scale), cx + w / 2.0)
                        y2 = min(float(scale), cy + h / 2.0)
                        if x2 > x1 and y2 > y1:
                            boxes.append([x1, y1, x2, y2])
                            labels.append(cid)
                    else:
                        if w > 0 and h > 0:
                            boxes.append([cx, cy, w, h])
                            labels.append(cid)

    if boxes:
        return np.array(boxes, dtype=np.float32), np.array(labels, dtype=np.int64)
    return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)


class FjordFishDataset(Dataset):
    """
    Dataset PyTorch unificado para FjordFish.
    Soporta ambos modos de entrenamiento:
      - mode='custom_cnn': retorna (img_tensor, boxes_cxcywh_norm, labels_0idx)
      - mode='faster_rcnn': retorna (img_tensor, target_dict_xyxy_pixels_1idx)
    """
    def __init__(self, img_dir, lbl_dir, img_size=512, augment=False, mode='custom_cnn'):
        self.img_files = sorted(glob.glob(os.path.join(img_dir, '*.jpg')))
        self.lbl_dir = lbl_dir
        self.img_size = img_size
        self.augment = augment
        self.mode = mode

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        img_id = os.path.splitext(os.path.basename(img_path))[0]
        lbl_path = os.path.join(self.lbl_dir, f"{img_id}.txt")

        img = Image.open(img_path).convert('RGB').resize(
            (self.img_size, self.img_size), Image.Resampling.BILINEAR
        )

        if self.mode == 'faster_rcnn':
            boxes_np, labels_np = load_yolo_labels(
                lbl_path, xyxy=True, scale=float(self.img_size), label_offset=1
            )
            # Filtrar cajas degeneradas menores a 1 px
            if len(boxes_np) > 0:
                valid = (boxes_np[:, 2] - boxes_np[:, 0] > 1.0) & (boxes_np[:, 3] - boxes_np[:, 1] > 1.0)
                boxes_np = boxes_np[valid]
                labels_np = labels_np[valid]

            if self.augment and torch.rand(1).item() > 0.5:
                img = TF.hflip(img)
                if len(boxes_np) > 0:
                    old_x1 = boxes_np[:, 0].copy()
                    old_x2 = boxes_np[:, 2].copy()
                    boxes_np[:, 0] = self.img_size - old_x2
                    boxes_np[:, 2] = self.img_size - old_x1

            img_tensor = TF.to_tensor(img)
            target = {
                'boxes': torch.as_tensor(boxes_np, dtype=torch.float32),
                'labels': torch.as_tensor(labels_np, dtype=torch.int64)
            }
            return img_tensor, target

        else:  # mode == 'custom_cnn'
            boxes_np, labels_np = load_yolo_labels(
                lbl_path, xyxy=False, scale=1.0, label_offset=0
            )
            if self.augment and torch.rand(1).item() > 0.5:
                img = TF.hflip(img)
                if len(boxes_np) > 0:
                    boxes_np[:, 0] = 1.0 - boxes_np[:, 0]

            img_tensor = TF.to_tensor(img)
            boxes_t = torch.as_tensor(boxes_np, dtype=torch.float32)
            labels_t = torch.as_tensor(labels_np, dtype=torch.long)
            return img_tensor, boxes_t, labels_t


# ==============================================================================
# 3. CÁLCULO VECTORIZADO DE IoU Y MÉTRICAS REALES (COCO mAP@50 y mAP@50:95)
# ==============================================================================
def compute_iou_matrix(boxes1, boxes2):
    """
    Calcula la matriz de IoU [N, M] entre dos conjuntos de cajas [x1, y1, x2, y2].
    Acepta arreglos NumPy o tensores PyTorch y retorna un arreglo NumPy [N, M].
    """
    if isinstance(boxes1, torch.Tensor):
        boxes1 = boxes1.detach().cpu().numpy()
    if isinstance(boxes2, torch.Tensor):
        boxes2 = boxes2.detach().cpu().numpy()

    boxes1 = np.asarray(boxes1, dtype=np.float32)
    boxes2 = np.asarray(boxes2, dtype=np.float32)

    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)), dtype=np.float32)

    area1 = np.maximum(0.0, boxes1[:, 2] - boxes1[:, 0]) * np.maximum(0.0, boxes1[:, 3] - boxes1[:, 1])
    area2 = np.maximum(0.0, boxes2[:, 2] - boxes2[:, 0]) * np.maximum(0.0, boxes2[:, 3] - boxes2[:, 1])

    lt = np.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = np.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = np.maximum(0.0, rb - lt)
    inter = wh[:, :, 0] * wh[:, :, 1]

    union = area1[:, None] + area2[None, :] - inter
    return np.where(union > 1e-7, inter / union, 0.0)


def compute_ap_101(recall, precision):
    """
    Calcula el Average Precision (AP) usando interpolación de envolvente monótona
    en 101 puntos de recall [0.00, 0.01, ..., 1.00] (estándar oficial COCO).
    """
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    # Envolvente monótona decreciente de precisión
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # 101 puntos de muestreo COCO
    x = np.linspace(0.0, 1.0, 101)
    trapz_fn = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
    ap = trapz_fn(np.interp(x, mrec, mpre), x)
    return float(ap)


def evaluate_detections_coco(all_predictions, all_targets, num_classes=NUM_CLASSES, op_conf_thresh=0.25):
    """
    Calcula las métricas reales de detección de objetos bajo el estándar COCO:
      - mAP@50: Media de AP en IoU=0.50 sobre todas las clases presentes en el conjunto.
      - mAP@50:95: Media de AP promediada en 10 umbrales IoU de 0.50 a 0.95 (paso 0.05).
      - Precision, Recall y F1-Score reales en el umbral operativo (op_conf_thresh).

    Args:
        all_predictions: Lista por imagen de dicts {'boxes': [N,4], 'scores': [N], 'labels': [N]}
        all_targets: Lista por imagen de dicts {'boxes': [M,4], 'labels': [M]}
        num_classes: Total de clases (28).
        op_conf_thresh: Umbral de confianza para reportar Precisión, Recall y F1 puntuales.
    """
    iou_thresholds = np.linspace(0.50, 0.95, 10)

    # 1. Calcular AP por clase y por umbral IoU integrando la curva PR completa
    ap_per_class_50 = []
    ap_per_class_50_95 = []

    for c in range(num_classes):
        # Recolectar todos los GT de la clase c por imagen
        gt_by_img = {}
        n_gt = 0
        for img_idx, tgt in enumerate(all_targets):
            t_boxes = np.asarray(tgt['boxes'], dtype=np.float32)
            t_labels = np.asarray(tgt['labels'], dtype=np.int64)
            mask_c = (t_labels == c)
            c_boxes = t_boxes[mask_c]
            gt_by_img[img_idx] = c_boxes
            n_gt += len(c_boxes)

        if n_gt == 0:
            continue

        # Recolectar todas las predicciones de la clase c
        preds_c = []
        for img_idx, pred in enumerate(all_predictions):
            p_boxes = np.asarray(pred['boxes'], dtype=np.float32)
            p_scores = np.asarray(pred['scores'], dtype=np.float32)
            p_labels = np.asarray(pred['labels'], dtype=np.int64)
            mask_c = (p_labels == c)
            for b, s in zip(p_boxes[mask_c], p_scores[mask_c]):
                preds_c.append((img_idx, float(s), b))

        if len(preds_c) == 0:
            ap_per_class_50.append(0.0)
            ap_per_class_50_95.append(0.0)
            continue

        # Ordenar predicciones por score descendente
        preds_c.sort(key=lambda x: x[1], reverse=True)
        n_preds = len(preds_c)

        aps_at_ious = []
        for tau in iou_thresholds:
            tp = np.zeros(n_preds, dtype=np.float32)
            fp = np.zeros(n_preds, dtype=np.float32)
            matched_gt = {img_idx: set() for img_idx in gt_by_img}

            for p_idx, (img_idx, score, p_box) in enumerate(preds_c):
                gt_boxes = gt_by_img[img_idx]
                if len(gt_boxes) == 0:
                    fp[p_idx] = 1.0
                    continue

                ious = compute_iou_matrix(p_box[None, :], gt_boxes)[0]
                best_iou = 0.0
                best_gt_idx = -1
                for g_i, iou_val in enumerate(ious):
                    if g_i not in matched_gt[img_idx] and iou_val > best_iou:
                        best_iou = iou_val
                        best_gt_idx = g_i

                if best_iou >= tau and best_gt_idx >= 0:
                    tp[p_idx] = 1.0
                    matched_gt[img_idx].add(best_gt_idx)
                else:
                    fp[p_idx] = 1.0

            cum_tp = np.cumsum(tp)
            cum_fp = np.cumsum(fp)
            recall_curve = cum_tp / float(n_gt)
            precision_curve = cum_tp / np.maximum(cum_tp + cum_fp, 1e-9)

            ap_tau = compute_ap_101(recall_curve, precision_curve)
            aps_at_ious.append(ap_tau)

        ap_per_class_50.append(aps_at_ious[0])
        ap_per_class_50_95.append(float(np.mean(aps_at_ious)))

    map50 = float(np.mean(ap_per_class_50)) if ap_per_class_50 else 0.0
    map50_95 = float(np.mean(ap_per_class_50_95)) if ap_per_class_50_95 else 0.0

    # 2. Calcular Precisión, Recall y F1 en el punto de operación (IoU >= 0.50, score >= op_conf_thresh)
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for pred, tgt in zip(all_predictions, all_targets):
        p_boxes = np.asarray(pred['boxes'], dtype=np.float32)
        p_scores = np.asarray(pred['scores'], dtype=np.float32)
        p_labels = np.asarray(pred['labels'], dtype=np.int64)

        keep = p_scores >= op_conf_thresh
        p_boxes = p_boxes[keep]
        p_scores = p_scores[keep]
        p_labels = p_labels[keep]

        # Ordenar por confianza descendente
        if len(p_scores) > 0:
            order = np.argsort(-p_scores)
            p_boxes = p_boxes[order]
            p_labels = p_labels[order]

        gt_boxes = np.asarray(tgt['boxes'], dtype=np.float32)
        gt_labels = np.asarray(tgt['labels'], dtype=np.int64)

        if len(gt_boxes) == 0:
            total_fp += len(p_boxes)
            continue
        if len(p_boxes) == 0:
            total_fn += len(gt_boxes)
            continue

        ious = compute_iou_matrix(p_boxes, gt_boxes)
        matched = set()
        for p_i in range(len(p_boxes)):
            best_iou = 0.0
            best_g = -1
            for g_i in range(len(gt_boxes)):
                if g_i not in matched and p_labels[p_i] == gt_labels[g_i]:
                    if ious[p_i, g_i] > best_iou:
                        best_iou = ious[p_i, g_i]
                        best_g = g_i
            if best_iou >= 0.50 and best_g >= 0:
                total_tp += 1
                matched.add(best_g)
            else:
                total_fp += 1
        total_fn += (len(gt_boxes) - len(matched))

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'mAP@50': round(map50, 4),
        'mAP@50:95': round(map50_95, 4),
        'Precision': round(float(precision), 4),
        'Recall': round(float(recall), 4),
        'F1-Score': round(float(f1), 4),
        'TP': int(total_tp),
        'FP': int(total_fp),
        'FN': int(total_fn)
    }
