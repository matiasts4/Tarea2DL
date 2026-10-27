"""
evaluate_all_models.py
----------------------
Módulo unificado de evaluación y comparación cuantitativa de los 4 modelos de detección:
  1. Custom CNN Detector (Desde Cero) - results/models/custom_cnn_best.pt
  2. YOLOv8n (Desde Cero) - results/models/yolo_scratch/weights/best.pt
  3. YOLOv8n (Transfer Learning COCO) - results/models/yolo_transfer/weights/best.pt
  4. Faster R-CNN v2 ResNet-50 FPN (Transfer Learning COCO) - results/models/faster_rcnn_best.pt

Evaluación rigurosa sobre el conjunto de prueba BLOQUEADO (test set: 451 imágenes).
Todas las métricas (mAP@50, mAP@50:95, Precisión, Recall, F1-Score) se calculan bajo el
estándar COCO (integración de envolvente Precisión-Recall de 101 puntos por clase) a través
del módulo centralizado models.utils, sin aproximaciones heurísticas.
"""

import os
import glob
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from ultralytics import YOLO

from models.custom_detector import CustomCNNDetector
from models.utils import (
    NUM_CLASSES,
    apply_white_plot_style,
    load_yolo_labels,
    evaluate_detections_coco,
)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG_SIZE = 512


def load_test_targets(test_img_files, test_lbl_dir):
    """Carga todas las anotaciones ground-truth normalizadas [0, 1] del conjunto de test."""
    all_targets = []
    for p in test_img_files:
        base_name = os.path.splitext(os.path.basename(p))[0]
        lbl_p = os.path.join(test_lbl_dir, f"{base_name}.txt")
        gt_boxes, gt_labels = load_yolo_labels(lbl_p, xyxy=True, scale=1.0, label_offset=0)
        all_targets.append({'boxes': gt_boxes, 'labels': gt_labels})
    return all_targets


def evaluate_custom_cnn_test(model_path, test_img_files, all_targets, device):
    """Evaluación COCO real de CustomCNNDetector sobre el test set bloqueado."""
    print("Evaluando Modelo 1: Custom CNN Detector en Test Set (COCO mAP real)...", flush=True)
    model = CustomCNNDetector(num_classes=NUM_CLASSES, base_channels=32).to(device)
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    state = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state)
    model.eval()

    # Warmup y Benchmark de Latencia
    latencies = []
    for p in test_img_files[:50]:
        with Image.open(p) as raw_im:
            im = raw_im.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BILINEAR).convert('RGB')
        tensor = TF.to_tensor(im).unsqueeze(0).to(device)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model.predict(tensor, conf_thresh=0.25, iou_thresh=0.45)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t0) * 1000.0)
    avg_latency = float(np.mean(latencies[5:]))
    fps = 1000.0 / avg_latency if avg_latency > 0 else 0.0

    # Recolectar predicciones en umbral bajo (0.05) para construir curva PR completa
    all_predictions = []
    for p in test_img_files:
        with Image.open(p) as raw_im:
            im = raw_im.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BILINEAR).convert('RGB')
        tensor = TF.to_tensor(im).unsqueeze(0).to(device)
        with torch.no_grad():
            preds = model.predict(tensor, conf_thresh=0.05, iou_thresh=0.45)[0]

        all_predictions.append({
            'boxes': preds['boxes'].cpu().numpy(),
            'scores': preds['scores'].cpu().numpy(),
            'labels': preds['labels'].cpu().numpy()
        })

    metrics = evaluate_detections_coco(all_predictions, all_targets, num_classes=NUM_CLASSES, op_conf_thresh=0.25)
    total_params = sum(p.numel() for p in model.parameters())
    size_mb = os.path.getsize(model_path) / (1024 * 1024)

    return {
        'ID': 'M1',
        'Modelo': 'CNN Propia',
        'Enfoque': 'Desde Cero (Scratch)',
        'Backbone': 'Deep Residual ConvNet + FPN (Propia)',
        'mAP@50': metrics['mAP@50'],
        'mAP@50:95': metrics['mAP@50:95'],
        'Precision': metrics['Precision'],
        'Recall': metrics['Recall'],
        'F1-Score': metrics['F1-Score'],
        'Latencia (ms)': round(avg_latency, 2),
        'FPS': round(fps, 1),
        'Parámetros (M)': round(total_params / 1e6, 2),
        'Tamaño (MB)': round(size_mb, 1)
    }


def evaluate_yolo_test(model_path, model_id, model_name, approach, backbone, data_yaml):
    """Evaluación oficial de YOLOv8n usando Ultralytics COCO API sobre el test set."""
    print(f"Evaluando {model_name} ({approach}) en Test Set...", flush=True)
    yolo = YOLO(model_path)
    val_res = yolo.val(
        data=data_yaml,
        split='test',
        device='cuda' if torch.cuda.is_available() else 'cpu',
        workers=0,
        batch=8,
        verbose=False
    )

    map50 = float(val_res.box.map50)
    map50_95 = float(val_res.box.map)
    prec = float(val_res.box.mp)
    rec = float(val_res.box.mr)
    f1 = 2.0 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    avg_latency = float(val_res.speed.get('inference', 7.0))
    fps = 1000.0 / avg_latency if avg_latency > 0 else 0.0

    total_params = sum(p.numel() for p in yolo.model.parameters())
    size_mb = os.path.getsize(model_path) / (1024 * 1024)

    return {
        'ID': model_id,
        'Modelo': model_name,
        'Enfoque': approach,
        'Backbone': backbone,
        'mAP@50': round(map50, 4),
        'mAP@50:95': round(map50_95, 4),
        'Precision': round(prec, 4),
        'Recall': round(rec, 4),
        'F1-Score': round(f1, 4),
        'Latencia (ms)': round(avg_latency, 2),
        'FPS': round(fps, 1),
        'Parámetros (M)': round(total_params / 1e6, 2),
        'Tamaño (MB)': round(size_mb, 1)
    }


def evaluate_faster_rcnn_test(model_path, test_img_files, all_targets, device):
    """Evaluación COCO real de Faster R-CNN v2 sobre el test set bloqueado."""
    print("Evaluando Modelo 4: Faster R-CNN v2 ResNet-50 FPN en Test Set (COCO mAP real)...", flush=True)
    model = fasterrcnn_resnet50_fpn_v2(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=NUM_CLASSES + 1)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    model.to(device)
    model.eval()

    # Benchmark Latencia
    latencies = []
    for p in test_img_files[:50]:
        with Image.open(p) as raw_im:
            im = raw_im.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BILINEAR).convert('RGB')
        tensor = [TF.to_tensor(im).to(device)]
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                _ = model(tensor)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t0) * 1000.0)
    avg_latency = float(np.mean(latencies[5:]))
    fps = 1000.0 / avg_latency if avg_latency > 0 else 0.0

    # Recolectar predicciones para curva PR completa
    all_predictions = []
    for p in test_img_files:
        with Image.open(p) as raw_im:
            im = raw_im.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BILINEAR).convert('RGB')
        tensor = [TF.to_tensor(im).to(device)]
        with torch.no_grad():
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                preds = model(tensor)[0]

        boxes = preds['boxes'].cpu().numpy() / float(IMG_SIZE)
        scores = preds['scores'].cpu().numpy()
        labels = preds['labels'].cpu().numpy() - 1  # 0-indexed para las 28 clases marinas

        keep = scores >= 0.05
        all_predictions.append({
            'boxes': boxes[keep],
            'scores': scores[keep],
            'labels': labels[keep]
        })

    metrics = evaluate_detections_coco(all_predictions, all_targets, num_classes=NUM_CLASSES, op_conf_thresh=0.25)
    total_params = sum(p.numel() for p in model.parameters())
    size_mb = os.path.getsize(model_path) / (1024 * 1024)

    return {
        'ID': 'M4',
        'Modelo': 'Faster R-CNN v2',
        'Enfoque': 'Transfer Learning (COCO)',
        'Backbone': 'ResNet-50-FPN v2 (COCO Weights)',
        'mAP@50': metrics['mAP@50'],
        'mAP@50:95': metrics['mAP@50:95'],
        'Precision': metrics['Precision'],
        'Recall': metrics['Recall'],
        'F1-Score': metrics['F1-Score'],
        'Latencia (ms)': round(avg_latency, 2),
        'FPS': round(fps, 1),
        'Parámetros (M)': round(total_params / 1e6, 2),
        'Tamaño (MB)': round(size_mb, 1)
    }


def generate_comparison_plots(df, metrics_dir, models_dir, plots_dir):
    """Genera los 3 gráficos comparativos con fondo blanco estándar."""
    os.makedirs(plots_dir, exist_ok=True)
    apply_white_plot_style()

    labels = [f"{r['Modelo']}\n({r['Enfoque']})" for _, r in df.iterrows()]

    # 1. Gráfico de Barras Agrupadas: mAP@50 y mAP@50:95
    fig, ax = plt.subplots(figsize=(11, 6), dpi=300)
    x = np.arange(len(labels))
    width = 0.35

    rects1 = ax.bar(x - width / 2, df['mAP@50'], width, label='mAP@50 (IoU=0.50)', color='#2563EB', edgecolor='#1E3A8A', linewidth=0.8)
    rects2 = ax.bar(x + width / 2, df['mAP@50:95'], width, label='mAP@50:95 (COCO Metric)', color='#059669', edgecolor='#065F46', linewidth=0.8)

    ax.set_ylabel('Mean Average Precision (mAP)', fontsize=12, color='#111827')
    ax.set_title('Desempeño en Detección Submarina: Desde Cero vs Transfer Learning\n(Test Set FjordFish - 451 Imágenes)', fontsize=13, color='#111827', weight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10, color='#1F2937')
    ax.legend(facecolor='#FFFFFF', edgecolor='#D1D5DB', fontsize=10)
    ax.grid(axis='y', color='#E5E7EB', linestyle='--', alpha=0.8)
    ax.set_ylim(0, max(df['mAP@50']) * 1.25 if len(df) > 0 else 0.5)

    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f"{h:.3f}", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 4),
                    textcoords="offset points", ha='center', va='bottom', color='#1D4ED8', fontsize=9, weight='bold')
    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f"{h:.3f}", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 4),
                    textcoords="offset points", ha='center', va='bottom', color='#047857', fontsize=9, weight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'comparison_map_bars.png'), facecolor='white', bbox_inches='tight')
    plt.close()

    # 2. Frontera de Eficiencia de Pareto: Precisión vs Latencia
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    palette = ['#D97706', '#2563EB', '#059669', '#7C3AED']

    for idx, row in df.iterrows():
        lat = row['Latencia (ms)']
        m50 = row['mAP@50']
        params = row['Parámetros (M)']
        size_pt = max(140, params * 15)
        ax.scatter(lat, m50, s=size_pt, color=palette[idx % len(palette)], alpha=0.85, edgecolors='#111827', linewidth=1.5, zorder=5)
        offset_y = 12 if idx % 2 == 0 else -18
        ax.annotate(
            f"{row['Modelo']} ({row['Enfoque']})\n{row['FPS']} FPS | {params}M params",
            xy=(lat, m50), xytext=(12, offset_y), textcoords="offset points",
            color='#111827', fontsize=9, weight='bold',
            arrowprops=dict(arrowstyle='-', color='#6B7280', lw=0.8)
        )

    ax.axvline(x=33.3, color='#DC2626', linestyle=':', linewidth=1.5, label='Límite Tiempo Real (30 FPS = 33.3 ms)')
    ax.set_xlabel('Latencia de Inferencia por Imagen (ms) [Menor es mejor]', fontsize=11, color='#111827')
    ax.set_ylabel('Precisión de Detección (mAP@50) [Mayor es mejor]', fontsize=11, color='#111827')
    ax.set_title('Frontera de Eficiencia de Pareto: Precisión vs Latencia de Inferencia', fontsize=13, color='#111827', weight='bold', pad=15)
    ax.grid(color='#E5E7EB', linestyle='--', alpha=0.8)
    ax.legend(facecolor='#FFFFFF', edgecolor='#D1D5DB', fontsize=9, loc='lower right')

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'comparison_latency_vs_map.png'), facecolor='white', bbox_inches='tight')
    plt.close()

    # 3. Curvas de Convergencia de Entrenamiento
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    cnn_hist_p = os.path.join(metrics_dir, 'custom_cnn_history.json')
    if os.path.exists(cnn_hist_p):
        with open(cnn_hist_p, 'r', encoding='utf-8') as f:
            ch = json.load(f)['epochs']
        ep_c = [x['epoch'] for x in ch]
        ax1.plot(ep_c, [x['val_loss'] for x in ch], marker='o', markersize=4, label='CNN Propia (Val Loss)', color='#D97706', lw=2)
        ax2.plot(ep_c, [x.get('map50', x['f1']) for x in ch], marker='o', markersize=4, label='CNN Propia (Val F1/mAP)', color='#D97706', lw=2)

    ys_csv = os.path.join(models_dir, 'yolo_scratch', 'results.csv')
    if os.path.exists(ys_csv):
        df_ys = pd.read_csv(ys_csv)
        df_ys.columns = [c.strip() for c in df_ys.columns]
        ax1.plot(df_ys['epoch'], df_ys['val/box_loss'] + df_ys['val/cls_loss'], marker='s', markersize=4, label='YOLOv8n Scratch (Val Box+Cls Loss)', color='#2563EB', lw=2)
        ax2.plot(df_ys['epoch'], df_ys['metrics/mAP50(B)'], marker='s', markersize=4, label='YOLOv8n Scratch (Val mAP@50)', color='#2563EB', lw=2)

    yt_p1_csv = os.path.join(models_dir, 'yolo_transfer_phase1', 'results.csv')
    yt_p2_csv = os.path.join(models_dir, 'yolo_transfer', 'results.csv')
    if os.path.exists(yt_p2_csv):
        df_yt2 = pd.read_csv(yt_p2_csv)
        df_yt2.columns = [c.strip() for c in df_yt2.columns]
        if os.path.exists(yt_p1_csv):
            df_yt1 = pd.read_csv(yt_p1_csv)
            df_yt1.columns = [c.strip() for c in df_yt1.columns]
            ep_yt = list(df_yt1['epoch']) + list(df_yt2['epoch'] + 10)
            loss_yt = list(df_yt1['val/box_loss'] + df_yt1['val/cls_loss']) + list(df_yt2['val/box_loss'] + df_yt2['val/cls_loss'])
            map_yt = list(df_yt1['metrics/mAP50(B)']) + list(df_yt2['metrics/mAP50(B)'])
        else:
            ep_yt = list(df_yt2['epoch'] + 10)
            loss_yt = list(df_yt2['val/box_loss'] + df_yt2['val/cls_loss'])
            map_yt = list(df_yt2['metrics/mAP50(B)'])
        ax1.plot(ep_yt, loss_yt, marker='^', markersize=4, label='YOLOv8n Transfer (Val Box+Cls Loss)', color='#059669', lw=2)
        ax2.plot(ep_yt, map_yt, marker='^', markersize=4, label='YOLOv8n Transfer (Val mAP@50)', color='#059669', lw=2)

    fr_hist_p = os.path.join(metrics_dir, 'faster_rcnn_history.json')
    if os.path.exists(fr_hist_p):
        with open(fr_hist_p, 'r', encoding='utf-8') as f:
            fh = json.load(f)['epochs']
        ax1.plot([x['epoch'] for x in fh], [x['val_loss'] for x in fh], marker='d', markersize=4, label='Faster R-CNN v2 (Val Loss)', color='#7C3AED', lw=2)

    ax1.set_title('Convergencia Real de Pérdida en Validación (Loss)', fontsize=12, color='#111827', weight='bold')
    ax1.set_xlabel('Época de Entrenamiento', fontsize=10, color='#1F2937')
    ax1.set_ylabel('Loss Validación', fontsize=10, color='#1F2937')
    ax1.legend(facecolor='#FFFFFF', edgecolor='#D1D5DB', fontsize=8)
    ax1.grid(color='#E5E7EB', linestyle='--', alpha=0.8)

    ax2.set_title('Evolución de Rendimiento en Validación (mAP / F1)', fontsize=12, color='#111827', weight='bold')
    ax2.set_xlabel('Época de Entrenamiento', fontsize=10, color='#1F2937')
    ax2.set_ylabel('Métrica Validación', fontsize=10, color='#1F2937')
    ax2.legend(facecolor='#FFFFFF', edgecolor='#D1D5DB', fontsize=8)
    ax2.grid(color='#E5E7EB', linestyle='--', alpha=0.8)

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'comparison_training_convergence.png'), facecolor='white', bbox_inches='tight')
    plt.close()


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    test_img_dir = os.path.join(base_dir, 'data', 'splits', 'images', 'test')
    test_lbl_dir = os.path.join(base_dir, 'data', 'splits', 'labels', 'test')
    data_yaml = os.path.join(base_dir, 'data', 'splits', 'fjordfish.yaml')

    models_dir = os.path.join(base_dir, 'results', 'models')
    metrics_dir = os.path.join(base_dir, 'results', 'metrics')
    plots_dir = os.path.join(base_dir, 'results', 'plots')
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    test_img_files = sorted(glob.glob(os.path.join(test_img_dir, '*.jpg')))
    all_targets = load_test_targets(test_img_files, test_lbl_dir)

    records = []

    # 1. Custom CNN
    cnn_p = os.path.join(models_dir, 'custom_cnn_best.pt')
    if os.path.exists(cnn_p):
        r1 = evaluate_custom_cnn_test(cnn_p, test_img_files, all_targets, DEVICE)
        hist_p = os.path.join(metrics_dir, 'custom_cnn_history.json')
        if os.path.exists(hist_p):
            with open(hist_p, 'r', encoding='utf-8') as f:
                h = json.load(f)
            r1['Tiempo Train (s)'] = round(h.get('total_time_seconds', 0.0), 1)
            r1['Tiempo Train (min)'] = round(h.get('total_time_seconds', 0.0) / 60.0, 2)
        records.append(r1)

    # 2. YOLOv8n Scratch
    yolo_s_p = os.path.join(models_dir, 'yolo_scratch', 'weights', 'best.pt')
    if os.path.exists(yolo_s_p):
        r2 = evaluate_yolo_test(yolo_s_p, 'M2', 'YOLOv8n', 'Desde Cero (Scratch)', 'Modified CSPDarknet53', data_yaml)
        sum_p = os.path.join(metrics_dir, 'yolo_scratch_summary.json')
        if os.path.exists(sum_p):
            with open(sum_p, 'r', encoding='utf-8') as f:
                s = json.load(f)
            r2['Tiempo Train (s)'] = round(s.get('total_time_seconds', 0.0), 1)
            r2['Tiempo Train (min)'] = round(s.get('total_time_seconds', 0.0) / 60.0, 2)
        records.append(r2)

    # 3. YOLOv8n Transfer Learning
    yolo_t_p = os.path.join(models_dir, 'yolo_transfer', 'weights', 'best.pt')
    if os.path.exists(yolo_t_p):
        r3 = evaluate_yolo_test(yolo_t_p, 'M3', 'YOLOv8n', 'Transfer Learning (COCO)', 'Modified CSPDarknet53 (COCO Weights)', data_yaml)
        sum_p = os.path.join(metrics_dir, 'yolo_transfer_summary.json')
        if os.path.exists(sum_p):
            with open(sum_p, 'r', encoding='utf-8') as f:
                s = json.load(f)
            r3['Tiempo Train (s)'] = round(s.get('total_time_seconds', 0.0), 1)
            r3['Tiempo Train (min)'] = round(s.get('total_time_seconds', 0.0) / 60.0, 2)
        records.append(r3)

    # 4. Faster R-CNN v2 Transfer Learning
    frcnn_p = os.path.join(models_dir, 'faster_rcnn_best.pt')
    if os.path.exists(frcnn_p):
        r4 = evaluate_faster_rcnn_test(frcnn_p, test_img_files, all_targets, DEVICE)
        hist_p = os.path.join(metrics_dir, 'faster_rcnn_history.json')
        if os.path.exists(hist_p):
            with open(hist_p, 'r', encoding='utf-8') as f:
                h = json.load(f)
            r4['Tiempo Train (s)'] = round(h.get('total_time_seconds', 0.0), 1)
            r4['Tiempo Train (min)'] = round(h.get('total_time_seconds', 0.0) / 60.0, 2)
        records.append(r4)

    df = pd.DataFrame(records)
    csv_path = os.path.join(metrics_dir, 'model_comparison_summary.csv')
    df.to_csv(csv_path, index=False)
    print("\n=== TABLA COMPARATIVA REAL CONSOLIDADA (COCO mAP) ===")
    print(df.to_string(index=False))

    generate_comparison_plots(df, metrics_dir, models_dir, plots_dir)


if __name__ == '__main__':
    main()
