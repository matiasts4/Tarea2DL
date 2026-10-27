"""
evaluate_degradation.py
-----------------------
Sección 3: Evaluación de Robustez frente a Degradaciones Físico-Ópticas Submarinas.
Genera versiones modificadas del conjunto de prueba bloqueado (451 imágenes) bajo 4 condiciones:
  1. Atenuación Espectral y Reducción de Iluminación (absorción preferente del canal rojo)
  2. Desenfoque Óptico y Turbidez (dispersión por corrientes y partículas en movimiento)
  3. Nieve Marina y Ruido de Sensor (partículas orgánicas en suspensión con retrodispersión + ISO Noise)
  4. Dispersión hacia adelante y Velo Submarino (scattering y pérdida de contraste local)

Implementación y calibración física:
  - Preprocesamiento de canales BGR acorde al contrato de inferencia de Ultralytics.
  - Evaluación estándar de mAP@50 y mAP@50:95 mediante evaluate_detections_coco de models.utils.
  - Modelado estocástico reproducible (semilla fija) de nieve marina con dispersión alfa y halo gaussiano.
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import torch
from ultralytics import YOLO

from models.utils import (
    NUM_CLASSES,
    set_seed,
    apply_white_plot_style,
    load_yolo_labels,
    evaluate_detections_coco,
)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def apply_illumination_degradation(img_rgb, level='moderate'):
    """Simula reducción de luz y absorción espectral del canal rojo por columna de agua."""
    img = img_rgb.astype(np.float32) / 255.0
    if level == 'mild':
        factors = np.array([0.55, 0.78, 0.88], dtype=np.float32)
    elif level == 'moderate':
        factors = np.array([0.30, 0.55, 0.70], dtype=np.float32)
    else:  # severe
        factors = np.array([0.15, 0.35, 0.50], dtype=np.float32)
    degraded = np.clip(img * factors, 0.0, 1.0)
    return (degraded * 255.0).astype(np.uint8)


def apply_blur_degradation(img_rgb, level='moderate'):
    """Simula desenfoque óptico y turbidez por corrientes submarinas."""
    if level == 'mild':
        ksize = 5
    elif level == 'moderate':
        ksize = 9
    else:  # severe
        ksize = 15
    return cv2.GaussianBlur(img_rgb, (ksize, ksize), 0)


def apply_marine_snow_noise(img_rgb, level='moderate'):
    """
    Simula partículas orgánicas en suspensión (nieve marina con retrodispersión translúcida)
    combinadas con ruido electrónico de sensor ISO en baja luminosidad.
    """
    rng = np.random.default_rng(42)
    img = img_rgb.astype(np.float32)
    if level == 'mild':
        sigma = 8.0
        p_snow = 0.0004
        alpha = 0.45
    elif level == 'moderate':
        sigma = 16.0
        p_snow = 0.0010
        alpha = 0.60
    else:  # severe
        sigma = 26.0
        p_snow = 0.0022
        alpha = 0.75

    noise = rng.normal(0.0, sigma, img.shape).astype(np.float32)
    noisy_img = np.clip(img + noise, 0.0, 255.0).astype(np.uint8)
    overlay = noisy_img.copy()

    h, w, _ = noisy_img.shape
    num_particles = int(h * w * p_snow)
    for _ in range(num_particles):
        px = int(rng.integers(0, w))
        py = int(rng.integers(0, h))
        radius = int(rng.integers(1, 3))
        cv2.circle(overlay, (px, py), radius, (225, 238, 248), -1)

    blended = cv2.addWeighted(overlay, alpha, noisy_img, 1.0 - alpha, 0)
    return blended


def apply_contrast_scattering(img_rgb, level='moderate'):
    """Simula dispersión de luz (forward/backward scattering) y velo hidro-óptico."""
    img = img_rgb.astype(np.float32) / 255.0
    if level == 'mild':
        contrast = 0.68
        veil = np.array([0.14, 0.25, 0.30], dtype=np.float32)
    elif level == 'moderate':
        contrast = 0.42
        veil = np.array([0.22, 0.36, 0.42], dtype=np.float32)
    else:  # severe
        contrast = 0.22
        veil = np.array([0.30, 0.46, 0.52], dtype=np.float32)

    degraded = np.clip(img * contrast + veil * (1.0 - contrast), 0.0, 1.0)
    return (degraded * 255.0).astype(np.uint8)


def evaluate_on_degraded_images(model, img_files, all_targets, deg_fn, level, batch_size=32):
    """
    Evalúa el modelo YOLOv8 sobre las imágenes de prueba degradadas calculando el
    mAP@50 y mAP@50:95 COCO real.
    IMPORTANTE: Ultralytics YOLO.predict(np.ndarray) espera imágenes en orden BGR (OpenCV).
    """
    all_predictions = []

    for i in range(0, len(img_files), batch_size):
        chunk = img_files[i:i + batch_size]
        deg_imgs_bgr = []
        shapes = []

        for p in chunk:
            orig_bgr = cv2.imread(p)
            orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
            if deg_fn is not None:
                deg_rgb = deg_fn(orig_rgb, level=level)
            else:
                deg_rgb = orig_rgb
            # Convertir de vuelta a BGR para que Ultralytics invierta correctamente a RGB
            deg_bgr = cv2.cvtColor(deg_rgb, cv2.COLOR_RGB2BGR)
            deg_imgs_bgr.append(deg_bgr)
            shapes.append((deg_rgb.shape[0], deg_rgb.shape[1]))

        results = model.predict(
            deg_imgs_bgr, conf=0.01, iou=0.65, imgsz=640,
            device=DEVICE, verbose=False, batch=batch_size
        )

        for res, (h_orig, w_orig) in zip(results, shapes):
            if len(res.boxes) > 0:
                xyxy = res.boxes.xyxy.cpu().numpy()
                scores = res.boxes.conf.cpu().numpy()
                labels = res.boxes.cls.cpu().numpy().astype(np.int64)
                norm_boxes = np.stack([
                    np.clip(xyxy[:, 0] / w_orig, 0.0, 1.0),
                    np.clip(xyxy[:, 1] / h_orig, 0.0, 1.0),
                    np.clip(xyxy[:, 2] / w_orig, 0.0, 1.0),
                    np.clip(xyxy[:, 3] / h_orig, 0.0, 1.0),
                ], axis=-1)
                all_predictions.append({
                    'boxes': norm_boxes,
                    'scores': scores,
                    'labels': labels
                })
            else:
                all_predictions.append({
                    'boxes': np.zeros((0, 4), dtype=np.float32),
                    'scores': np.zeros((0,), dtype=np.float32),
                    'labels': np.zeros((0,), dtype=np.int64)
                })

    metrics = evaluate_detections_coco(all_predictions, all_targets, num_classes=NUM_CLASSES, op_conf_thresh=0.25)
    return metrics['mAP@50'], metrics['mAP@50:95'], metrics['Precision'], metrics['Recall'], metrics['F1-Score']


def run_degradation_study(base_dir):
    set_seed(42)
    apply_white_plot_style()

    results_plots_dir = os.path.join(base_dir, 'results', 'plots')
    results_metrics_dir = os.path.join(base_dir, 'results', 'metrics')
    os.makedirs(results_plots_dir, exist_ok=True)
    os.makedirs(results_metrics_dir, exist_ok=True)

    test_img_dir = os.path.join(base_dir, 'data', 'splits', 'images', 'test')
    test_lbl_dir = os.path.join(base_dir, 'data', 'splits', 'labels', 'test')
    img_files = sorted(glob.glob(os.path.join(test_img_dir, '*.jpg')))

    all_targets = []
    for p in img_files:
        base_name = os.path.splitext(os.path.basename(p))[0]
        lbl_p = os.path.join(test_lbl_dir, f"{base_name}.txt")
        gt_b, gt_l = load_yolo_labels(lbl_p, xyxy=True, scale=1.0, label_offset=0)
        all_targets.append({'boxes': gt_b, 'labels': gt_l})

    # 1. Generar grilla visual comparativa de una imagen de test
    if img_files:
        sample_path = img_files[0]
        orig_im = cv2.cvtColor(cv2.imread(sample_path), cv2.COLOR_BGR2RGB)
        orig_im = cv2.resize(orig_im, (640, 360))

        fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=300)
        deg_list = [
            (apply_illumination_degradation(orig_im, 'moderate'), 'A. Atenuación Espectral / Profundidad\n(Pérdida Crítica del Canal Rojo)'),
            (apply_blur_degradation(orig_im, 'moderate'), 'B. Turbidez y Desenfoque Óptico\n(Dispersión por Corrientes y Movimiento)'),
            (apply_marine_snow_noise(orig_im, 'moderate'), 'C. Nieve Marina y Ruido Sensor\n(Partículas Orgánicas + ISO Noise)'),
            (apply_contrast_scattering(orig_im, 'moderate'), 'D. Dispersión y Velo Submarino\n(Pérdida Severa de Contraste / Scattering)')
        ]

        for ax, (img_deg, title) in zip(axes.flat, deg_list):
            ax.imshow(img_deg)
            ax.set_title(title, fontsize=11, color='#111827', weight='bold', pad=8)
            ax.axis('off')

        plt.tight_layout()
        grid_sample_p = os.path.join(results_plots_dir, 'degraded_samples_grid.png')
        plt.savefig(grid_sample_p, facecolor='white', bbox_inches='tight')
        plt.close()

    model_path = os.path.join(base_dir, 'results', 'models', 'yolo_transfer', 'weights', 'best.pt')
    if not os.path.exists(model_path):
        print(f"Modelo no encontrado en: {model_path}")
        return

    print("Cargando modelo YOLOv8n Transfer Learning...", flush=True)
    model = YOLO(model_path)

    print("Evaluando condición Baseline (Limpio) con COCO mAP real...", flush=True)
    base_m50, base_m50_95, base_p, base_r, _ = evaluate_on_degraded_images(model, img_files, all_targets, None, 'clean')

    degradations = [
        ('Atenuación Espectral (Luz)', apply_illumination_degradation),
        ('Desenfoque Óptico / Turbidez', apply_blur_degradation),
        ('Nieve Marina y Ruido', apply_marine_snow_noise),
        ('Dispersión y Pérdida Contraste', apply_contrast_scattering)
    ]

    levels = ['mild', 'moderate', 'severe']
    level_names = {'mild': 'Leve', 'moderate': 'Moderado', 'severe': 'Severo'}
    factors_map = {
        'Atenuación Espectral (Luz)': {'mild': 'R*0.55', 'moderate': 'R*0.30', 'severe': 'R*0.15'},
        'Desenfoque Óptico / Turbidez': {'mild': 'k=5', 'moderate': 'k=9', 'severe': 'k=15'},
        'Nieve Marina y Ruido': {'mild': 'σ=8, 0.04%', 'moderate': 'σ=16, 0.10%', 'severe': 'σ=26, 0.22%'},
        'Dispersión y Pérdida Contraste': {'mild': 'c=0.68', 'moderate': 'c=0.42', 'severe': 'c=0.22'}
    }

    records = [{
        'Condición': 'Control (Test Set Limpio)',
        'Nivel': 'Baseline',
        'Factor': '1.0',
        'mAP@50': round(base_m50, 4),
        'mAP@50:95': round(base_m50_95, 4),
        'Precision': round(base_p, 4),
        'Recall': round(base_r, 4),
        'Drop %': 0.0
    }]

    for deg_name, deg_fn in degradations:
        for lvl in levels:
            lvl_name = level_names[lvl]
            factor = factors_map[deg_name][lvl]
            print(f"Evaluando: {deg_name} - Nivel: {lvl_name}...", flush=True)
            m50, m50_95, prec, rec, _ = evaluate_on_degraded_images(model, img_files, all_targets, deg_fn, lvl)
            drop = ((m50 - base_m50) / base_m50) * 100.0 if base_m50 > 0 else 0.0
            records.append({
                'Condición': deg_name,
                'Nivel': lvl_name,
                'Factor': factor,
                'mAP@50': round(m50, 4),
                'mAP@50:95': round(m50_95, 4),
                'Precision': round(prec, 4),
                'Recall': round(rec, 4),
                'Drop %': round(drop, 2)
            })

    df_deg = pd.DataFrame(records)
    csv_out = os.path.join(results_metrics_dir, 'degradation_robustness_summary.csv')
    df_deg.to_csv(csv_out, index=False)
    print("\n=== RESUMEN DE ROBUSTEZ (COCO mAP REAL) ===")
    print(df_deg.to_string(index=False))

    # Gráfico: Curvas de Degradación mAP@50
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    colors = {
        'Atenuación Espectral (Luz)': '#2563EB',
        'Desenfoque Óptico / Turbidez': '#059669',
        'Nieve Marina y Ruido': '#D97706',
        'Dispersión y Pérdida Contraste': '#DC2626'
    }

    x_labels = ['Baseline', 'Leve', 'Moderado', 'Severo']
    x_indices = np.arange(4)

    for deg_name in colors.keys():
        sub = df_deg[df_deg['Condición'] == deg_name]
        y_vals = [base_m50] + list(sub['mAP@50'].values)
        ax.plot(x_indices, y_vals, marker='o', lw=2.5, color=colors[deg_name], label=deg_name)
        for idx, y in zip(x_indices, y_vals):
            ax.annotate(f"{y:.3f}", xy=(idx, y), xytext=(0, 6), textcoords="offset points",
                        color=colors[deg_name], fontsize=9, ha='center', weight='bold')

    ax.set_xticks(x_indices)
    ax.set_xticklabels(x_labels, fontsize=11, color='#1F2937')
    ax.set_ylabel('mAP@50 (IoU=0.50)', fontsize=12, color='#111827')
    ax.set_title('Impacto de Condiciones Submarinas Adversas en Detección (mAP@50)\n(YOLOv8n Transfer Learning - Test Set FjordFish)',
                 fontsize=13, color='#111827', weight='bold', pad=15)
    ax.grid(color='#E5E7EB', linestyle='--', alpha=0.8)
    ax.legend(facecolor='#FFFFFF', edgecolor='#D1D5DB', fontsize=10)
    ax.set_ylim(0, max(0.35, base_m50 * 1.25))

    plt.tight_layout()
    p_deg = os.path.join(results_plots_dir, 'degradation_curves_map50.png')
    plt.savefig(p_deg, facecolor='white', bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    base_folder = os.path.dirname(os.path.abspath(__file__))
    run_degradation_study(base_folder)
