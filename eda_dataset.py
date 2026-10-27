"""
eda_dataset.py
--------------
Análisis Exploratorio de Datos (EDA) del dataset FjordFish:
- Distribución de frecuencias de las 28 clases biológicas.
- Distribución de dimensiones y geometrías de bounding boxes (áreas relativas y aspect ratio).
- Distribución espectral de color submarino (absorción del canal Rojo frente a Verde y Azul).
- Muestra visual anotada de detección submarina.
Utiliza las definiciones y funciones centralizadas de models.utils.
"""

import os
import glob
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageDraw

from models.utils import (
    CLASS_NAMES,
    NUM_CLASSES,
    apply_white_plot_style,
    load_yolo_labels,
)


def run_eda(data_dir, results_dir):
    os.makedirs(results_dir, exist_ok=True)
    apply_white_plot_style()

    manifest_path = os.path.join(data_dir, 'dataset_manifest.csv')
    if os.path.exists(manifest_path):
        df = pd.read_csv(manifest_path)
        print("Resumen de particiones:")
        print(df['split'].value_counts())

    box_records = []
    class_counts = Counter()

    for split in ['train', 'val', 'test']:
        lbl_files = sorted(glob.glob(os.path.join(data_dir, 'labels', split, '*.txt')))
        for fpath in lbl_files:
            img_id = os.path.splitext(os.path.basename(fpath))[0]
            boxes_np, labels_np = load_yolo_labels(fpath, xyxy=False, scale=1.0)
            for (cx, cy, w, h), cid in zip(boxes_np, labels_np):
                cid = int(cid)
                area = float(w * h)
                aspect_ratio = float(w / (h + 1e-6))
                class_counts[cid] += 1
                box_records.append({
                    'img_id': img_id,
                    'split': split,
                    'class_id': cid,
                    'class_name': CLASS_NAMES.get(cid, str(cid)),
                    'w': float(w),
                    'h': float(h),
                    'area': area,
                    'aspect_ratio': aspect_ratio
                })

    df_boxes = pd.DataFrame(box_records)
    print(f"Total bboxes analizados: {len(df_boxes)}")

    # --- GRÁFICO 1: Distribución de Clases (Long-Tail) ---
    fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
    sorted_classes = sorted(
        [(cid, CLASS_NAMES[cid], class_counts[cid]) for cid in range(NUM_CLASSES)],
        key=lambda x: x[2],
        reverse=True
    )
    c_names = [f"{x[1]} ({x[0]})" for x in sorted_classes]
    c_vals = [x[2] for x in sorted_classes]

    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(c_names)))
    bars = ax.barh(c_names[::-1], c_vals[::-1], color=colors, edgecolor='#1E3A8A', linewidth=0.5)

    ax.set_title(
        'Distribución Ecológica de Especies Marinas (FjordFish)\nDesbalance de Clases Tipo Long-Tail',
        fontsize=14, color='#111827', weight='bold', pad=15
    )
    ax.set_xlabel('Número Total de Anotaciones (Bounding Boxes)', fontsize=11, color='#1F2937')
    ax.tick_params(colors='#1F2937', labelsize=9)
    ax.grid(axis='x', color='#E5E7EB', linestyle='--', alpha=0.8)

    for bar in bars:
        w = bar.get_width()
        if w > 0:
            ax.text(w + 15, bar.get_y() + bar.get_height() / 2, f"{int(w)}", va='center', ha='left', color='#111827', fontsize=9)

    plt.tight_layout()
    g1_path = os.path.join(results_dir, 'eda_class_distribution.png')
    plt.savefig(g1_path, facecolor='white', bbox_inches='tight')
    plt.close()

    # --- GRÁFICO 2: Geometría de Bounding Boxes (Área y Aspect Ratio) ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
    small_mask = df_boxes['area'] < 0.001
    med_mask = (df_boxes['area'] >= 0.001) & (df_boxes['area'] < 0.01)
    large_mask = df_boxes['area'] >= 0.01

    area_counts = [small_mask.sum(), med_mask.sum(), large_mask.sum()]
    area_labels = [
        f"Pequeño (<0.1%)\n{area_counts[0]} ({area_counts[0]/len(df_boxes):.1%})",
        f"Mediano (0.1%-1%)\n{area_counts[1]} ({area_counts[1]/len(df_boxes):.1%})",
        f"Grande (>1%)\n{area_counts[2]} ({area_counts[2]/len(df_boxes):.1%})"
    ]

    ax1.pie(
        area_counts, labels=area_labels, colors=['#3B82F6', '#10B981', '#F59E0B'],
        wedgeprops={'edgecolor': 'white', 'linewidth': 2},
        textprops={'color': '#111827', 'fontsize': 10}, startangle=140
    )
    ax1.set_title('Escala de Objetos (Área Relativa)\nReto de Detección Multiescala', fontsize=12, color='#111827', weight='bold')

    sns.kdeplot(df_boxes['aspect_ratio'].clip(0, 5), ax=ax2, fill=True, color='#2563EB', alpha=0.4, linewidth=2)
    ax2.set_title('Distribución de Aspect Ratio (Ancho / Alto)', fontsize=12, color='#111827', weight='bold')
    ax2.set_xlabel('Aspect Ratio (w / h)', fontsize=11, color='#1F2937')
    ax2.set_ylabel('Densidad', fontsize=11, color='#1F2937')
    ax2.grid(color='#E5E7EB', linestyle='--', alpha=0.8)

    plt.tight_layout()
    g2_path = os.path.join(results_dir, 'eda_bbox_geometry.png')
    plt.savefig(g2_path, facecolor='white', bbox_inches='tight')
    plt.close()

    # --- GRÁFICO 3: Perfil Espectral Submarino (Canales R, G, B) ---
    train_imgs = sorted(glob.glob(os.path.join(data_dir, 'images', 'train', '*.jpg')))[:100]
    r_vals, g_vals, b_vals = [], [], []
    for ipath in train_imgs:
        im = Image.open(ipath).convert('RGB')
        im_arr = np.array(im) / 255.0
        r_vals.append(im_arr[:, :, 0].mean())
        g_vals.append(im_arr[:, :, 1].mean())
        b_vals.append(im_arr[:, :, 2].mean())

    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    channel_means = [np.mean(r_vals), np.mean(g_vals), np.mean(b_vals)]
    channel_stds = [np.std(r_vals), np.std(g_vals), np.std(b_vals)]
    channel_names = ['Rojo (Red)', 'Verde (Green)', 'Azul (Blue)']
    bar_colors = ['#EF4444', '#10B981', '#3B82F6']

    bars = ax.bar(
        channel_names, channel_means, yerr=channel_stds, color=bar_colors, capsize=8,
        edgecolor='#374151', linewidth=0.8, alpha=0.85
    )
    ax.set_title('Atenuación Óptica Submarina en FjordFish\nAbsorción Selectiva del Canal Rojo', fontsize=13, color='#111827', weight='bold', pad=15)
    ax.set_ylabel('Intensidad Media Normalizada [0, 1]', fontsize=11, color='#1F2937')
    ax.set_ylim(0, 0.8)
    ax.grid(axis='y', color='#E5E7EB', linestyle='--', alpha=0.8)

    for bar, val in zip(bars, channel_means):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.03, f"{val:.3f}", ha='center', va='bottom', color='#111827', weight='bold')

    plt.tight_layout()
    g3_path = os.path.join(results_dir, 'eda_color_absorption.png')
    plt.savefig(g3_path, facecolor='white', bbox_inches='tight')
    plt.close()

    # --- GRÁFICO 4: Muestra Visual Anotada ---
    sample_img_p = os.path.join(data_dir, 'images', 'train', '1.jpg')
    sample_lbl_p = os.path.join(data_dir, 'labels', 'train', '1.txt')
    if os.path.exists(sample_img_p) and os.path.exists(sample_lbl_p):
        im = Image.open(sample_img_p).convert('RGB')
        w_img, h_img = im.size
        draw = ImageDraw.Draw(im)
        boxes_np, labels_np = load_yolo_labels(sample_lbl_p, xyxy=False, scale=1.0)
        for (cx, cy, bw, bh), cid in zip(boxes_np, labels_np):
            cid = int(cid)
            x1, y1 = (cx - bw / 2) * w_img, (cy - bh / 2) * h_img
            x2, y2 = (cx + bw / 2) * w_img, (cy + bh / 2) * h_img
            draw.rectangle([x1, y1, x2, y2], outline='#38BDF8', width=4)
            draw.text((x1 + 4, max(0, y1 - 18)), f"{CLASS_NAMES.get(cid, str(cid))} ({cid})", fill='#FFFFFF')

        im_resized = im.resize((1280, 720))
        g4_path = os.path.join(results_dir, 'eda_sample_annotation.png')
        im_resized.save(g4_path)


if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_split_dir = os.path.join(base_dir, 'data', 'splits')
    res_dir = os.path.join(base_dir, 'results', 'eda')
    run_eda(data_split_dir, res_dir)
