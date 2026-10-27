"""
prepare_dataset.py
------------------
Script de particionado estratificado multi-etiqueta y preparación del dataset FjordFish.
Divide el dataset de 3.002 imágenes submarinas en:
  - Train: 70% (2.101 imágenes)
  - Val:   15% (450 imágenes)
  - Test:  15% (451 imágenes)
Utiliza la taxonomía centralizada de models.utils y balancea explícitamente las frecuencias
por clase (train_class_counts, val_class_counts, test_class_counts) junto con la capacidad global.
"""

import os
import glob
import shutil
import random
from collections import Counter
import pandas as pd

from models.utils import CLASS_NAMES, NUM_CLASSES, load_yolo_labels


def load_all_samples(raw_dir):
    """Carga todas las rutas de imágenes y sus archivos de etiquetas correspondientes."""
    all_imgs = {}
    for d in ['train', 'val']:
        for p in glob.glob(os.path.join(raw_dir, 'images', d, '*.jpg')):
            img_id = os.path.splitext(os.path.basename(p))[0]
            all_imgs[img_id] = p

    all_lbls = {}
    for d in ['train', 'val']:
        for p in glob.glob(os.path.join(raw_dir, 'labels', d, '*.txt')):
            img_id = os.path.splitext(os.path.basename(p))[0]
            all_lbls[img_id] = p

    samples = []
    missing_lbls = 0
    for img_id, img_path in sorted(all_imgs.items()):
        lbl_path = all_lbls.get(img_id, None)
        boxes_np, labels_np = load_yolo_labels(lbl_path, xyxy=False, scale=1.0)
        if lbl_path is None or not os.path.exists(lbl_path):
            missing_lbls += 1

        samples.append({
            'img_id': img_id,
            'img_path': img_path,
            'lbl_path': lbl_path,
            'classes': labels_np.tolist(),
            'num_boxes': len(boxes_np),
            'boxes': boxes_np.tolist()
        })

    print(f"Total imágenes encontradas: {len(samples)}")
    print(f"Imágenes sin etiqueta o vacías: {missing_lbls}")
    return samples


def stratified_split(samples, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Realiza una partición estratificada multi-etiqueta (Iterative / Greedy Multilabel Stratification)
    ponderando tanto el déficit de cada especie en cada partición como el tamaño global del split.
    """
    random.seed(seed)

    class_freq = Counter()
    for s in samples:
        for c in set(s['classes']):
            class_freq[c] += 1

    def sample_rarity(sample):
        if not sample['classes']:
            return 999999
        return min(class_freq[c] for c in sample['classes'])

    sorted_samples = sorted(samples, key=lambda s: (sample_rarity(s), random.random()))

    splits = {'train': [], 'val': [], 'test': []}
    class_counts = {'train': Counter(), 'val': Counter(), 'test': Counter()}
    ratios = {'train': train_ratio, 'val': val_ratio, 'test': test_ratio}

    target_sizes = {
        'train': int(len(samples) * train_ratio),
        'val': int(len(samples) * val_ratio),
    }
    target_sizes['test'] = len(samples) - target_sizes['train'] - target_sizes['val']

    for s in sorted_samples:
        s_classes = set(s['classes'])
        available = [sp for sp in ['train', 'val', 'test'] if len(splits[sp]) < target_sizes[sp]]
        if not available:
            available = ['train']

        # Calcular puntaje de llenado combinando ratio por clase rara y ratio de tamaño global
        candidates = []
        for sp in available:
            global_fill = len(splits[sp]) / float(max(1, target_sizes[sp]))
            if s_classes:
                class_fills = [
                    class_counts[sp][c] / max(1e-6, class_freq[c] * ratios[sp])
                    for c in s_classes
                ]
                score = 0.75 * max(class_fills) + 0.25 * global_fill
            else:
                score = global_fill
            candidates.append((sp, score))

        candidates.sort(key=lambda x: x[1])
        chosen_split = candidates[0][0]

        splits[chosen_split].append(s)
        for c in s_classes:
            class_counts[chosen_split][c] += 1

    print(
        f"Particiones generadas: Train={len(splits['train'])}, "
        f"Val={len(splits['val'])}, Test={len(splits['test'])}"
    )
    return splits['train'], splits['val'], splits['test']


def export_splits(train_s, val_s, test_s, out_dir):
    """Copia los archivos a la estructura organizada y genera el manifiesto CSV y YAML."""
    for split_name, s_list in [('train', train_s), ('val', val_s), ('test', test_s)]:
        img_out = os.path.join(out_dir, 'images', split_name)
        lbl_out = os.path.join(out_dir, 'labels', split_name)
        os.makedirs(img_out, exist_ok=True)
        os.makedirs(lbl_out, exist_ok=True)

        for s in s_list:
            dst_img = os.path.join(img_out, f"{s['img_id']}.jpg")
            dst_lbl = os.path.join(lbl_out, f"{s['img_id']}.txt")

            shutil.copy2(s['img_path'], dst_img)
            if s['lbl_path'] and os.path.exists(s['lbl_path']):
                shutil.copy2(s['lbl_path'], dst_lbl)
            else:
                with open(dst_lbl, 'w', encoding='utf-8'):
                    pass

    rows = []
    for split_name, s_list in [('train', train_s), ('val', val_s), ('test', test_s)]:
        for s in s_list:
            rows.append({
                'img_id': s['img_id'],
                'split': split_name,
                'num_boxes': s['num_boxes'],
                'classes': ';'.join(map(str, sorted(list(set(s['classes'])))))
            })
    df_manifest = pd.DataFrame(rows)
    manifest_path = os.path.join(out_dir, 'dataset_manifest.csv')
    df_manifest.to_csv(manifest_path, index=False)
    print(f"Manifiesto guardado en: {manifest_path}")

    abs_out = os.path.abspath(out_dir).replace('\\', '/')
    yaml_content = f"path: {abs_out}\ntrain: images/train\nval: images/val\ntest: images/test\n\nnames:\n"
    for cid in range(NUM_CLASSES):
        yaml_content += f"  {cid}: {CLASS_NAMES[cid]}\n"

    yaml_path = os.path.join(out_dir, 'fjordfish.yaml')
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)
    print(f"YAML de configuración YOLO guardado en: {yaml_path}")


if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.abspath(__file__))
    raw_path = os.path.join(base_dir, 'data', 'raw', 'FjordFish')
    out_path = os.path.join(base_dir, 'data', 'splits')
    print("Iniciando preparación del dataset FjordFish...")
    samples = load_all_samples(raw_path)
    tr, va, te = stratified_split(samples, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42)
    export_splits(tr, va, te, out_path)
    print("Dataset preparado exitosamente.")
