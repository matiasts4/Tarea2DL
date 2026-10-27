"""
train_yolo_scratch.py
---------------------
Entrenamiento de arquitectura YOLO entrenada DESDE CERO (Scratch).
- Inicialización con pesos aleatorios (pretrained=False, sin pesos de COCO).
- Utiliza la configuración arquitectónica de YOLOv8n (yolov8n.yaml).
- Mismas particiones exactas del dataset (fjordfish.yaml).
- Optimizador AdamW con Cosine Annealing, data augmentation submarino (hsv, fliplr, mosaic).
- Registra métricas de entrenamiento, mAP@50, mAP@50:95, velocidad y parámetros.
"""

import os
import time
import json
from ultralytics import YOLO

def main():
    print("=== Entrenamiento: Modelo 2 - YOLOv8n (Desde Cero / Scratch) ===")
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    yaml_dataset = os.path.join(base_dir, 'data', 'splits', 'fjordfish.yaml')
    project_dir = os.path.join(base_dir, 'results', 'models')
    run_name = 'yolo_scratch'
    
    # Inicializar modelo desde la estructura YAML sin pesos pre-entrenados
    model = YOLO('yolov8n.yaml')
    
    start_time = time.time()
    
    results = model.train(
        data=yaml_dataset,
        epochs=20,
        imgsz=640,
        batch=16,
        device=0,
        workers=2,
        project=project_dir,
        name=run_name,
        exist_ok=True,
        pretrained=False,     # ESTRICTAMENTE DESDE CERO
        optimizer='AdamW',
        lr0=1e-3,
        lrf=0.01,
        cos_lr=True,
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.4,
        fliplr=0.5,
        mosaic=0.5,
        seed=42,
        verbose=True
    )
    
    total_time = time.time() - start_time
    print(f"Entrenamiento YOLO Scratch completado en {total_time:.1f}s (~{total_time/60:.2f} min).")
    
    # Exportar resumen
    metrics_dir = os.path.join(base_dir, 'results', 'metrics')
    os.makedirs(metrics_dir, exist_ok=True)
    summary_path = os.path.join(metrics_dir, 'yolo_scratch_summary.json')
    
    best_weight_path = os.path.join(project_dir, run_name, 'weights', 'best.pt')
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump({
            'model_name': 'YOLOv8n (Desde Cero)',
            'pretrained': False,
            'epochs': 20,
            'total_time_seconds': total_time,
            'best_weight': best_weight_path
        }, f, indent=2)
    print(f"Resumen guardado en: {summary_path}")

if __name__ == '__main__':
    main()
