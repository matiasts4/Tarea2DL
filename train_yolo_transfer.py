"""
train_yolo_transfer.py
----------------------
Entrenamiento de YOLOv8n utilizando Transfer Learning desde COCO:
- Fase 1 (Backbone Congelado - 10 épocas): Se congelan las primeras 10 capas del modelo (Backbone extractor de COCO)
  y se entrenan únicamente las cabezas detectoras y cuello multiescala (Neck) con lr0=1e-3.
- Fase 2 (Fine-Tuning - 10 épocas): Se descongelan todas las capas del modelo y se realiza un ajuste fino
  con una tasa de aprendizaje reducida (lr0=1e-4) para adaptar las representaciones profundas al entorno submarino
  sin sufrir olvido catastrófico (catastrophic forgetting).
Total: 20 épocas comparables de forma equitativa con el resto de modelos.
"""

import os
import time
import json
from ultralytics import YOLO

def main():
    print("=== Entrenamiento: Modelo 3 - YOLOv8n (Transfer Learning COCO + Fine-Tuning) ===")
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    yaml_dataset = os.path.join(base_dir, 'data', 'splits', 'fjordfish.yaml')
    project_dir = os.path.join(base_dir, 'results', 'models')
    run_name = 'yolo_transfer'
    
    # 1. Cargar modelo pre-entrenado en COCO
    print("Cargando pesos pre-entrenados oficiales de COCO (yolov8n.pt)...")
    coco_weights = os.path.join(base_dir, 'yolov8n.pt')
    if not os.path.exists(coco_weights):
        coco_weights = 'yolov8n.pt'
    model = YOLO(coco_weights)
    
    start_total_time = time.time()
    
    # FASE 1: Congelamiento de Backbone (Capas 0 a 9)
    print("\n--- FASE 1: Entrenamiento con Backbone Congelado (freeze=10) ---")
    results_phase1 = model.train(
        data=yaml_dataset,
        epochs=10,
        imgsz=640,
        batch=16,
        device=0,
        workers=2,
        project=project_dir,
        name=f"{run_name}_phase1",
        exist_ok=True,
        freeze=10,          # Congela las 10 capas del Backbone
        optimizer='AdamW',
        lr0=1e-3,
        cos_lr=True,
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.4,
        fliplr=0.5,
        mosaic=0.5,
        seed=42,
        verbose=True
    )
    
    # FASE 2: Fine-Tuning de la red completa con LR reducido
    print("\n--- FASE 2: Fine-Tuning de Red Completa Descongelada (freeze=0, lr0=1e-4) ---")
    best_phase1_weights = os.path.join(project_dir, f"{run_name}_phase1", 'weights', 'best.pt')
    model_ft = YOLO(best_phase1_weights)
    
    results_phase2 = model_ft.train(
        data=yaml_dataset,
        epochs=10,
        imgsz=640,
        batch=16,
        device=0,
        workers=2,
        project=project_dir,
        name=run_name,
        exist_ok=True,
        freeze=0,           # Descongelamiento completo para fine-tuning
        optimizer='AdamW',
        lr0=1e-4,           # Tasa 10x menor para evitar olvido catastrófico
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
    
    total_time = time.time() - start_total_time
    print(f"\nTransfer Learning YOLO completado en {total_time:.1f}s (~{total_time/60:.2f} min).")
    
    metrics_dir = os.path.join(base_dir, 'results', 'metrics')
    os.makedirs(metrics_dir, exist_ok=True)
    summary_path = os.path.join(metrics_dir, 'yolo_transfer_summary.json')
    
    best_weight_path = os.path.join(project_dir, run_name, 'weights', 'best.pt')
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump({
            'model_name': 'YOLOv8n (Transfer Learning COCO)',
            'pretrained': True,
            'phase1_frozen_layers': 10,
            'phase1_epochs': 10,
            'phase2_fine_tuning_epochs': 10,
            'total_time_seconds': total_time,
            'best_weight': best_weight_path
        }, f, indent=2)
    print(f"Resumen guardado en: {summary_path}")

if __name__ == '__main__':
    main()
