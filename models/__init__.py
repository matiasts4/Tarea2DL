"""
Paquete models: Arquitecturas de detección y utilidades centralizadas.
"""
from models.custom_detector import CustomCNNDetector
from models.utils import (
    NUM_CLASSES,
    CLASS_NAMES,
    set_seed,
    apply_white_plot_style,
    load_yolo_labels,
    FjordFishDataset,
    compute_iou_matrix,
    evaluate_detections_coco,
)
