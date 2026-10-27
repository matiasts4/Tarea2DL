# Tarea 2 - Deep Learning: Detección de Objetos en Imágenes Submarinas

**Autores:** Matías Tobar & Sebastián Parada  

---

## 1. Descripción

Este repositorio implementa un estudio experimental reproducible para abordar la detección, localización espacial y clasificación multiclase de 28 especies biológicas marinas en el ecosistema de fiordos noruegos a partir del dataset **FjordFish** ([Zenodo: record/17950781](https://zenodo.org/records/17950781)).

El trabajo compara sistemáticamente cuatro detectores bajo dos paradigmas de entrenamiento:
1. **Entrenamiento desde Cero (Scratch):** Un detector convolucional propio multiescala con backbone residual y cuello FPN (`CustomCNNDetector`) frente a la arquitectura `YOLOv8n` inicializada con pesos aleatorios.
2. **Transferencia de Aprendizaje (Transfer Learning):** Adaptación y fine-tuning en dos fases de detectores preentrenados en COCO (`YOLOv8n` y `Faster R-CNN v2` con backbone `ResNet-50-FPN`).
3. **Robustez ante Degradaciones Submarinas:** Evaluación del detector óptimo ante cuatro perturbaciones físico-ópticas sintéticas (desenfoque óptico, atenuación lumínica espectral, dispersión/scattering y nieve marina con ruido de sensor) en tres niveles de severidad.

---

## 2. Protocolo experimental y Dataset

El conjunto de datos **FjordFish** contiene 3.002 imágenes submarinas de alta resolución y 13.799 anotaciones de organismos marinos pertenecientes a 28 clases taxonómicas con un marcado desbalance ecológico (*long-tail*).

Para asegurar la reproducibilidad determinista y evitar cualquier riesgo de fuga de información (*data leakage*):
* Se auditó el dataset crudo, reconciliando la inconsistencia de origen en la imagen `2515` (cuyo archivo de imagen y etiqueta estaban en carpetas dispares).
* Se diseñó un esquema de partición estratificado multi-etiqueta con semilla fija (`SEED = 42`):

```
Dataset FjordFish completo fijado (3.002 imágenes, 28 clases)
 └─ split estratificado multi-etiqueta 70 / 15 / 15
     ├─ train: 2.101 imágenes para optimización de pesos y gradientes
     ├─ validation: 450 imágenes para selección de checkpoints y early stopping
     └─ test: 451 imágenes, estrictamente bloqueado para evaluación final
```

* **Control de Data Leakage:** Las decisiones de modelado, parada temprana (*early stopping*) y ajuste de hiperparámetros se ejecutaron exclusivamente sobre `train` y `val`. El conjunto `test` permaneció sellado y se consultó una única vez por cada arquitectura ya entrenada.
* **Trazabilidad:** La correspondencia exacta de cada muestra se encuentra registrada en `data/splits/dataset_manifest.csv` y configurada en `data/splits/fjordfish.yaml`.

---

## 3. Modelos y evidencia generada

| Familia | Paradigma | Arquitectura y Estrategia | Evidencia y Artefactos |
| :--- | :--- | :--- | :--- |
| **M1: CNN Propia** | Desde Cero (Scratch) | Backbone residual 5 etapas, SPPF multiescala, FPN ($P_3, P_4, P_5$), cabezas desacopladas (7.17M params) | `models/custom_detector.py`<br>`results/metrics/custom_cnn_history.json` |
| **M2: YOLOv8n Scratch** | Desde Cero (Scratch) | Backbone CSPDarknet con bloques C2f, anchor-free, pérdida CIoU + DFL, aumento Mosaic (3.02M params) | `train_yolo_scratch.py`<br>`results/metrics/yolo_scratch_summary.json` |
| **M3: YOLOv8n Transfer** | Transfer Learning (COCO) | Protocolo 2 fases: Fase 1 (freeze=10, 10 épocas, lr=1e-3) + Fase 2 (descongelamiento total, lr=1e-4) | `train_yolo_transfer.py`<br>`results/metrics/yolo_transfer_summary.json` |
| **M4: Faster R-CNN v2** | Transfer Learning (COCO) | ResNet-50-FPN v2, FastRCNNPredictor 29 clases, freeze base -> fine-tuning layer3/layer4 con AMP | `train_faster_rcnn_transfer.py`<br>`results/metrics/faster_rcnn_history.json` |
| **Comparación Global** | Benchmark 4 Modelos | Evaluación estándar COCO mAP (envolvente 101 puntos), latencia GPU y frontera de Pareto | `results/metrics/model_comparison_summary.csv`<br>`results/plots/comparison_*.png` |
| **Robustez Submarina** | Estrés Físico Óptico | 4 perturbaciones en 3 niveles de severidad sobre el conjunto de test bloqueado (451 casos) | `results/metrics/degradation_robustness_summary.csv`<br>`results/plots/degradation_*.png` |

---

## 4. Resultados de la ejecución verificada

Todas las evaluaciones se realizaron sobre el mismo conjunto de prueba bloqueado de 451 imágenes utilizando el protocolo oficial de evaluación **COCO mAP** (integración trapezoidal de 101 puntos sobre la curva Precisión-Recall por cada una de las 28 clases):

| ID | Modelo | Paradigma | mAP@50 | mAP@50:95 | Precisión | Recall | F1-Score | Latencia (ms) | FPS | Parámetros | Tamaño Disco |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **M1** | **CNN Propia** | Desde Cero (Scratch) | 0.0398 | 0.0116 | 0.1364 | 0.1775 | 0.1542 | 10.48 ms | 95.4 | 7.17 M | 82.3 MB |
| **M2** | **YOLOv8n** | Desde Cero (Scratch) | 0.1228 | 0.0600 | 0.2930 | 0.1424 | 0.1917 | **5.91 ms** | **169.1** | **3.02 M** | **6.0 MB** |
| **M3** | **YOLOv8n** | Transfer Learning (COCO) | **0.2819** | **0.1942** | 0.2919 | 0.2987 | 0.2952 | 7.23 ms | 138.3 | 3.14 M | 6.2 MB |
| **M4** | **Faster R-CNN v2** | Transfer Learning (COCO) | 0.1276 | 0.0762 | **0.3081** | **0.4493** | **0.3656** | 66.76 ms | 15.0 | 43.39 M | 165.9 MB |

### Diagnóstico de la Frontera de Pareto
* **YOLOv8n Transfer Learning (Óptimo de Pareto):** Lidera en discriminación multiclase (`mAP@50 = 0.2819`) y velocidad operativa (`7.23 ms`, 138.3 FPS), requiriendo un peso de apenas 6.2 MB en disco. Es la solución óptima para despliegues en tiempo real a bordo de vehículos submarinos autónomos (AUV/ROV).
* **Faster R-CNN v2 (Cobertura Taxonómica):** Maximiza el Recall (`44.93%`) y el F1-Score (`0.3656`) al recuperar ejemplares ocultos o de difícil segmentación en especies mayoritarias gracias a su RPN desacoplado, resultando la arquitectura ideal para censos biológicos de laboratorio *offline*.
* **Impacto del Transfer Learning:** En la misma arquitectura YOLOv8n, la transferencia de pesos desde COCO eleva el mAP@50 desde `0.1228` hasta `0.2819` (**+129.6% relativo**) y triplica la localización fina en mAP@50:95 (`0.0600` a `0.1942`), mitigando el costo de escasez de datos.

---

## 5. Robustez ante degradaciones ambientales submarinas (Sección 3)

Se sometió al mejor modelo (**YOLOv8n Transfer Learning**, baseline mAP@50 = 0.2804) a perturbaciones físico-ópticas sintéticas sobre las 451 imágenes del conjunto de test:

| Condición Ambiental | Severidad | Parámetro Físico | mAP@50 | mAP@50:95 | Precisión | Recall | Caída Relativa (`mAP@50`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline Limpio** | Control | Sin perturbar | **0.2804** | **0.1923** | 0.6490 | 0.5396 | **0.00 %** |
| **Desenfoque Óptico / Turbidez** | Leve | $k=5$ | 0.2758 | 0.1879 | 0.6423 | 0.5389 | $-1.64 \%$ |
| | Moderado | $k=9$ | 0.2720 | 0.1846 | 0.6256 | 0.5256 | $-3.00 \%$ |
| | Severo | $k=15$ | 0.2651 | 0.1753 | 0.6149 | 0.5060 | **$-5.46 \%$** |
| **Atenuación Espectral (Luz)** | Leve | $R \times 0.55$ | 0.2581 | 0.1743 | 0.6213 | 0.5245 | $-7.95 \%$ |
| | Moderado | $R \times 0.30$ | 0.2386 | 0.1605 | 0.6227 | 0.4896 | $-14.91 \%$ |
| | Severo | $R \times 0.15$ | 0.1953 | 0.1303 | 0.5761 | 0.4400 | **$-30.35 \%$** |
| **Dispersión / Scattering** | Leve | $c=0.68$ | 0.2672 | 0.1817 | 0.6258 | 0.5179 | $-4.71 \%$ |
| | Moderado | $c=0.42$ | 0.2292 | 0.1558 | 0.6248 | 0.4408 | $-18.26\%$ |
| | Severo | $c=0.22$ | 0.1415 | 0.0970 | 0.5625 | 0.3056 | **$-49.54 \%$** |
| **Nieve Marina y Ruido Sensor** | Leve | $\sigma=8, 0.04\%$ part. | 0.2332 | 0.1566 | 0.6132 | 0.4648 | $-16.83\%$ |
| | Moderado | $\sigma=16, 0.10\%$ part. | 0.1592 | 0.1045 | 0.5280 | 0.3401 | $-43.22\%$ |
| | Severo | $\sigma=26, 0.22\%$ part. | 0.0877 | 0.0562 | 0.4596 | 0.2073 | **$-68.72 \%$** |

### Diagnóstico Físico del Colapso
* **Nieve Marina (-68.72%) y Dispersión Óptica (-49.54%):** Son los factores más críticos. Las partículas en suspensión inducen falsos positivos masivos y enmascaran texturas finas, mientras que el scattering destruye los gradientes locales de contraste ($\nabla I$) en las primeras capas convolucionales.
* **Resiliencia al Desenfoque (-5.46%):** El detector conserva su rendimiento al basarse en características contextuales multiescala ($P_4$ y $P_5$).
* **Recomendación:** Se sugiere incorporar filtros de mediana para supresión de partículas y preprocesamiento de contraste adaptativo (CLAHE) en el hardware frontal de adquisición.

---

## 6. Requisitos e instalación

El proyecto fue desarrollado y verificado en Python 3.11/3.12 y PyTorch con aceleración CUDA.

```bash
# 1. Crear y activar entorno virtual
python -m venv .venv
.\.venv\Scripts\activate   # En Windows
# source .venv/bin/activate  # En Linux/macOS

# 2. Instalar dependencias
pip install -r requirements.txt
```

---

## 7. Ejecución reproducible

El pipeline se encuentra modularizado para su ejecución secuencial completa:

```bash
# Paso 1: Preparación del dataset y particionado estratificado (seed=42)
python prepare_dataset.py

# Paso 2: Generación del análisis exploratorio de datos (EDA)
python eda_dataset.py

# Paso 3: Entrenamiento de los cuatro modelos
python train_custom_cnn.py
python train_yolo_scratch.py
python train_yolo_transfer.py
python train_faster_rcnn_transfer.py

# Paso 4: Evaluación unificada en el test set bloqueado (451 imágenes)
python evaluate_all_models.py

# Paso 5: Evaluación de robustez ante degradaciones submarinas
python evaluate_degradation.py
```

---

## 8. Estructura del repositorio

```text
T2DL/
├── data/
│   └── splits/                     # Particiones fijas y reproducibles (seed=42)
│       ├── dataset_manifest.csv    # Registro inmutable de imágenes, etiquetas y splits
│       └── fjordfish.yaml          # Definición de clases y rutas para Ultralytics
├── models/
│   ├── __init__.py                 # Paquete de modelos
│   ├── custom_detector.py          # Arquitectura convolucional propia (ResNet + SPPF + FPN)
│   └── utils.py                    # Módulo central: dataset, taxonomía, IoU y evaluador COCO mAP
├── results/
│   ├── eda/                        # Gráficos de análisis exploratorio (fondo blanco)
│   ├── metrics/                    # Tablas CSV y curvas de convergencia JSON
│   │   ├── model_comparison_summary.csv
│   │   └── degradation_robustness_summary.csv
│   └── plots/                      # Gráficos comparativos y curvas de degradación
│       ├── comparison_map_bars.png
│       ├── comparison_latency_vs_map.png
│       ├── comparison_training_convergence.png
│       ├── degradation_curves_map50.png
│       └── degraded_samples_grid.png
├── prepare_dataset.py              # Script de particionado estratificado
├── eda_dataset.py                  # Script de visualización y EDA
├── train_custom_cnn.py             # Entrenamiento Custom CNN Scratch
├── train_yolo_scratch.py           # Entrenamiento YOLOv8n Scratch
├── train_yolo_transfer.py          # Entrenamiento YOLOv8n Transfer Learning
├── train_faster_rcnn_transfer.py   # Entrenamiento Faster R-CNN v2 Transfer Learning
├── evaluate_all_models.py          # Evaluación consolidada en test set bloqueado
├── evaluate_degradation.py         # Evaluación de robustez ante perturbaciones ópticas
├── requirements.txt                # Dependencias del proyecto
└── README.md                       # Documentación principal
```

---

## 9. Declaración de uso de herramientas generativas

Durante el desarrollo de este trabajo se utilizaron herramientas de inteligencia artificial generativa como apoyo técnico para la estructuración de scripts de preprocesamiento, la configuración visual de los gráficos en Matplotlib y la redacción y formateo de la documentación.
