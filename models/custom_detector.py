"""
models/custom_detector.py
-------------------------
Arquitectura de Detección Convolucional Propia (Custom CNN Detector)
Diseñada desde cero para detección multiescala de organismos marinos.

Componentes:
1. Backbone: Deep Residual ConvNet con activaciones SiLU y bloques Conv-BN-Act (5 etapas).
2. Neck: Feature Pyramid multiescala (P3, P4, P5) con fusión lateral top-down.
3. Head: Cabezas desacopladas de Regresión (4 coordenadas), Objetividad (1) y Clasificación (28 clases).
4. Decodificación consistente con el entrenamiento:
   - Objetividad: Sigmoid (entrenada con BCEWithLogitsLoss ponderado).
   - Clasificación: Sigmoid (entrenada con BCEWithLogitsLoss).
   - Regresión: Offsets [dx, dy, bw, bh] acotados en [0, 1] (entrenados con SmoothL1Loss).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as ops


class ConvBlock(nn.Module):
    """Bloque Convolucional estándar: Conv2d + BatchNorm2d + SiLU."""
    def __init__(self, in_c, out_c, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=k, stride=s, padding=p, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class ResBlock(nn.Module):
    """Bloque Residual con cuello de botella y conexión de atajo directo."""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = ConvBlock(channels, channels // 2, k=1, s=1, p=0)
        self.conv2 = ConvBlock(channels // 2, channels, k=3, s=1, p=1)

    def forward(self, x):
        return x + self.conv2(self.conv1(x))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast: captura contexto de campo receptivo multiescala."""
    def __init__(self, in_c, out_c, k=5):
        super().__init__()
        c_ = in_c // 2
        self.cv1 = ConvBlock(in_c, c_, 1, 1, 0)
        self.cv2 = ConvBlock(c_ * 4, out_c, 1, 1, 0)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.m(y2)
        return self.cv2(torch.cat((x, y1, y2, y3), 1))


class CustomUnderwaterBackbone(nn.Module):
    """
    Backbone convolucional propio de 5 etapas para extracción jerárquica de características.
    Salidas en strides 8 (C3), 16 (C4) y 32 (C5).
    """
    def __init__(self, in_channels=3, base_channels=32):
        super().__init__()
        self.stage1 = ConvBlock(in_channels, base_channels, k=3, s=2, p=1)
        self.stage2 = nn.Sequential(
            ConvBlock(base_channels, base_channels * 2, k=3, s=2, p=1),
            ResBlock(base_channels * 2)
        )
        self.stage3 = nn.Sequential(
            ConvBlock(base_channels * 2, base_channels * 4, k=3, s=2, p=1),
            ResBlock(base_channels * 4),
            ResBlock(base_channels * 4)
        )
        self.stage4 = nn.Sequential(
            ConvBlock(base_channels * 4, base_channels * 8, k=3, s=2, p=1),
            ResBlock(base_channels * 8),
            ResBlock(base_channels * 8)
        )
        self.stage5 = nn.Sequential(
            ConvBlock(base_channels * 8, base_channels * 16, k=3, s=2, p=1),
            ResBlock(base_channels * 16),
            SPPF(base_channels * 16, base_channels * 16)
        )

    def forward(self, x):
        c1 = self.stage1(x)
        c2 = self.stage2(c1)
        c3 = self.stage3(c2)  # stride 8
        c4 = self.stage4(c3)  # stride 16
        c5 = self.stage5(c4)  # stride 32
        return c3, c4, c5


class UnderwaterFPN(nn.Module):
    """
    Cuello (Neck) de pirámide de características top-down con conexiones laterales.
    """
    def __init__(self, channels=(128, 256, 512), out_channels=128):
        super().__init__()
        c3, c4, c5 = channels
        self.lat5 = ConvBlock(c5, out_channels, k=1, s=1, p=0)
        self.lat4 = ConvBlock(c4, out_channels, k=1, s=1, p=0)
        self.lat3 = ConvBlock(c3, out_channels, k=1, s=1, p=0)

        self.smooth5 = ConvBlock(out_channels, out_channels, k=3, s=1, p=1)
        self.smooth4 = ConvBlock(out_channels, out_channels, k=3, s=1, p=1)
        self.smooth3 = ConvBlock(out_channels, out_channels, k=3, s=1, p=1)

    def forward(self, c3, c4, c5):
        p5 = self.lat5(c5)
        p4 = self.lat4(c4) + F.interpolate(p5, size=c4.shape[-2:], mode='nearest')
        p3 = self.lat3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode='nearest')

        p3 = self.smooth3(p3)
        p4 = self.smooth4(p4)
        p5 = self.smooth5(p5)
        return p3, p4, p5


class DetectionHead(nn.Module):
    """
    Cabeza desacoplada por escala:
      - 4 canales para regresión espacial [dx, dy, bw, bh]
      - 1 canal para objetividad (objectness)
      - num_classes canales para clasificación taxonómica
    """
    def __init__(self, in_c, num_classes=28):
        super().__init__()
        self.num_classes = num_classes

        self.reg_conv = nn.Sequential(
            ConvBlock(in_c, in_c, 3, 1, 1),
            ConvBlock(in_c, in_c, 3, 1, 1),
            nn.Conv2d(in_c, 4, 1)
        )
        self.obj_conv = nn.Sequential(
            ConvBlock(in_c, in_c, 3, 1, 1),
            nn.Conv2d(in_c, 1, 1)
        )
        self.cls_conv = nn.Sequential(
            ConvBlock(in_c, in_c, 3, 1, 1),
            ConvBlock(in_c, in_c, 3, 1, 1),
            nn.Conv2d(in_c, num_classes, 1)
        )

    def forward(self, x):
        reg = self.reg_conv(x)  # (B, 4, H, W)
        obj = self.obj_conv(x)  # (B, 1, H, W)
        cls = self.cls_conv(x)  # (B, num_classes, H, W)
        return reg, obj, cls


class CustomCNNDetector(nn.Module):
    """
    Detector completo de objetos CNN propio para FjordFish.
    Entrenado 100% desde cero con inicialización Kaiming He.
    """
    def __init__(self, num_classes=28, base_channels=32):
        super().__init__()
        self.num_classes = num_classes
        self.backbone = CustomUnderwaterBackbone(in_channels=3, base_channels=base_channels)
        self.neck = UnderwaterFPN(
            channels=(base_channels * 4, base_channels * 8, base_channels * 16),
            out_channels=128
        )
        self.head_p3 = DetectionHead(128, num_classes)
        self.head_p4 = DetectionHead(128, num_classes)
        self.head_p5 = DetectionHead(128, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        c3, c4, c5 = self.backbone(x)
        p3, p4, p5 = self.neck(c3, c4, c5)

        out_p3 = self.head_p3(p3)
        out_p4 = self.head_p4(p4)
        out_p5 = self.head_p5(p5)
        return [out_p3, out_p4, out_p5]

    @torch.no_grad()
    def predict(self, x, conf_thresh=0.20, iou_thresh=0.45, max_det=100):
        """
        Decodifica predicciones multiescala (P3, P4, P5) de forma 100% consistente
        con la función de pérdida de entrenamiento:
          - obj_prob = sigmoid(obj)
          - cls_prob = sigmoid(cls)  (coherente con BCEWithLogitsLoss)
          - dx, dy = clamp(reg[0:2], 0, 1)  (coherente con SmoothL1Loss lineal en [0, 1])
          - bw, bh = clamp(reg[2:4], 0.01, 1.0)
        Aplica Batched NMS por clase y limita a max_det detecciones por imagen.
        """
        self.eval()
        outputs = self(x)
        B = x.shape[0]
        results = []

        for b in range(B):
            all_boxes = []
            all_scores = []
            all_labels = []

            for reg, obj, cls in outputs:
                _, _, H, W = reg.shape

                obj_prob = torch.sigmoid(obj[b, 0])          # (H, W)
                cls_prob = torch.sigmoid(cls[b])             # (C, H, W) coherente con BCEWithLogitsLoss

                combined_scores = obj_prob.unsqueeze(0) * cls_prob
                max_scores, max_classes = torch.max(combined_scores, dim=0)

                mask = max_scores > conf_thresh
                if not mask.any():
                    continue

                gy, gx = torch.where(mask)
                scores_sel = max_scores[gy, gx]
                classes_sel = max_classes[gy, gx]

                # Decodificar coordenadas coherentes con reg_target = [cx*W - gx, cy*H - gy, bw, bh]
                dx = torch.clamp(reg[b, 0, gy, gx], min=0.0, max=1.0)
                dy = torch.clamp(reg[b, 1, gy, gx], min=0.0, max=1.0)
                bw = torch.clamp(reg[b, 2, gy, gx], min=0.01, max=1.0)
                bh = torch.clamp(reg[b, 3, gy, gx], min=0.01, max=1.0)

                cx = (gx.float() + dx) / float(W)
                cy = (gy.float() + dy) / float(H)

                x1 = torch.clamp(cx - bw / 2.0, min=0.0, max=1.0)
                y1 = torch.clamp(cy - bh / 2.0, min=0.0, max=1.0)
                x2 = torch.clamp(cx + bw / 2.0, min=0.0, max=1.0)
                y2 = torch.clamp(cy + bh / 2.0, min=0.0, max=1.0)

                boxes = torch.stack([x1, y1, x2, y2], dim=-1)
                all_boxes.append(boxes)
                all_scores.append(scores_sel)
                all_labels.append(classes_sel)

            if all_boxes:
                all_boxes = torch.cat(all_boxes, dim=0)
                all_scores = torch.cat(all_scores, dim=0)
                all_labels = torch.cat(all_labels, dim=0)

                keep = ops.batched_nms(all_boxes, all_scores, all_labels, iou_thresh)
                keep = keep[:max_det]
                results.append({
                    'boxes': all_boxes[keep],
                    'scores': all_scores[keep],
                    'labels': all_labels[keep]
                })
            else:
                device = x.device
                results.append({
                    'boxes': torch.zeros((0, 4), device=device),
                    'scores': torch.zeros((0,), device=device),
                    'labels': torch.zeros((0,), dtype=torch.long, device=device)
                })

        return results


UnderwaterBackbone = CustomUnderwaterBackbone
