

import os
import sys
import random
from dataclasses import dataclass
from typing import Optional, List

import numpy as np
from PIL import Image
import matplotlib.cm as cm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import torchvision

from PyQt5.QtCore import Qt, QSize, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QPushButton, QTextEdit, QLabel,
    QFrame, QFileDialog, QVBoxLayout, QHBoxLayout, QGridLayout
)


MODEL_PATH = "E:/MSC_University Academics/Fall 2025/CSE754/Project_Remote_Sensing/Model/EarthVQA/All True/EarthVQA_ALL/best_model_earthvqa.pth"

IMG_ROOT_DIR = "F:/Remote Sensing/EArthVQA/Original_good_split/Val/images_png"
MASK_DIR = "F:/Remote Sensing/EArthVQA/Original_good_split/Val/masks_png"

NUM_CLASSES = 8           # labels 0..7 (0 ignored)
IGNORE_INDEX = 0
IMG_SIZE = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
AMP_ENABLED = (DEVICE == "cuda")

CLASS_NAMES = [ 
    "No-data (ignored)",  # 0
    "Background",          # 1
    "Building",           # 2
    "Road",                # 3
    "Water",              # 4
    "Barren",              # 5
    "Forest",              # 6
    "Agriculture",        # 7
]

COLOR_MAP = np.array([
    [0, 0, 0],         # 0 ignored
    [128, 0, 0],       # 1
    [0, 128, 0],       # 2
    [128, 128, 0],    # 3
    [0, 0, 128],       # 4
    [128, 0, 128],    # 5
    [0, 128, 128],     # 6
    [128, 128, 128],  # 7
], dtype=np.uint8)

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


BORDER_STYLE = """
QFrame {
    border: 2px solid #2b3a4a;
    background: #0f141a;
    border-radius: 14px;
}
"""

SLOT_STYLE = """
QLabel {
    border: 2px solid #2b3a4a;
    background: #141b22;
    color: #e8eef6;
    border-radius: 12px;
    font-size: 15px;
    font-weight: 600;
}
"""

SLOT_SELECTED_STYLE = """
QLabel {
    border: 3px solid #c7782a;
    background: #141b22;
    color: #e8eef6;
    border-radius: 12px;
    font-size: 15px;
    font-weight: 700;
}
"""

APP_STYLE = """
/* App base */
QMainWindow { background: #0b1016; }
QWidget { background: #0b1016; color: #e8eef6; font-family: Segoe UI; font-size: 14px; }

/* Buttons */
QPushButton {
    background: #18212b;
    color: #e8eef6;
    border: 1px solid #2b3a4a;
    border-radius: 10px;
    padding: 10px 14px;
    font-size: 15px;
    font-weight: 700;
}
QPushButton:hover { background: #1f2a36; border-color: #3b5066; }
QPushButton:pressed { background: #121923; }
QPushButton:disabled { color: #9aa7b5; background: #121923; border-color: #233140; }

/* Text box */
QTextEdit {
    background: #0f141a;
    color: #e8eef6;
    border: 1px solid #2b3a4a;
    border-radius: 12px;
    padding: 10px;
    font-size: 16px;      /* bigger */
    font-weight: 600;
}
QTextEdit QScrollBar:vertical {
    background: #0f141a; width: 12px; margin: 2px; border-radius: 6px;
}
QTextEdit QScrollBar::handle:vertical {
    background: #2b3a4a; min-height: 24px; border-radius: 6px;
}
QTextEdit QScrollBar::handle:vertical:hover { background: #3b5066; }
QTextEdit QScrollBar::add-line:vertical, QTextEdit QScrollBar::sub-line:vertical { height: 0px; }

/* Tooltips */
QToolTip {
    background: #141b22;
    color: #e8eef6;
    border: 1px solid #2b3a4a;
    padding: 6px;
    border-radius: 8px;
    font-size: 13px;
}
"""




class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, groups=1, use_dropout=False, p_drop=0.1):
        super().__init__()
        

        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, bias=False, groups=groups)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout2d(p_drop) if use_dropout else nn.Identity()
        nn.init.kaiming_normal_(self.conv.weight, mode="fan_out", nonlinearity="relu")
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.drop(x)
        return x

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, use_dropout=False, p_drop=0.1):
        super().__init__()
        self.depth = ConvBNReLU(in_ch, in_ch, k=3, s=1, p=1,
                                groups=in_ch, use_dropout=use_dropout, p_drop=p_drop)
        self.point = ConvBNReLU(in_ch, out_ch, k=1, s=1, p=0,
                                use_dropout=use_dropout, p_drop=p_drop)
    def forward(self, x):
        x = self.depth(x)
        x = self.point(x)
        return x

class SelfAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)
    
    def forward(self, x):
        b, c, h, w = x.shape
        n = h * w
        x_flat = x.permute(0, 2, 3, 1).reshape(b, n, c)
        x_norm = self.norm(x_flat)
        q, k, v = self.qkv(x_norm).chunk(3, dim=-1)
        q = q.view(b, n, self.num_heads, c // self.num_heads).transpose(1, 2)
        k = k.view(b, n, self.num_heads, c // self.num_heads).transpose(1, 2)
        v = v.view(b, n, self.num_heads, c // self.num_heads).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = attn @ v
        out = out.transpose(1, 2).reshape(b, n, c)
        out = self.drop(self.proj(out))
        out = out + x_flat
        out = out.reshape(b, h, w, c).permute(0, 3, 1, 2)
        return out

class GlobalContextBlock(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super().__init__()
        self.attention = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.transform = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, bias=False),
        )
    
    def forward(self, x):
        B, C, H, W = x.shape
        attn = self.attention(x)
        attn = attn.view(B, 1, -1)
        attn = torch.softmax(attn, dim=-1)
        attn = attn.view(B, 1, H, W)
        context = (x * attn).sum(dim=[2, 3], keepdim=True)
        out = self.transform(context)
        return x + out

class ConvNeXtTinyEncoder(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        try:
            backbone = torchvision.models.convnext_tiny(weights="DEFAULT" if pretrained else None)
        except TypeError:
            backbone = torchvision.models.convnext_tiny(pretrained=pretrained)
        self.stem = backbone.features[0]   # /4
        self.stage1 = backbone.features[1] # 1/4,  96C
        self.stage2 = backbone.features[2] # 1/8,  192C
        self.stage3 = backbone.features[3] # 1/16, 384C
        self.stage4 = backbone.features[4] # 1/32, 768C
    
    def forward(self, x):
        x = self.stem(x)
        e1 = self.stage1(x)      # [B,96,H/4,W/4]
        e2 = self.stage2(e1)     # [B,192,H/8,W/8]
        e3 = self.stage3(e2)     # [B,384,H/16,W/16]
        e4 = self.stage4(e3)     # [B,768,H/32,W/32]
        return e1, e2, e3, e4

class Decoder(nn.Module):
    def __init__(self, encoder_channels=(96, 192, 192, 384),
                 decoder_channels=(256, 96)):
        super().__init__()
        c1, c2, c3, c4 = encoder_channels
        d1, d2 = decoder_channels
        
 
        self.use_gc = True
        self.use_mhsa = True
        self.use_aux = True
        
        self.gc_e4 = GlobalContextBlock(c4)
        self.gc_e3 = GlobalContextBlock(c3)
        self.att_block = SelfAttentionBlock(c4, num_heads=8, dropout=0.1)
        self.up3 = nn.ConvTranspose2d(c4, c3, kernel_size=2, stride=2)
        self.dec3 = DepthwiseSeparableConv(c3 + c3 + c2, d1, use_dropout=True)  # 384+384+192=960 -> 256
        self.up2 = nn.ConvTranspose2d(d1, 128, kernel_size=2, stride=2)
        self.dec2 = DepthwiseSeparableConv(128 + c1, d2, use_dropout=True)  # 128+96=224 -> 96
        self.out_ch = d2
        self.aux_head_16 = nn.Conv2d(d1, NUM_CLASSES, kernel_size=1)
        self.aux_head_8  = nn.Conv2d(d2, NUM_CLASSES, kernel_size=1)
    
    def forward(self, e1, e2, e3, e4):
        if self.use_gc:
            e4 = self.gc_e4(e4)
            e3 = self.gc_e3(e3)
        if self.use_mhsa:
            e4 = self.att_block(e4)
        
        x = self.up3(e4)                    # 768->384
        x = torch.cat([x, e3, e2], dim=1)    # 384+384+192=960
        x = self.dec3(x)                    # 960->256
        aux_16 = self.aux_head_16(x) if self.use_aux else None
        
        x = self.up2(x)                       # 256->128
        x = torch.cat([x, e1], dim=1)       # 128+96=224
        x = self.dec2(x)                    # 224->96
        aux_8 = self.aux_head_8(x) if self.use_aux else None
        
        return x, aux_16, aux_8

class DetailBranch(nn.Module):
    def __init__(self, in_ch=3, out_ch=96):
        super().__init__()
        self.down = nn.Sequential(
            ConvBNReLU(in_ch, 32, k=3, s=2, p=1),   # 1/2
            ConvBNReLU(32, 64, k=3, s=2, p=1),      # 1/4
        )
        self.block = DepthwiseSeparableConv(64, out_ch, use_dropout=True)
    
    def forward(self, x):
        x = self.down(x)
        x = self.block(x)
        return x

class SegmentationModel(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.encoder = ConvNeXtTinyEncoder(pretrained=True)
        self.decoder = Decoder(
            encoder_channels=(96, 192, 192, 384),
            decoder_channels=(256, 96)
        )
        self.use_detail = True
        self.use_boundary = True
        self.detail_branch = DetailBranch(in_ch=3, out_ch=96)
        self.fuse = DepthwiseSeparableConv(self.decoder.out_ch + 96, 96, use_dropout=True)
        self.seg_head = nn.Conv2d(96, num_classes, kernel_size=1)
        self.boundary_head = nn.Conv2d(96, 1, kernel_size=1)
    
    def forward(self, x):
        e1, e2, e3, e4 = self.encoder(x)
        dec_1_4, aux_16, aux_8 = self.decoder(e1, e2, e3, e4)
        
        if self.use_detail:
            detail = self.detail_branch(x)
            fused = torch.cat([dec_1_4, detail], dim=1)
        else:
            fused = dec_1_4
        
        fused = self.fuse(fused)
        seg_logits_1_4 = self.seg_head(fused)
        
        if self.use_boundary:
            boundary_logits_1_4 = self.boundary_head(fused)
        else:
            boundary_logits_1_4 = None
        
        H, W = x.shape[2], x.shape[3]
        seg_logits = F.interpolate(seg_logits_1_4, size=(H, W), mode="bilinear")
        
        if boundary_logits_1_4 is not None:
            boundary_logits = F.interpolate(boundary_logits_1_4, size=(H, W), mode="bilinear")
        else:
            boundary_logits = None
        
        if aux_16 is not None:
            aux_16 = F.interpolate(aux_16, size=(H, W), mode="bilinear")
        if aux_8 is not None:
            aux_8 = F.interpolate(aux_8, size=(H, W), mode="bilinear")
        
        return {
            "logits": seg_logits,
            "aux_16": aux_16,
            "aux_8": aux_8,
            "boundary_logits": boundary_logits
        }


def load_model(model_path: str) -> nn.Module:
    model = SegmentationModel(num_classes=NUM_CLASSES)

    ckpt = torch.load(model_path, map_location=DEVICE)

    if isinstance(ckpt, dict) and "model_state" in ckpt:
        state = ckpt["model_state"]
    else:
        state = ckpt

    state = {k: v for k, v in state.items() if not k.endswith("total_ops") and not k.endswith("total_params")}

    model.load_state_dict(state, strict=True)
    model.to(DEVICE)
    model.eval()
    return model


def decode_segmap(mask_np: np.ndarray) -> np.ndarray:
    h, w = mask_np.shape
    rgb = COLOR_MAP[mask_np.flatten()].reshape(h, w, 3)
    return rgb

def compute_image_miou(gt: np.ndarray, pred: np.ndarray, num_classes: int = NUM_CLASSES, ignore_index: int = IGNORE_INDEX):
    """
    Per-image mean IoU over classes 1..(num_classes-1), excluding ignore_index.
    Only averages classes that have union > 0 for that image (standard practice).
    Returns: (miou_float_or_nan, per_class_iou_dict)
    """
    assert gt.shape == pred.shape, "GT and Pred must have same shape"
    valid_classes = [c for c in range(num_classes) if c != ignore_index]

    per_class = {}
    ious = []
    for c in valid_classes:
        gt_c = (gt == c)
        pr_c = (pred == c)
        inter = np.logical_and(gt_c, pr_c).sum()
        union = np.logical_or(gt_c, pr_c).sum()
        if union == 0:
            continue  
        iou = inter / (union + 1e-8)
        per_class[c] = float(iou)
        ious.append(iou)

    if len(ious) == 0:
        return float("nan"), per_class

    return float(np.mean(ious)), per_class


def load_ground_truth(gt_path: str, image_size_wh) -> Optional[np.ndarray]:

    if not os.path.exists(gt_path):
        return None
    gt = Image.open(gt_path).convert("L")
    gt = gt.resize(image_size_wh, Image.NEAREST)
    gt_np = np.array(gt, dtype=np.int64)

    gt_np[gt_np == 8] = 1
    gt_np[gt_np == 0] = 0
    return gt_np

def preprocess_image_pil(img: Image.Image) -> torch.Tensor:
    tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])
    return tf(img).unsqueeze(0)

def pil_to_qpixmap(pil_img: Image.Image) -> QPixmap:
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    arr = np.array(pil_img)
    h, w, c = arr.shape
    qimg = QImage(arr.data, w, h, 3 * w, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())

def np_rgb_to_qpixmap(rgb: np.ndarray) -> QPixmap:
    rgb = np.ascontiguousarray(rgb)
    h, w, c = rgb.shape
    qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())

def overlay_heatmap_on_image(rgb_img: np.ndarray, cam_01: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """
    rgb_img: uint8 [H,W,3]
    cam_01:  float  [H,W] in [0,1]
    """
    cam_01 = np.clip(cam_01, 0, 1)


    heat = cm.get_cmap("jet")(cam_01)[:, :, :3]           # float [H,W,3] in [0,1]
    heat = (heat * 255).astype(np.uint8)

    out = (1 - alpha) * rgb_img.astype(np.float32) + alpha * heat.astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)



class GradCAM:

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.h1 = target_layer.register_forward_hook(self._fwd_hook)
        self.h2 = target_layer.register_full_backward_hook(self._bwd_hook)

    def _fwd_hook(self, module, inp, out):
        self.activations = out

    def _bwd_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]

    def close(self):
        self.h1.remove()
        self.h2.remove()

    def __call__(self, x: torch.Tensor, class_id: int) -> np.ndarray:
        self.model.zero_grad(set_to_none=True)
        out = self.model(x)
        logits = out["logits"] if isinstance(out, dict) else out
        score = logits[:, class_id, :, :].mean()


        score.backward(retain_graph=False)

        acts = self.activations           # [B,C,h,w]
        grads = self.gradients           # [B,C,h,w]
        if acts is None or grads is None:
            raise RuntimeError("GradCAM hooks did not capture activations/gradients. Choose another target layer.")

        weights = grads.mean(dim=(2, 3), keepdim=True)     # [B,C,1,1]
        cam = (weights * acts).sum(dim=1, keepdim=True)     # [B,1,h,w]
        cam = F.relu(cam)

        cam = F.interpolate(cam, size=(x.shape[2], x.shape[3]), mode="bilinear", align_corners=False)
        cam = cam[0, 0].detach().cpu().numpy()
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam



@dataclass
class InferenceResult:
    orig_rgb: np.ndarray
    gt_rgb: Optional[np.ndarray]
    pred_rgb: np.ndarray
    gradcam_overlays: List[np.ndarray]
    gradcam_labels: List[str]
    status_text: str


class InferenceWorker(QThread):
    done = pyqtSignal(object)   # emits InferenceResult
    fail = pyqtSignal(str)

    def __init__(self, model: nn.Module, image_path: str):
        super().__init__()
        self.model = model
        self.image_path = image_path

    def run(self):
        try:
            # Load image (orig size)
            img_pil = Image.open(self.image_path).convert("RGB")
            orig_wh = img_pil.size  # (W,H)
            orig_rgb = np.array(img_pil, dtype=np.uint8)

            # GT (same basename in MASK_DIR)
            base = os.path.basename(self.image_path)
            gt_path = os.path.join(MASK_DIR, base)
            gt_np = load_ground_truth(gt_path, orig_wh)
            gt_rgb = decode_segmap(gt_np).astype(np.uint8) if gt_np is not None else None

            # Inference tensor
            x = preprocess_image_pil(img_pil).to(DEVICE)

            with torch.no_grad():
                with torch.amp.autocast("cuda", enabled=AMP_ENABLED):
                    out = self.model(x)

                # out can be dict or tensor
                logits = out["logits"] if isinstance(out, dict) else out
                pred = torch.argmax(logits, dim=1)[0].detach().cpu().numpy().astype(np.uint8)

            pred_pil = Image.fromarray(pred)
            pred_pil = pred_pil.resize(orig_wh, Image.NEAREST)
            pred_np = np.array(pred_pil, dtype=np.uint8)
            pred_rgb = decode_segmap(pred_np).astype(np.uint8)
            if gt_np is not None:
                miou, _ = compute_image_miou(gt_np.astype(np.int64), pred_np.astype(np.int64),
                                            num_classes=NUM_CLASSES, ignore_index=IGNORE_INDEX)
            else:
                miou = float("nan")

            if hasattr(self.model, "fuse"):
                target_layer = self.model.fuse
            elif hasattr(self.model, "decoder") and hasattr(self.model.decoder, "dec2"):
                target_layer = self.model.decoder.dec2
            elif hasattr(self.model, "decoder") and hasattr(self.model.decoder, "dec3"):
                target_layer = self.model.decoder.dec3
            elif hasattr(self.model, "encoder") and hasattr(self.model.encoder, "stage4"):
                target_layer = self.model.encoder.stage4
            else:
                raise RuntimeError("No suitable target layer found for Grad-CAM.")

            cam_engine = GradCAM(self.model, target_layer)

            self.model.eval()

            x_cam = preprocess_image_pil(img_pil).to(DEVICE).float()
            x_cam.requires_grad_(True)

            classes = [cid for cid in range(1, 8) if cid != 5]
            k = min(4, len(classes))
            chosen = random.sample(classes, k=k)


            overlays = []
            labels = []
            for cid in chosen:
                            
                with torch.cuda.amp.autocast(enabled=False):
                    cam = cam_engine(x_cam, cid)

                # Resize cam to original
                cam_p = Image.fromarray((cam * 255).astype(np.uint8)).resize(orig_wh, Image.BILINEAR)
                cam_01 = np.array(cam_p, dtype=np.float32) / 255.0
                ov = overlay_heatmap_on_image(orig_rgb, cam_01)
                overlays.append(ov)
                labels.append(f"Grad-CAM: {cid} ({CLASS_NAMES[cid]})")

            cam_engine.close()

            if gt_np is None or not np.isfinite(miou):
                iou_line = "IoU: N/A (GT missing)"
            else:
                iou_line = f"IoU (mIoU): {miou*100:.2f}%"

            cam_line = "Grad-CAM classes:  " + "  |  ".join(
                [f"{cid}:{CLASS_NAMES[cid]} " for cid in chosen]
            )

            status = iou_line + "\n" + cam_line


            self.done.emit(InferenceResult(
                orig_rgb=orig_rgb,
                gt_rgb=gt_rgb,
                pred_rgb=pred_rgb,
                gradcam_overlays=overlays,
                gradcam_labels=labels,
                status_text=status
            ))
        except Exception as e:
            self.fail.emit(str(e))


class ImageSlot(QLabel):
    def __init__(self, title: str, slot_id: str, size: QSize):
        super().__init__(title)
        self.slot_id = slot_id
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(size)
        self.setStyleSheet(SLOT_STYLE)
        self.setScaledContents(False)
        self._pix = None

    def set_selected(self, selected: bool):
        self.setStyleSheet(SLOT_SELECTED_STYLE if selected else SLOT_STYLE)

    def set_pixmap(self, pix: QPixmap):
        self._pix = pix
        self._render_pixmap()

    def resizeEvent(self, event):
        self._render_pixmap()
        super().resizeEvent(event)

    def _render_pixmap(self):
        if self._pix is None or self._pix.isNull():
            return
        src = self._pix
        side = min(src.width(), src.height())
        x = (src.width() - side) // 2
        y = (src.height() - side) // 2
        cropped = src.copy(x, y, side, side)

        size = min(self.width(), self.height())
        scaled = cropped.scaled(size, size, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
        self.setPixmap(scaled)
        self.setText("")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setStyleSheet(APP_STYLE)
        self.setWindowTitle("EarthVQA Viewer (Original / GT / Pred + Grad-CAM)")
        self.setMinimumSize(1180, 720)
        self.resize(1280, 760)

        self.model = None
        self.worker = None

        central = QWidget()
        self.setCentralWidget(central)

        outer = QFrame()
        outer.setStyleSheet(BORDER_STYLE)

        outer_layout = QVBoxLayout(outer)
        outer_layout.setContentsMargins(25, 25, 25, 25)
        outer_layout.setSpacing(18)

        top_row = QHBoxLayout()
        top_row.setSpacing(18)

        self.btn1 = QPushButton("Button 1 (Load Image)")
        self.btn1.setFixedSize(220, 46)
        self.btn1.clicked.connect(self.on_load_image)

        self.textbox = QTextEdit()
        self.textbox.setReadOnly(True)
        self.textbox.setFixedHeight(85)
        self.textbox.setPlaceholderText("Status...")

        top_row.addWidget(self.btn1, 0, Qt.AlignLeft | Qt.AlignTop)
        top_row.addWidget(self.textbox, 1)
        outer_layout.addLayout(top_row)

        # Image grid
        grid = QGridLayout()
        grid.setHorizontalSpacing(40)
        grid.setVerticalSpacing(25)

        big = QSize(300, 250)
        small = QSize(230, 190)

        self.slot_orig = ImageSlot("Original", "orig", big)
        self.slot_gt = ImageSlot("GT", "gt", big)
        self.slot_pred = ImageSlot("Prediction", "pred", big)

        grid.addWidget(self.slot_orig, 0, 0, Qt.AlignCenter)
        grid.addWidget(self.slot_gt,   0, 1, Qt.AlignCenter)
        grid.addWidget(self.slot_pred, 0, 2, Qt.AlignCenter)

        self.slot_cam1 = ImageSlot("Grad-CAM 1", "cam1", small)
        self.slot_cam2 = ImageSlot("Grad-CAM 2", "cam2", small)
        self.slot_cam3 = ImageSlot("Grad-CAM 3", "cam3", small)
        self.slot_cam4 = ImageSlot("Grad-CAM 4", "cam4", small)

        bottom_row = QHBoxLayout()
        bottom_row.setSpacing(28)
        bottom_row.addWidget(self.slot_cam1)
        bottom_row.addWidget(self.slot_cam2)
        bottom_row.addWidget(self.slot_cam3)
        bottom_row.addWidget(self.slot_cam4)

        grid.addLayout(bottom_row, 1, 0, 1, 3, Qt.AlignHCenter)

        outer_layout.addLayout(grid)

        root = QVBoxLayout(central)
        root.setContentsMargins(30, 25, 30, 25)
        root.addWidget(outer)

        # Load model once at startup
        self._load_model_once()

    def _load_model_once(self):
        try:
            self.textbox.setPlainText(f"Loading model...\n{MODEL_PATH}")
            QApplication.processEvents()
            self.model = load_model(MODEL_PATH)
            self.textbox.setPlainText(f"Model loaded.\nDevice: {DEVICE}\nReady.")
        except Exception as e:
            self.model = None
            self.textbox.setPlainText(f"Model load FAILED:\n{e}")

    def on_load_image(self):
        if self.model is None:
            self.textbox.setPlainText("Model is not loaded. Fix MODEL_PATH.")
            return

        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select an image",
            IMG_ROOT_DIR if os.path.isdir(IMG_ROOT_DIR) else "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp);;All Files (*)"
        )
        if not path:
            return

        self.textbox.setPlainText("Running inference + Grad-CAM...")
        self.btn1.setEnabled(False)

        self.worker = InferenceWorker(self.model, path)
        self.worker.done.connect(self._on_result)
        self.worker.fail.connect(self._on_fail)
        self.worker.start()

    def _on_result(self, res: InferenceResult):

        self.slot_orig.set_pixmap(np_rgb_to_qpixmap(res.orig_rgb))
        self.slot_pred.set_pixmap(np_rgb_to_qpixmap(res.pred_rgb))
        if res.gt_rgb is not None:
            self.slot_gt.set_pixmap(np_rgb_to_qpixmap(res.gt_rgb))
        else:
            self.slot_gt.setPixmap(QPixmap())
            self.slot_gt.setText("GT Missing")

        cams = res.gradcam_overlays
        labels = res.gradcam_labels

        self.slot_cam1.set_pixmap(np_rgb_to_qpixmap(cams[0])); self.slot_cam1.setToolTip(labels[0]); self.slot_cam1.setText("")
        self.slot_cam2.set_pixmap(np_rgb_to_qpixmap(cams[1])); self.slot_cam2.setToolTip(labels[1]); self.slot_cam2.setText("")
        self.slot_cam3.set_pixmap(np_rgb_to_qpixmap(cams[2])); self.slot_cam3.setToolTip(labels[2]); self.slot_cam3.setText("")
        self.slot_cam4.set_pixmap(np_rgb_to_qpixmap(cams[3])); self.slot_cam4.setToolTip(labels[3]); self.slot_cam4.setText("")

     
        self.textbox.setPlainText(res.status_text)


        self.btn1.setEnabled(True)
        self.worker = None

    def _on_fail(self, msg: str):
        self.textbox.setPlainText(f"FAILED:\n{msg}")
        self.btn1.setEnabled(True)
        self.worker = None


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())
