"""
models.py — Все модели для DVD Violence Detection в одном месте.

Все модели: [B, C, T, H, W] → [B, num_classes, T]

Бэкбоны:
  Оригинальные (из ноутбука):
    resnet18, efficientnet_b0_bilstm, video_swin_tiny, videomae_small

  Дополнительные (были в патчах скрипта):
    videomae_crime_init, slowfast_r50,
    r3d_18_temporal, r2plus1d_18_temporal, s3d_temporal

  V2 — с BiLSTM temporal head:
    r3d_18_temporal_v2, r2plus1d_18_temporal_v2, s3d_temporal_v2,
    slowfast_r50_v2, videomae_crime_init_v2

  V3 — новые архитектуры:
    convnext_tiny_transformer, efficientnet_b0_transformer  (Transformer head)
    convnext_tiny_multiscale, efficientnet_b0_multiscale    (Multi-scale features)
    convnext_tiny_crf, efficientnet_b0_crf                  (BiLSTM + CRF)
    convnext_tiny_skeleton                                  (Skeleton fusion)
    convnext_tiny_flow                                      (Optical flow dual-stream)

  V5 — Two-Stream Fusion (RGB + Optical Flow):
    twostream_concat_bilstm                                 (Concat fusion + BiLSTM)
    twostream_gated_bilstm                                  (Gated fusion + BiLSTM)
    twostream_attention_bilstm                               (Cross-attention fusion + BiLSTM)
    twostream_gated_tcn                                     (Gated fusion + TCN)
    twostream_attention_tcn                                  (Cross-attention fusion + TCN)

  Доп. компоненты:
    BoundaryAwareLoss — лосс с повышенным весом на границах violence

Использование:
    from models import build_model
    model = build_model("convnext_tiny_transformer", cfg=CFG)
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


# =====================================================================
#  Общий BiLSTM temporal head (используется V2 моделями)
# =====================================================================
class BiLSTMTemporalHead(nn.Module):
    """
    BiLSTM head для per-frame предсказаний.
    Вход: [B, feat_dim, T'] → Выход: [B, num_classes, T_original]
    """
    def __init__(self, feat_dim: int, num_classes: int = 2,
                 hidden_size: int = 256, num_layers: int = 2,
                 dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            feat_dim, hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.drop = nn.Dropout(dropout)
        self.proj = nn.Conv1d(hidden_size * 2, num_classes, kernel_size=1)

    def forward(self, feats: torch.Tensor, T_target: int) -> torch.Tensor:
        # Интерполируем к T ДО lstm (лучше чем после)
        if feats.shape[2] != T_target:
            feats = F.interpolate(feats, size=T_target, mode="linear", align_corners=False)
        x = feats.permute(0, 2, 1)      # [B, T, feat_dim]
        x, _ = self.lstm(x)              # [B, T, hidden*2]
        x = self.drop(x)
        x = x.permute(0, 2, 1)          # [B, hidden*2, T]
        return self.proj(x)              # [B, num_classes, T]


# =====================================================================
#  1) ResNet18 + temporal Conv1d
# =====================================================================
class ResNet18Temporal(nn.Module):
    def __init__(self, num_classes: int = 2, pretrained: bool = True,
                 dropout: float = 0.2, **kwargs):
        super().__init__()
        weights = torchvision.models.ResNet18_Weights.DEFAULT if pretrained else None
        base = torchvision.models.resnet18(weights=weights)
        self.backbone = nn.Sequential(*list(base.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Conv1d(512, num_classes, kernel_size=1)

    def forward(self, x):
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        f = self.backbone(x)
        f = self.pool(f).flatten(1)
        f = self.drop(f)
        f = f.view(B, T, 512).transpose(1, 2)
        return self.head(f)


# =====================================================================
#  2) EfficientNet-B0 + BiLSTM
# =====================================================================
class EfficientNetBiLSTM(nn.Module):
    def __init__(self, num_classes: int = 2, pretrained: bool = True,
                 hidden_size: int = 256, num_layers: int = 2,
                 dropout: float = 0.3, **kwargs):
        super().__init__()
        import timm
        self.encoder = timm.create_model(
            "efficientnet_b0", pretrained=pretrained,
            num_classes=0, global_pool="avg",
        )
        feat_dim = self.encoder.num_features
        self.lstm = nn.LSTM(
            feat_dim, hidden_size, num_layers=num_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        B, C, T, H, W = x.shape
        frames = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        feats = self.encoder(frames).view(B, T, -1)
        out, _ = self.lstm(feats)
        out = self.drop(out)
        logits = self.head(out)
        return logits.permute(0, 2, 1)


# =====================================================================
#  3) Video Swin Transformer Tiny
# =====================================================================
class VideoSwinTinyTemporal(nn.Module):
    def __init__(self, num_classes: int = 2, pretrained: bool = True,
                 dropout: float = 0.3, **kwargs):
        super().__init__()
        from torchvision.models.video import swin3d_t, Swin3D_T_Weights
        weights = Swin3D_T_Weights.KINETICS400_V1 if pretrained else None
        base = swin3d_t(weights=weights)

        self.patch_embed = base.patch_embed
        self.pos_drop = base.pos_drop
        self.features = base.features
        self.norm = base.norm

        feat_dim = base.head.in_features
        self.drop = nn.Dropout(dropout)
        self.cls_head = nn.Conv1d(feat_dim, num_classes, kernel_size=1)

    def forward(self, x):
        T_target = x.shape[2]
        x = self.patch_embed(x)
        x = self.pos_drop(x)
        x = self.features(x)
        x = self.norm(x)
        x = x.permute(0, 4, 1, 2, 3)
        x = x.mean(dim=(-2, -1))
        if x.shape[2] != T_target:
            x = F.interpolate(x, size=T_target, mode="linear", align_corners=False)
        x = self.drop(x)
        return self.cls_head(x)


# =====================================================================
#  4) VideoMAE Small
# =====================================================================
class VideoMAESmallTemporal(nn.Module):
    MODEL_NAME = "MCG-NJU/videomae-small-finetuned-kinetics"

    def __init__(self, num_classes: int = 2, pretrained: bool = True,
                 dropout: float = 0.3, **kwargs):
        super().__init__()
        from transformers import VideoMAEModel, VideoMAEConfig
        if pretrained:
            self.encoder = VideoMAEModel.from_pretrained(self.MODEL_NAME)
        else:
            cfg = VideoMAEConfig.from_pretrained(self.MODEL_NAME)
            self.encoder = VideoMAEModel(cfg)

        hidden_size = self.encoder.config.hidden_size
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, num_classes)
        self.temporal_stride = 2
        self.spatial_patches = (224 // 16) ** 2

    def forward(self, x):
        B, C, T, H, W = x.shape
        pixel_values = x.permute(0, 2, 1, 3, 4)
        outputs = self.encoder(pixel_values=pixel_values, bool_masked_pos=None)
        hidden = outputs.last_hidden_state

        T_t = T // self.temporal_stride
        S = self.spatial_patches
        hidden = hidden[:, :T_t * S, :]
        hidden = hidden.view(B, T_t, S, -1).mean(dim=2)

        hidden = hidden.permute(0, 2, 1)
        if T_t != T:
            hidden = F.interpolate(hidden, size=T, mode="linear", align_corners=False)
        hidden = hidden.permute(0, 2, 1)
        hidden = self.drop(hidden)
        logits = self.head(hidden)
        return logits.permute(0, 2, 1)


# =====================================================================
#  5) VideoMAE crime-init (original)
# =====================================================================
class VideoMAECrimeInitTemporal(nn.Module):
    DEFAULT_MODEL_NAME = "Nikeytas/videomae-crime-detector-production-v1"
    FALLBACK_MODEL_NAME = "MCG-NJU/videomae-small-finetuned-kinetics"

    def __init__(self, num_classes: int = 2, pretrained: bool = True,
                 dropout: float = 0.3, cfg: dict = None, **kwargs):
        super().__init__()
        cfg = cfg or {}
        from transformers import VideoMAEModel, VideoMAEConfig
        model_name = str(cfg.get("videomae_crime_model_name", self.DEFAULT_MODEL_NAME))
        if pretrained:
            try:
                self.encoder = VideoMAEModel.from_pretrained(model_name)
                print(f"[models] VideoMAE crime init loaded: {model_name}")
            except Exception as e:
                print(f"[models][warn] {model_name} failed: {e}, fallback → {self.FALLBACK_MODEL_NAME}")
                self.encoder = VideoMAEModel.from_pretrained(self.FALLBACK_MODEL_NAME)
        else:
            c = VideoMAEConfig.from_pretrained(model_name)
            self.encoder = VideoMAEModel(c)

        hidden_size = int(self.encoder.config.hidden_size)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, num_classes)

        if bool(cfg.get("videomae_crime_grad_checkpointing", True)):
            try:
                self.encoder.gradient_checkpointing_enable()
            except Exception:
                pass

        tubelet = getattr(self.encoder.config, "tubelet_size", 2)
        self.temporal_stride = max(1, int(tubelet[0] if isinstance(tubelet, (list, tuple)) else tubelet))
        patch_size = getattr(self.encoder.config, "patch_size", 16)
        self.patch_size = int(patch_size[0] if isinstance(patch_size, (list, tuple)) else patch_size)

    def _extract_temporal_feats(self, x):
        """Returns [B, hidden_size, T_t] after spatial pooling."""
        B, C, T, H, W = x.shape
        pixel_values = x.permute(0, 2, 1, 3, 4)
        outputs = self.encoder(pixel_values=pixel_values, bool_masked_pos=None)
        hidden = outputs.last_hidden_state

        T_t = max(1, T // self.temporal_stride)
        S_h = max(1, H // self.patch_size)
        S_w = max(1, W // self.patch_size)
        S = S_h * S_w
        need = T_t * S
        if hidden.shape[1] < need:
            S = max(1, hidden.shape[1] // T_t)
            need = T_t * S
        hidden = hidden[:, :need, :]
        hidden = hidden.view(B, T_t, S, -1).mean(dim=2)  # [B, T_t, hidden]
        return hidden.permute(0, 2, 1), T  # [B, hidden, T_t], T_original

    def forward(self, x):
        feats, T = self._extract_temporal_feats(x)
        if feats.shape[2] != T:
            feats = F.interpolate(feats, size=T, mode="linear", align_corners=False)
        hidden = feats.permute(0, 2, 1)
        hidden = self.drop(hidden)
        logits = self.head(hidden)
        return logits.permute(0, 2, 1)


# =====================================================================
#  5b) VideoMAE crime-init V2 (+ BiLSTM)
# =====================================================================
class VideoMAECrimeInitTemporalV2(VideoMAECrimeInitTemporal):
    """Наследуем энкодер, заменяем голову на BiLSTM."""
    def __init__(self, num_classes: int = 2, pretrained: bool = True,
                 dropout: float = 0.3, cfg: dict = None, **kwargs):
        # Инициализируем базовый энкодер (head будет перезаписан)
        super().__init__(num_classes=num_classes, pretrained=pretrained,
                         dropout=dropout, cfg=cfg)
        cfg = cfg or {}
        hidden_size = int(self.encoder.config.hidden_size)
        # Убираем старый Linear head, ставим BiLSTM
        del self.head
        del self.drop
        lstm_hidden = int(cfg.get("bilstm_hidden", 256))
        lstm_layers = int(cfg.get("bilstm_layers", 2))
        self.temporal_head = BiLSTMTemporalHead(
            hidden_size, num_classes, lstm_hidden, lstm_layers, dropout
        )

    def forward(self, x):
        feats, T = self._extract_temporal_feats(x)  # [B, hidden, T_t]
        return self.temporal_head(feats, T)


# =====================================================================
#  6) SlowFast-R50 (original)
# =====================================================================
def _load_slowfast_base(cfg: dict, pretrained: bool):
    """Загрузка SlowFast hub модели с fallback."""
    hub_repo = str(cfg.get("slowfast_hub_repo", "facebookresearch/pytorchvideo"))
    hub_model = str(cfg.get("slowfast_hub_model", "slowfast_r50"))
    use_pretrained = bool(pretrained) and bool(cfg.get("slowfast_use_pretrained", False))
    hub_local_dir = str(cfg.get(
        "slowfast_hub_local_dir",
        "/home/HDD6TB/hse_student/.cache/torch/hub/facebookresearch_pytorchvideo_main",
    ))
    try:
        if os.path.isdir(hub_local_dir):
            base = torch.hub.load(hub_local_dir, hub_model, pretrained=use_pretrained, source="local")
            print(f"[models] SlowFast loaded local, pretrained={use_pretrained}")
        else:
            raise FileNotFoundError(hub_local_dir)
    except Exception:
        try:
            base = torch.hub.load(hub_repo, hub_model, pretrained=use_pretrained)
            print(f"[models] SlowFast loaded remote, pretrained={use_pretrained}")
        except Exception:
            base = torch.hub.load(hub_repo, hub_model, pretrained=False)
            print("[models] SlowFast fallback pretrained=False")
    return base


class SlowFastR50Temporal(nn.Module):
    def __init__(self, num_classes: int = 2, pretrained: bool = True,
                 dropout: float = 0.3, cfg: dict = None, **kwargs):
        super().__init__()
        cfg = cfg or {}
        self.alpha = max(2, int(cfg.get("slowfast_alpha", 4)))
        self.fused_dim = int(cfg.get("slowfast_fused_dim", 2304))

        base = _load_slowfast_base(cfg, pretrained)
        self.blocks = nn.ModuleList(list(base.blocks)[:5])
        self.drop = nn.Dropout(dropout)
        self.head = nn.Conv1d(self.fused_dim, num_classes, kernel_size=1)

    def _pack_pathways(self, x):
        return [x[:, :, ::self.alpha, :, :], x]

    def _extract_feats(self, x):
        B, C, T, H, W = x.shape
        xs = self._pack_pathways(x)
        for b in self.blocks:
            xs = b(xs)
        feats = []
        for xi in xs:
            xi = xi.mean(dim=(-1, -2))
            if xi.shape[2] != T:
                xi = F.interpolate(xi, size=T, mode="linear", align_corners=False)
            feats.append(xi)
        fused = torch.cat(feats, dim=1)
        # Align dim
        if fused.shape[1] > self.fused_dim:
            fused = fused[:, :self.fused_dim, :]
        elif fused.shape[1] < self.fused_dim:
            pad = fused.new_zeros(B, self.fused_dim - fused.shape[1], fused.shape[2])
            fused = torch.cat([fused, pad], dim=1)
        return fused  # [B, fused_dim, T']

    def forward(self, x):
        feats = self._extract_feats(x)
        feats = self.drop(feats.transpose(1, 2)).transpose(1, 2)
        return self.head(feats)


# =====================================================================
#  6b) SlowFast-R50 V2 (+ BiLSTM, pretrained по умолчанию)
# =====================================================================
class SlowFastR50TemporalV2(nn.Module):
    def __init__(self, num_classes: int = 2, pretrained: bool = True,
                 dropout: float = 0.3, cfg: dict = None, **kwargs):
        super().__init__()
        cfg = cfg or {}
        # V2: pretrained=True по умолчанию!
        if "slowfast_use_pretrained" not in cfg:
            cfg = {**cfg, "slowfast_use_pretrained": True}
        self.alpha = max(2, int(cfg.get("slowfast_alpha", 4)))
        self.fused_dim = int(cfg.get("slowfast_fused_dim", 2304))

        base = _load_slowfast_base(cfg, pretrained)
        self.blocks = nn.ModuleList(list(base.blocks)[:5])

        lstm_hidden = int(cfg.get("bilstm_hidden", 256))
        lstm_layers = int(cfg.get("bilstm_layers", 2))
        self.temporal_head = BiLSTMTemporalHead(
            self.fused_dim, num_classes, lstm_hidden, lstm_layers, dropout
        )

    def _pack_pathways(self, x):
        return [x[:, :, ::self.alpha, :, :], x]

    def forward(self, x):
        B, C, T, H, W = x.shape
        xs = self._pack_pathways(x)
        for b in self.blocks:
            xs = b(xs)
        feats = []
        for xi in xs:
            xi = xi.mean(dim=(-1, -2))
            if xi.shape[2] != T:
                xi = F.interpolate(xi, size=T, mode="linear", align_corners=False)
            feats.append(xi)
        fused = torch.cat(feats, dim=1)
        if fused.shape[1] > self.fused_dim:
            fused = fused[:, :self.fused_dim, :]
        elif fused.shape[1] < self.fused_dim:
            pad = fused.new_zeros(B, self.fused_dim - fused.shape[1], fused.shape[2])
            fused = torch.cat([fused, pad], dim=1)
        return self.temporal_head(fused, T)


# =====================================================================
#  7) Torchvision 3D CNNs — r3d_18, r2plus1d_18, s3d (original)
# =====================================================================
class TorchvisionConv3DTemporal(nn.Module):
    def __init__(self, arch: str, num_classes: int = 2, pretrained: bool = True,
                 dropout: float = 0.3, **kwargs):
        super().__init__()
        import torchvision.models.video as tvv
        self.arch = str(arch)
        weight_map = {
            "r3d_18": tvv.R3D_18_Weights.DEFAULT,
            "r2plus1d_18": tvv.R2Plus1D_18_Weights.DEFAULT,
            "s3d": tvv.S3D_Weights.DEFAULT,
        }
        ctor = getattr(tvv, self.arch)
        weights = weight_map.get(self.arch) if pretrained else None
        try:
            base = ctor(weights=weights)
        except Exception:
            base = ctor(weights=None)

        if self.arch == "s3d":
            self.features = base.features
            self.feat_dim = 1024
        else:
            self.stem = base.stem
            self.layer1 = base.layer1
            self.layer2 = base.layer2
            self.layer3 = base.layer3
            self.layer4 = base.layer4
            self.feat_dim = 512

        self.drop = nn.Dropout(dropout)
        self.head = nn.Conv1d(self.feat_dim, num_classes, kernel_size=1)

    def _extract_feats(self, x):
        if self.arch == "s3d":
            y = self.features(x)
        else:
            y = self.stem(x)
            y = self.layer1(y)
            y = self.layer2(y)
            y = self.layer3(y)
            y = self.layer4(y)
        return y.mean(dim=(-1, -2))  # [B, feat_dim, T']

    def forward(self, x):
        B, C, T, H, W = x.shape
        y = self._extract_feats(x)
        if y.shape[2] != T:
            y = F.interpolate(y, size=T, mode="linear", align_corners=False)
        y = self.drop(y.transpose(1, 2)).transpose(1, 2)
        return self.head(y)


# =====================================================================
#  7b) Torchvision 3D CNNs V2 (+ BiLSTM)
# =====================================================================
class TorchvisionConv3DTemporalV2(nn.Module):
    def __init__(self, arch: str, num_classes: int = 2, pretrained: bool = True,
                 dropout: float = 0.3, cfg: dict = None, **kwargs):
        super().__init__()
        cfg = cfg or {}
        import torchvision.models.video as tvv
        self.arch = str(arch)
        weight_map = {
            "r3d_18": tvv.R3D_18_Weights.DEFAULT,
            "r2plus1d_18": tvv.R2Plus1D_18_Weights.DEFAULT,
            "s3d": tvv.S3D_Weights.DEFAULT,
        }
        ctor = getattr(tvv, self.arch)
        weights = weight_map.get(self.arch) if pretrained else None
        try:
            base = ctor(weights=weights)
            print(f"[models] {self.arch} loaded pretrained={weights is not None}")
        except Exception:
            base = ctor(weights=None)

        if self.arch == "s3d":
            self.features = base.features
            feat_dim = 1024
        else:
            self.stem = base.stem
            self.layer1 = base.layer1
            self.layer2 = base.layer2
            self.layer3 = base.layer3
            self.layer4 = base.layer4
            feat_dim = 512

        self.feat_dim = feat_dim
        lstm_hidden = int(cfg.get("bilstm_hidden", 256))
        lstm_layers = int(cfg.get("bilstm_layers", 2))
        self.temporal_head = BiLSTMTemporalHead(
            feat_dim, num_classes, lstm_hidden, lstm_layers, dropout
        )

    def _extract_feats(self, x):
        if self.arch == "s3d":
            y = self.features(x)
        else:
            y = self.stem(x)
            y = self.layer1(y)
            y = self.layer2(y)
            y = self.layer3(y)
            y = self.layer4(y)
        return y.mean(dim=(-1, -2))

    def forward(self, x):
        B, C, T, H, W = x.shape
        feats = self._extract_feats(x)  # [B, feat_dim, T']
        return self.temporal_head(feats, T)


# =====================================================================
#  8) Generic Timm 2D Backbone + BiLSTM
#     ConvNeXt-Tiny, ConvNeXt-Base, EfficientNet-B3, и любой другой timm
# =====================================================================
class TimmBackboneBiLSTM(nn.Module):
    """
    Любой 2D timm backbone + BiLSTM temporal head.
    Работает с: convnext_tiny, convnext_base, efficientnet_b3, и т.д.
    """
    def __init__(self, model_name: str, num_classes: int = 2, pretrained: bool = True,
                 hidden_size: int = 256, num_layers: int = 2,
                 dropout: float = 0.3, cfg: dict = None, **kwargs):
        super().__init__()
        cfg = cfg or {}
        import timm
        self.encoder = timm.create_model(
            model_name, pretrained=pretrained,
            num_classes=0, global_pool="avg",
        )
        feat_dim = self.encoder.num_features
        print(f"[models] TimmBackboneBiLSTM: {model_name}, feat_dim={feat_dim}")

        h = int(cfg.get("bilstm_hidden", hidden_size))
        nl = int(cfg.get("bilstm_layers", num_layers))
        self.lstm = nn.LSTM(
            feat_dim, h, num_layers=nl,
            batch_first=True, bidirectional=True,
            dropout=dropout if nl > 1 else 0.0,
        )
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(h * 2, num_classes)

    def forward(self, x):
        B, C, T, H, W = x.shape
        frames = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        feats = self.encoder(frames).view(B, T, -1)
        out, _ = self.lstm(feats)
        out = self.drop(out)
        logits = self.head(out)
        return logits.permute(0, 2, 1)


# =====================================================================
#  9) I3D R50 + BiLSTM (via pytorchvideo)
# =====================================================================
class I3DR50TemporalV2(nn.Module):
    """
    I3D ResNet-50 (Kinetics pretrained) + BiLSTM.
    Загружается через pytorchvideo hub.
    Больше capacity чем r3d_18 (2048 vs 512 features).
    """
    def __init__(self, num_classes: int = 2, pretrained: bool = True,
                 dropout: float = 0.3, cfg: dict = None, **kwargs):
        super().__init__()
        cfg = cfg or {}
        hub_local_dir = str(cfg.get(
            "slowfast_hub_local_dir",
            "/home/HDD6TB/hse_student/.cache/torch/hub/facebookresearch_pytorchvideo_main",
        ))
        hub_repo = "facebookresearch/pytorchvideo"

        try:
            if os.path.isdir(hub_local_dir):
                base = torch.hub.load(hub_local_dir, "i3d_r50", pretrained=pretrained, source="local")
            else:
                base = torch.hub.load(hub_repo, "i3d_r50", pretrained=pretrained)
            print(f"[models] I3D R50 loaded pretrained={pretrained}")
        except Exception as e:
            print(f"[models][warn] I3D R50 pretrained failed: {e}, fallback")
            try:
                base = torch.hub.load(hub_repo, "i3d_r50", pretrained=False)
            except Exception:
                if os.path.isdir(hub_local_dir):
                    base = torch.hub.load(hub_local_dir, "i3d_r50", pretrained=False, source="local")
                else:
                    raise

        self.blocks = nn.ModuleList(list(base.blocks)[:5])
        feat_dim = int(cfg.get("i3d_feat_dim", 2048))
        self.feat_dim = feat_dim

        lstm_hidden = int(cfg.get("bilstm_hidden", 256))
        lstm_layers = int(cfg.get("bilstm_layers", 2))
        self.temporal_head = BiLSTMTemporalHead(
            feat_dim, num_classes, lstm_hidden, lstm_layers, dropout
        )

    def forward(self, x):
        B, C, T, H, W = x.shape
        y = x
        for b in self.blocks:
            y = b(y)
        # y: [B, feat_dim, T', H', W']
        y = y.mean(dim=(-1, -2))  # [B, feat_dim, T']
        if y.shape[1] != self.feat_dim:
            # Adapt if dim mismatch
            y = y[:, :self.feat_dim, :] if y.shape[1] > self.feat_dim else F.pad(y, (0, 0, 0, self.feat_dim - y.shape[1]))
        return self.temporal_head(y, T)


# =====================================================================
#  10) VideoMAE Base + BiLSTM (larger than Small, hidden=768)
# =====================================================================
class VideoMAEBaseTemporal(nn.Module):
    """
    VideoMAE Base (768 hidden, ~86M params) vs Small (384 hidden, ~22M).
    Значительно мощнее. Тот же HuggingFace API.
    """
    MODEL_NAME = "MCG-NJU/videomae-base-finetuned-kinetics"

    def __init__(self, num_classes: int = 2, pretrained: bool = True,
                 dropout: float = 0.3, cfg: dict = None, **kwargs):
        super().__init__()
        cfg = cfg or {}
        from transformers import VideoMAEModel, VideoMAEConfig
        model_name = str(cfg.get("videomae_base_model_name", self.MODEL_NAME))

        if pretrained:
            try:
                self.encoder = VideoMAEModel.from_pretrained(model_name)
                print(f"[models] VideoMAE Base loaded: {model_name}")
            except Exception as e:
                print(f"[models][warn] {model_name} failed: {e}")
                self.encoder = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base")
        else:
            c = VideoMAEConfig.from_pretrained(model_name)
            self.encoder = VideoMAEModel(c)

        if bool(cfg.get("videomae_grad_checkpointing", True)):
            try:
                self.encoder.gradient_checkpointing_enable()
            except Exception:
                pass

        hidden_size = int(self.encoder.config.hidden_size)  # 768 for base
        tubelet = getattr(self.encoder.config, "tubelet_size", 2)
        self.temporal_stride = max(1, int(tubelet[0] if isinstance(tubelet, (list, tuple)) else tubelet))
        patch_size = getattr(self.encoder.config, "patch_size", 16)
        self.patch_size = int(patch_size[0] if isinstance(patch_size, (list, tuple)) else patch_size)

        # BiLSTM head
        lstm_hidden = int(cfg.get("bilstm_hidden", 256))
        lstm_layers = int(cfg.get("bilstm_layers", 2))
        self.temporal_head = BiLSTMTemporalHead(
            hidden_size, num_classes, lstm_hidden, lstm_layers, dropout
        )

    def forward(self, x):
        B, C, T, H, W = x.shape
        pixel_values = x.permute(0, 2, 1, 3, 4)
        outputs = self.encoder(pixel_values=pixel_values, bool_masked_pos=None)
        hidden = outputs.last_hidden_state

        T_t = max(1, T // self.temporal_stride)
        S_h = max(1, H // self.patch_size)
        S_w = max(1, W // self.patch_size)
        S = S_h * S_w
        need = T_t * S
        if hidden.shape[1] < need:
            S = max(1, hidden.shape[1] // T_t)
            need = T_t * S
        hidden = hidden[:, :need, :]
        hidden = hidden.view(B, T_t, S, -1).mean(dim=2)  # [B, T_t, hidden]
        feats = hidden.permute(0, 2, 1)  # [B, hidden, T_t]
        return self.temporal_head(feats, T)


# =====================================================================
#  11) VideoMAE v2 Base + BiLSTM (через AutoModel + trust_remote_code)
# =====================================================================
class VideoMAEv2BaseTemporal(nn.Module):
    """
    VideoMAEv2 Base (86M params, 768 hidden) — загружается через HF AutoModel.
    Ключ: trust_remote_code=True загружает их кастомный код модели.
    """
    MODEL_NAME = "OpenGVLab/VideoMAEv2-Base"

    def __init__(self, num_classes: int = 2, pretrained: bool = True,
                 dropout: float = 0.3, cfg: dict = None, **kwargs):
        super().__init__()
        cfg = cfg or {}
        from transformers import AutoModel, AutoConfig

        model_name = str(cfg.get("videomaev2_model_name", self.MODEL_NAME))

        if pretrained:
            try:
                config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
                self.encoder = AutoModel.from_pretrained(
                    model_name, config=config, trust_remote_code=True
                )
                print(f"[models] VideoMAEv2 loaded: {model_name}")
            except Exception as e:
                print(f"[models][warn] VideoMAEv2 {model_name} failed: {e}")
                # Fallback to VideoMAE v1 base
                from transformers import VideoMAEModel
                self.encoder = VideoMAEModel.from_pretrained(
                    "MCG-NJU/videomae-base-finetuned-kinetics"
                )
                print("[models] Fallback to VideoMAE v1 base")
        else:
            config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
            self.encoder = AutoModel.from_config(config, trust_remote_code=True)

        if bool(cfg.get("videomae_grad_checkpointing", True)):
            try:
                self.encoder.gradient_checkpointing_enable()
            except Exception:
                pass

        # Определяем hidden_size из конфига
        enc_config = self.encoder.config
        hidden_size = int(getattr(enc_config, "hidden_size",
                         getattr(enc_config, "embed_dim", 768)))
        tubelet = getattr(enc_config, "tubelet_size", 2)
        self.temporal_stride = max(1, int(tubelet[0] if isinstance(tubelet, (list, tuple)) else tubelet))
        patch_size = getattr(enc_config, "patch_size", 16)
        self.patch_size = int(patch_size[0] if isinstance(patch_size, (list, tuple)) else patch_size)

        print(f"[models] VideoMAEv2 hidden={hidden_size}, tubelet={self.temporal_stride}, patch={self.patch_size}")

        lstm_hidden = int(cfg.get("bilstm_hidden", 256))
        lstm_layers = int(cfg.get("bilstm_layers", 2))
        self.temporal_head = BiLSTMTemporalHead(
            hidden_size, num_classes, lstm_hidden, lstm_layers, dropout
        )

    def forward(self, x):
        B, C, T, H, W = x.shape

        # forward_features возвращает [B, num_patches, hidden] ДО pooling.
        # Обычный forward() возвращает [B, hidden] (уже pooled) — бесполезно для BiLSTM.
        encoder = self.encoder

        # Стратегия 1: forward_features (OpenGVLab VisionTransformer)
        if hasattr(encoder, 'forward_features'):
            hidden = encoder.forward_features(x)  # [B, T_t*S, C]
        else:
            # Стратегия 2: ручной проход через блоки
            # patch_embed -> blocks -> norm (пропуская head/pool)
            if hasattr(encoder, 'patch_embed') and hasattr(encoder, 'blocks'):
                hidden = encoder.patch_embed(x)
                if hasattr(encoder, 'pos_embed') and encoder.pos_embed is not None:
                    hidden = hidden + encoder.pos_embed
                if hasattr(encoder, 'pos_drop'):
                    hidden = encoder.pos_drop(hidden)
                for blk in encoder.blocks:
                    hidden = blk(hidden)
                if hasattr(encoder, 'norm') and encoder.norm is not None:
                    hidden = encoder.norm(hidden)
                # hidden: [B, T_t*S, C]
            else:
                # Стратегия 3: fallback на обычный forward (pooled)
                outputs = encoder(pixel_values=x)
                if hasattr(outputs, 'last_hidden_state') and outputs.last_hidden_state is not None:
                    hidden = outputs.last_hidden_state
                elif isinstance(outputs, torch.Tensor):
                    hidden = outputs
                else:
                    hidden = outputs[0] if isinstance(outputs, (tuple, list)) else outputs

        expected_dim = int(self.temporal_head.lstm.input_size)

        # Нормализация формы
        if hidden.ndim == 2:
            # Pooled [B, C] — interpolate (worst case)
            hidden = hidden.unsqueeze(1)  # [B, 1, C]
        elif hidden.ndim == 3:
            if hidden.shape[1] == expected_dim and hidden.shape[2] != expected_dim:
                hidden = hidden.transpose(1, 2).contiguous()
        elif hidden.ndim == 4:
            if hidden.shape[-1] == expected_dim:
                hidden = hidden.view(hidden.shape[0], -1, expected_dim)
            else:
                hidden = hidden.reshape(hidden.shape[0], -1, hidden.shape[-1])

        if hidden.shape[-1] != expected_dim:
            raise RuntimeError(
                f"VideoMAEv2: feature dim {hidden.shape[-1]} != expected {expected_dim}"
            )

        # Reshape tokens -> temporal bins
        T_t = max(1, T // self.temporal_stride)
        n_tokens = hidden.shape[1]

        if n_tokens <= 1:
            # Pooled — interpolate (degraded mode)
            feats = hidden.permute(0, 2, 1)
            feats = F.interpolate(feats, size=T, mode='linear', align_corners=False)
            return self.temporal_head(feats, T)

        S = max(1, n_tokens // T_t)
        need = T_t * S
        if need > n_tokens:
            T_t = max(1, n_tokens // S)
            need = T_t * S

        hidden = hidden[:, :need, :]
        hidden = hidden.reshape(B, T_t, S, -1).mean(dim=2)  # [B, T_t, C]
        feats = hidden.permute(0, 2, 1)  # [B, C, T_t]
        if T_t != T:
            feats = F.interpolate(feats, size=T, mode='linear', align_corners=False)
        return self.temporal_head(feats, T)


# =====================================================================
#  NEW: Transformer Temporal Head (пункт 1)
# =====================================================================
class TransformerTemporalHead(nn.Module):
    """
    Transformer encoder head для per-frame predictions.
    Self-attention вместо BiLSTM: лучше моделирует длинные зависимости.
    Вход: [B, feat_dim, T'] → Выход: [B, num_classes, T_original]
    """
    def __init__(self, feat_dim: int, num_classes: int = 2,
                 d_model: int = 256, nhead: int = 4,
                 num_layers: int = 2, dropout: float = 0.3,
                 max_len: int = 128):
        super().__init__()
        self.input_proj = nn.Linear(feat_dim, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.proj = nn.Linear(d_model, num_classes)

    def forward(self, feats: torch.Tensor, T_target: int) -> torch.Tensor:
        if feats.shape[2] != T_target:
            feats = F.interpolate(feats, size=T_target, mode="linear", align_corners=False)
        x = feats.permute(0, 2, 1)
        x = self.input_proj(x)
        T_cur = x.shape[1]
        x = x + self.pos_embed[:, :T_cur, :]
        x = self.transformer(x)
        x = self.norm(x)
        x = self.drop(x)
        x = self.proj(x)
        return x.permute(0, 2, 1)


class TimmBackboneTransformer(nn.Module):
    """Любой 2D timm backbone + Transformer temporal head."""
    def __init__(self, model_name: str, num_classes: int = 2, pretrained: bool = True,
                 dropout: float = 0.3, cfg: dict = None, **kwargs):
        super().__init__()
        cfg = cfg or {}
        import timm
        self.encoder = timm.create_model(
            model_name, pretrained=pretrained,
            num_classes=0, global_pool="avg",
        )
        feat_dim = self.encoder.num_features
        d_model = int(cfg.get("transformer_d_model", 256))
        nhead = int(cfg.get("transformer_nhead", 4))
        n_layers = int(cfg.get("transformer_layers", 2))
        print(f"[models] TimmBackboneTransformer: {model_name}, feat={feat_dim}, d={d_model}, heads={nhead}, layers={n_layers}")
        self.temporal_head = TransformerTemporalHead(
            feat_dim, num_classes, d_model, nhead, n_layers, dropout
        )

    def forward(self, x):
        B, C, T, H, W = x.shape
        frames = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        feats = self.encoder(frames).view(B, T, -1).permute(0, 2, 1)
        return self.temporal_head(feats, T)


# =====================================================================
#  NEW: Multi-Scale Feature Extraction (пункт 3)
# =====================================================================
class TimmBackboneMultiScale(nn.Module):
    """
    Извлекает фичи из нескольких слоёв backbone и конкатенирует.
    Последний слой = семантика, предпоследний = мелкие детали.
    """
    def __init__(self, model_name: str, num_classes: int = 2, pretrained: bool = True,
                 hidden_size: int = 256, num_layers: int = 2,
                 dropout: float = 0.3, cfg: dict = None, **kwargs):
        super().__init__()
        cfg = cfg or {}
        import timm
        self.encoder = timm.create_model(
            model_name, pretrained=pretrained,
            features_only=True,
            out_indices=(-2, -1),
        )
        feat_info = self.encoder.feature_info
        dims = [feat_info[i]['num_chs'] for i in [-2, -1]]
        total_dim = sum(dims)
        self.pools = nn.ModuleList([nn.AdaptiveAvgPool2d(1) for _ in dims])
        print(f"[models] MultiScale: {model_name}, dims={dims}, total={total_dim}")

        h = int(cfg.get("bilstm_hidden", hidden_size))
        nl = int(cfg.get("bilstm_layers", num_layers))
        self.lstm = nn.LSTM(
            total_dim, h, num_layers=nl,
            batch_first=True, bidirectional=True,
            dropout=dropout if nl > 1 else 0.0,
        )
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(h * 2, num_classes)

    def forward(self, x):
        B, C, T, H, W = x.shape
        frames = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        feature_maps = self.encoder(frames)
        pooled = []
        for fm, pool in zip(feature_maps, self.pools):
            pooled.append(pool(fm).flatten(1))
        feats = torch.cat(pooled, dim=1).view(B, T, -1)
        out, _ = self.lstm(feats)
        out = self.drop(out)
        logits = self.head(out)
        return logits.permute(0, 2, 1)


# =====================================================================
#  NEW: Temporal CRF (пункт 7)
# =====================================================================
class TemporalCRF(nn.Module):
    """
    Conditional Random Field поверх temporal logits.
    Гладкие предсказания: violence длится секунды, не миллисекунды.
    """
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.num_classes = num_classes
        self.transitions = nn.Parameter(torch.randn(num_classes, num_classes) * 0.1)
        self.start_transitions = nn.Parameter(torch.randn(num_classes) * 0.1)
        self.end_transitions = nn.Parameter(torch.randn(num_classes) * 0.1)

    def forward_score(self, emissions, tags, mask=None):
        """Compute negative log-likelihood of tag sequence."""
        B, T, C = emissions.shape
        if mask is None:
            mask = emissions.new_ones(B, T, dtype=torch.bool)
        mask = mask.bool()
        score = self.start_transitions[tags[:, 0]]
        score += emissions[:, 0].gather(1, tags[:, 0:1]).squeeze(1)
        for t in range(1, T):
            emit_score = emissions[:, t].gather(1, tags[:, t:t+1]).squeeze(1)
            trans_score = self.transitions[tags[:, t-1], tags[:, t]]
            score = score + (emit_score + trans_score) * mask[:, t].float()
        last_valid = mask.long().sum(dim=1) - 1
        last_tags = tags.gather(1, last_valid.unsqueeze(1)).squeeze(1)
        score += self.end_transitions[last_tags]
        log_Z = self._forward_alg(emissions, mask)
        return -(score - log_Z)

    def _forward_alg(self, emissions, mask):
        B, T, C = emissions.shape
        alpha = self.start_transitions.unsqueeze(0) + emissions[:, 0]
        for t in range(1, T):
            emit = emissions[:, t].unsqueeze(1)
            trans = self.transitions.unsqueeze(0)
            alpha_next = torch.logsumexp(alpha.unsqueeze(2) + trans + emit, dim=1)
            alpha = torch.where(mask[:, t].unsqueeze(1), alpha_next, alpha)
        alpha += self.end_transitions.unsqueeze(0)
        return torch.logsumexp(alpha, dim=1)

    def decode(self, emissions, mask=None):
        """Viterbi decoding."""
        B, T, C = emissions.shape
        if mask is None:
            mask = emissions.new_ones(B, T, dtype=torch.bool)
        mask = mask.bool()
        alpha = self.start_transitions + emissions[:, 0]
        backpointers = []
        for t in range(1, T):
            scores = alpha.unsqueeze(2) + self.transitions.unsqueeze(0)
            max_scores, bp = scores.max(dim=1)
            alpha_next = max_scores + emissions[:, t]
            alpha = torch.where(mask[:, t].unsqueeze(1), alpha_next, alpha)
            backpointers.append(bp)
        alpha += self.end_transitions.unsqueeze(0)
        _, best_last = alpha.max(dim=1)
        best_path = [best_last]
        for bp in reversed(backpointers):
            best_path.append(bp.gather(1, best_path[-1].unsqueeze(1)).squeeze(1))
        best_path.reverse()
        return torch.stack(best_path, dim=1)


class TimmBackboneBiLSTM_CRF(nn.Module):
    """2D backbone + BiLSTM + CRF: гладкие per-frame predictions."""
    def __init__(self, model_name: str, num_classes: int = 2, pretrained: bool = True,
                 hidden_size: int = 256, num_layers: int = 2,
                 dropout: float = 0.3, cfg: dict = None, **kwargs):
        super().__init__()
        cfg = cfg or {}
        import timm
        self.encoder = timm.create_model(
            model_name, pretrained=pretrained,
            num_classes=0, global_pool="avg",
        )
        feat_dim = self.encoder.num_features
        h = int(cfg.get("bilstm_hidden", hidden_size))
        nl = int(cfg.get("bilstm_layers", num_layers))
        self.lstm = nn.LSTM(
            feat_dim, h, num_layers=nl,
            batch_first=True, bidirectional=True,
            dropout=dropout if nl > 1 else 0.0,
        )
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(h * 2, num_classes)
        self.crf = TemporalCRF(num_classes)
        print(f"[models] TimmBackbone+BiLSTM+CRF: {model_name}, feat={feat_dim}")

    def forward(self, x, tags=None, mask=None):
        B, C, T, H, W = x.shape
        frames = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        feats = self.encoder(frames).view(B, T, -1)
        out, _ = self.lstm(feats)
        out = self.drop(out)
        emissions = self.head(out)
        if tags is not None:
            crf_loss = self.crf.forward_score(emissions, tags, mask).mean()
            logits = emissions.permute(0, 2, 1)
            return logits, crf_loss
        return emissions.permute(0, 2, 1)


# =====================================================================
#  NEW: Boundary-Aware Loss (пункт 8)
# =====================================================================
class BoundaryAwareLoss(nn.Module):
    """
    Дополнительный лосс на границах violence↔non-violence.
    Переходные кадры получают больший вес.
    """
    def __init__(self, boundary_weight: float = 2.0, boundary_radius: int = 3):
        super().__init__()
        self.boundary_weight = boundary_weight
        self.boundary_radius = boundary_radius

    def forward(self, logits, targets, mask=None):
        B, C, T = logits.shape
        if mask is None:
            mask = (targets != -1)
        targets_clamped = targets.clamp(min=0)
        diff = torch.zeros_like(targets_clamped, dtype=torch.float32)
        diff[:, 1:] = (targets_clamped[:, 1:] != targets_clamped[:, :-1]).float()
        if self.boundary_radius > 1:
            kernel = torch.ones(1, 1, self.boundary_radius * 2 + 1, device=diff.device)
            expanded = F.conv1d(diff.unsqueeze(1).float(), kernel, padding=self.boundary_radius).squeeze(1)
            boundary_mask = (expanded > 0).float()
        else:
            boundary_mask = diff
        weights = torch.ones_like(diff) + boundary_mask * (self.boundary_weight - 1.0)
        weights = weights * mask.float()
        loss = F.cross_entropy(logits, targets_clamped, reduction='none')
        loss = (loss * weights).sum() / weights.sum().clamp(min=1.0)
        return loss


# =====================================================================
#  NEW: Skeleton-Fusion Model (пункт 4)
# =====================================================================
class SkeletonTemporalDenoise(nn.Module):
    """
    Lightweight temporal denoise block for skeleton streams.
    Input/Output: [B, T, D]
    """
    def __init__(self, dim: int, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        k = max(1, int(kernel_size))
        if k % 2 == 0:
            k += 1
        self.norm = nn.LayerNorm(dim)
        self.conv = nn.Conv1d(dim, dim, kernel_size=k, padding=k // 2, groups=1)
        self.act = nn.GELU()
        self.drop = nn.Dropout(float(dropout))

    def forward(self, x):
        y = self.norm(x).transpose(1, 2)
        y = self.conv(y)
        y = self.act(y)
        y = self.drop(y).transpose(1, 2)
        return x + y


class SkeletonReliabilityGate(nn.Module):
    """
    Estimates per-frame skeleton reliability in [0,1].
    Input: [B, T, D], Output gate: [B, T, 1]
    """
    def __init__(self, skeleton_dim: int, hidden: int = 16):
        super().__init__()
        del skeleton_dim  # kept for cfg API symmetry
        h = max(4, int(hidden))
        self.mlp = nn.Sequential(
            nn.Linear(3, h),
            nn.GELU(),
            nn.Linear(h, 1),
            nn.Sigmoid(),
        )

    def forward(self, skeleton):
        abs_mean = skeleton.abs().mean(dim=-1, keepdim=True)
        energy = (skeleton * skeleton).mean(dim=-1, keepdim=True).sqrt()
        nonzero = (skeleton.abs() > 1e-6).float().mean(dim=-1, keepdim=True)
        stats = torch.cat([abs_mean, energy, nonzero], dim=-1)
        return self.mlp(stats)


class TimmBackboneSkeletonFusion(nn.Module):
    """
    RGB features + precomputed skeleton features → BiLSTM.
    Skeleton: [B, T, skeleton_dim] из MediaPipe (top-2 людей + velocity + pairwise = 406).
    """
    def __init__(self, model_name: str, num_classes: int = 2, pretrained: bool = True,
                 hidden_size: int = 256, num_layers: int = 2,
                 dropout: float = 0.3, cfg: dict = None, **kwargs):
        super().__init__()
        cfg = cfg or {}
        import timm
        self.encoder = timm.create_model(
            model_name, pretrained=pretrained,
            num_classes=0, global_pool="avg",
        )
        feat_dim = self.encoder.num_features
        skeleton_dim = int(cfg.get("skeleton_dim", 406))  # 2 persons×99 + 2 velocity×99 + 10 pairwise
        total_dim = feat_dim + skeleton_dim
        print(f"[models] SkeletonFusion: {model_name}, rgb={feat_dim}, skel={skeleton_dim}, total={total_dim}")
        self.skeleton_proj = nn.Sequential(
            nn.Linear(skeleton_dim, skeleton_dim), nn.ReLU(), nn.Dropout(dropout),
        )
        self.skeleton_dim = int(skeleton_dim)
        self.strict_skeleton = bool(cfg.get("strict_require_skeleton", False))
        self.use_skeleton_denoise = bool(cfg.get("skeleton_denoise", False))
        if self.use_skeleton_denoise:
            self.skeleton_denoise = SkeletonTemporalDenoise(
                dim=self.skeleton_dim,
                kernel_size=int(cfg.get("skeleton_denoise_kernel", 3)),
                dropout=float(cfg.get("skeleton_denoise_dropout", dropout)),
            )
        else:
            self.skeleton_denoise = nn.Identity()

        self.use_skeleton_reliability_gate = bool(cfg.get("skeleton_reliability_gate", False))
        if self.use_skeleton_reliability_gate:
            self.skeleton_reliability_gate = SkeletonReliabilityGate(
                skeleton_dim=self.skeleton_dim,
                hidden=int(cfg.get("skeleton_reliability_hidden", 16)),
            )
        else:
            self.skeleton_reliability_gate = None
        h = int(cfg.get("bilstm_hidden", hidden_size))
        nl = int(cfg.get("bilstm_layers", num_layers))
        self.lstm = nn.LSTM(
            total_dim, h, num_layers=nl,
            batch_first=True, bidirectional=True,
            dropout=dropout if nl > 1 else 0.0,
        )
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(h * 2, num_classes)

    def forward(self, x, skeleton=None):
        B, C, T, H, W = x.shape
        frames = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        rgb_feats = self.encoder(frames).view(B, T, -1)
        if skeleton is not None:
            if skeleton.ndim != 3:
                raise ValueError(f"Skeleton tensor must be [B,T,D], got shape={tuple(skeleton.shape)}")
            if skeleton.shape[0] != B or skeleton.shape[1] != T:
                raise ValueError(f"Skeleton B/T mismatch: x={tuple(x.shape)} skeleton={tuple(skeleton.shape)}")
            if skeleton.shape[2] != self.skeleton_dim:
                raise ValueError(f"Skeleton D mismatch: expected {self.skeleton_dim}, got {skeleton.shape[2]}")
            gate = self.skeleton_reliability_gate(skeleton) if self.skeleton_reliability_gate is not None else None
            skeleton = self.skeleton_denoise(skeleton)
            skel_feats = self.skeleton_proj(skeleton)
            if gate is not None:
                skel_feats = skel_feats * gate
            feats = torch.cat([rgb_feats, skel_feats], dim=-1)
        else:
            if self.strict_skeleton:
                raise RuntimeError(
                    "Skeleton input is required (strict_require_skeleton=True), but got None. "
                    "Check CFG.use_skeleton and skeleton_root paths."
                )
            skel_zero = rgb_feats.new_zeros(B, T, self.skeleton_dim)
            skel_zero = self.skeleton_denoise(skel_zero)
            skel_zero = self.skeleton_proj(skel_zero)
            feats = torch.cat([rgb_feats, skel_zero], dim=-1)
        out, _ = self.lstm(feats)
        out = self.drop(out)
        logits = self.head(out)
        return logits.permute(0, 2, 1)


# =====================================================================
#  NEW: Optical Flow Dual-Stream (пункт 5)
# =====================================================================
class TimmBackboneFlowFusion(nn.Module):
    """
    Dual-stream: RGB backbone + Flow backbone → concat → BiLSTM.
    Flow: precomputed optical flow (3ch: dx, dy, magnitude).
    """
    def __init__(self, model_name: str, num_classes: int = 2, pretrained: bool = True,
                 hidden_size: int = 256, num_layers: int = 2,
                 dropout: float = 0.3, cfg: dict = None, **kwargs):
        super().__init__()
        cfg = cfg or {}
        import timm
        self.rgb_encoder = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0, global_pool="avg",
        )
        rgb_dim = self.rgb_encoder.num_features
        flow_model = str(cfg.get("flow_backbone", "efficientnet_b0"))
        self.flow_encoder = timm.create_model(
            flow_model, pretrained=pretrained, num_classes=0, global_pool="avg",
        )
        self.strict_flow = bool(cfg.get("strict_require_flow", False))
        flow_dim = self.flow_encoder.num_features
        total_dim = rgb_dim + flow_dim
        print(f"[models] FlowFusion: rgb={model_name}({rgb_dim}), flow={flow_model}({flow_dim}), total={total_dim}")
        h = int(cfg.get("bilstm_hidden", hidden_size))
        nl = int(cfg.get("bilstm_layers", num_layers))
        self.lstm = nn.LSTM(
            total_dim, h, num_layers=nl,
            batch_first=True, bidirectional=True,
            dropout=dropout if nl > 1 else 0.0,
        )
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(h * 2, num_classes)

    def forward(self, x, flow=None):
        B, C, T, H, W = x.shape
        rgb_frames = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        rgb_feats = self.rgb_encoder(rgb_frames).view(B, T, -1)
        if flow is not None:
            if flow.ndim != 5:
                raise ValueError(f"Flow tensor must be [B,3,T,H,W], got shape={tuple(flow.shape)}")
            if flow.shape[0] != B or flow.shape[2] != T:
                raise ValueError(f"Flow B/T mismatch: x={tuple(x.shape)} flow={tuple(flow.shape)}")
            if flow.shape[1] != 3:
                raise ValueError(f"Flow channels mismatch: expected 3, got {flow.shape[1]}")
            flow_frames = flow.permute(0, 2, 1, 3, 4).reshape(B * T, 3, H, W)
        else:
            if self.strict_flow:
                raise RuntimeError(
                    "Flow input is required (strict_require_flow=True), but got None. "
                    "Check CFG.use_flow and flow_frames_root paths."
                )
            flow_frames = rgb_frames.new_zeros(B * T, 3, H, W)
        flow_feats = self.flow_encoder(flow_frames).view(B, T, -1)
        feats = torch.cat([rgb_feats, flow_feats], dim=-1)
        out, _ = self.lstm(feats)
        out = self.drop(out)
        logits = self.head(out)
        return logits.permute(0, 2, 1)


# =====================================================================
#  NEW V4: TCN (Temporal Convolutional Network) head
#  Из обзора [1]: TCN конкурирует с RNN на temporal tasks.
#  Dilated convolutions: слой 1 видит ±1 кадр, слой 2 — ±2, слой 3 — ±4...
#  На clip_len=32: 5 слоёв покрывают весь клип (2^5 = 32).
# =====================================================================
class TCNTemporalHead(nn.Module):
    """
    Temporal Convolutional Network для per-frame predictions.
    Каждый слой = dilated causal conv1d → BatchNorm → ReLU → Dropout.
    Dilation растёт экспоненциально: 1, 2, 4, 8, 16...
    """
    def __init__(self, feat_dim: int, num_classes: int = 2,
                 hidden_dim: int = 256, num_layers: int = 5,
                 kernel_size: int = 3, dropout: float = 0.3):
        super().__init__()
        self.input_proj = nn.Conv1d(feat_dim, hidden_dim, 1)
        layers = []
        for i in range(num_layers):
            dilation = 2 ** i
            padding = (kernel_size - 1) * dilation // 2  # same padding
            layers.append(nn.Sequential(
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size,
                          padding=padding, dilation=dilation),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ))
        self.layers = nn.ModuleList(layers)
        self.proj = nn.Conv1d(hidden_dim, num_classes, 1)

    def forward(self, feats: torch.Tensor, T_target: int) -> torch.Tensor:
        # feats: [B, feat_dim, T']
        if feats.shape[2] != T_target:
            feats = F.interpolate(feats, size=T_target, mode="linear", align_corners=False)
        x = self.input_proj(feats)          # [B, hidden, T]
        for layer in self.layers:
            residual = x
            x = layer(x)
            # Ensure same length for residual connection
            if x.shape[2] != residual.shape[2]:
                x = x[:, :, :residual.shape[2]]
            x = x + residual                # residual connection
        return self.proj(x)                 # [B, num_classes, T]


class TimmBackboneTCN(nn.Module):
    """Любой 2D timm backbone + TCN temporal head."""
    def __init__(self, model_name: str, num_classes: int = 2, pretrained: bool = True,
                 dropout: float = 0.3, cfg: dict = None, **kwargs):
        super().__init__()
        cfg = cfg or {}
        import timm
        self.encoder = timm.create_model(
            model_name, pretrained=pretrained,
            num_classes=0, global_pool="avg",
        )
        feat_dim = self.encoder.num_features
        hidden = int(cfg.get("tcn_hidden", 256))
        n_layers = int(cfg.get("tcn_layers", 5))
        ks = int(cfg.get("tcn_kernel_size", 3))
        print(f"[models] TimmBackboneTCN: {model_name}, feat={feat_dim}, hidden={hidden}, layers={n_layers}, ks={ks}")
        self.temporal_head = TCNTemporalHead(
            feat_dim, num_classes, hidden, n_layers, ks, dropout
        )

    def forward(self, x):
        B, C, T, H, W = x.shape
        frames = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        feats = self.encoder(frames).view(B, T, -1).permute(0, 2, 1)  # [B, feat, T]
        return self.temporal_head(feats, T)


# =====================================================================
#  NEW V5: Two-Stream Fusion (RGB + Optical Flow)
#
#  Три варианта объединения потоков:
#    - concat:    [rgb; flow] → linear → temporal head
#    - gated:     g = σ(W[rgb;flow]), fused = g⊙rgb + (1-g)⊙flow
#    - attention: cross-attention между rgb и flow
#
#  Использование:
#    model = build_model("twostream_gated_bilstm", cfg=CFG)
#    logits = model(rgb_clip, flow=flow_clip)  # оба [B, C, T, H, W]
# =====================================================================

class ConcatFusion(nn.Module):
    """Простой concat + linear projection."""
    def __init__(self, rgb_dim: int, flow_dim: int, out_dim: int, **kwargs):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(rgb_dim + flow_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )

    def forward(self, rgb_feat, flow_feat):
        return self.proj(torch.cat([rgb_feat, flow_feat], dim=-1))


class GatedFusion(nn.Module):
    """
    Adaptive gated fusion: модель учится, когда слушать RGB, а когда flow.
    g = σ(W_gate · [rgb; flow] + b)
    fused = g ⊙ rgb_proj + (1-g) ⊙ flow_proj
    """
    def __init__(self, rgb_dim: int, flow_dim: int, out_dim: int, **kwargs):
        super().__init__()
        self.rgb_proj = nn.Linear(rgb_dim, out_dim)
        self.flow_proj = nn.Linear(flow_dim, out_dim)
        self.gate = nn.Sequential(
            nn.Linear(rgb_dim + flow_dim, out_dim),
            nn.Sigmoid(),
        )
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, rgb_feat, flow_feat):
        g = self.gate(torch.cat([rgb_feat, flow_feat], dim=-1))
        fused = g * self.rgb_proj(rgb_feat) + (1 - g) * self.flow_proj(flow_feat)
        return self.norm(fused)


class CrossAttentionFusion(nn.Module):
    """
    Cross-attention: каждый поток attend-ит на другой.
    RGB смотрит на flow (что двигалось?), flow смотрит на RGB (что это?).
    """
    def __init__(self, rgb_dim: int, flow_dim: int, out_dim: int,
                 nhead: int = 4, dropout: float = 0.1, **kwargs):
        super().__init__()
        self.rgb_proj = nn.Linear(rgb_dim, out_dim)
        self.flow_proj = nn.Linear(flow_dim, out_dim)
        self.cross_attn_rgb = nn.MultiheadAttention(
            out_dim, nhead, dropout=dropout, batch_first=True
        )
        self.cross_attn_flow = nn.MultiheadAttention(
            out_dim, nhead, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(out_dim)
        self.norm2 = nn.LayerNorm(out_dim)
        self.ffn = nn.Sequential(
            nn.Linear(out_dim * 2, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.out_norm = nn.LayerNorm(out_dim)

    def forward(self, rgb_feat, flow_feat):
        rgb = self.rgb_proj(rgb_feat)
        flow = self.flow_proj(flow_feat)
        # RGB attends to flow: "где движение?"
        rgb_cross, _ = self.cross_attn_rgb(rgb, flow, flow)
        rgb_out = self.norm1(rgb + rgb_cross)
        # Flow attends to RGB: "что именно двигалось?"
        flow_cross, _ = self.cross_attn_flow(flow, rgb, rgb)
        flow_out = self.norm2(flow + flow_cross)
        fused = self.ffn(torch.cat([rgb_out, flow_out], dim=-1))
        return self.out_norm(fused)


_FUSION_REGISTRY = {
    "concat": ConcatFusion,
    "gated": GatedFusion,
    "attention": CrossAttentionFusion,
}


def _build_fusion(fusion_type, rgb_dim, flow_dim, fusion_dim, nhead=4, dropout=0.1):
    """Создаёт fusion module по имени."""
    cls = _FUSION_REGISTRY.get(fusion_type, GatedFusion)
    return cls(rgb_dim, flow_dim, fusion_dim, nhead=nhead, dropout=dropout)


class TwoStreamFusionBiLSTM(nn.Module):
    """
    Two-stream RGB + Flow → fusion → BiLSTM → per-frame logits.

    cfg параметры:
        fusion_type:     "concat" | "gated" | "attention" (default: "gated")
        fusion_dim:      размерность после fusion (default: 512)
        flow_backbone:   timm backbone для flow (default: "efficientnet_b0")
        flow_pretrained: использовать pretrained для flow (default: True)
        attn_nhead:      число голов для attention fusion (default: 4)
        bilstm_hidden:   BiLSTM hidden size (default: 256)
        bilstm_layers:   BiLSTM layers (default: 2)
    """
    def __init__(self, model_name: str, num_classes: int = 2, pretrained: bool = True,
                 dropout: float = 0.3, cfg: dict = None, **kwargs):
        super().__init__()
        cfg = cfg or {}
        import timm

        # RGB backbone
        self.rgb_encoder = timm.create_model(
            model_name, pretrained=pretrained,
            num_classes=0, global_pool="avg",
        )
        rgb_dim = self.rgb_encoder.num_features

        # Flow backbone (может быть легче — flow проще RGB)
        flow_model = str(cfg.get("flow_backbone", "efficientnet_b0"))
        flow_pt = bool(cfg.get("flow_pretrained", True))
        self.flow_encoder = timm.create_model(
            flow_model, pretrained=flow_pt,
            num_classes=0, global_pool="avg",
        )
        self.strict_flow = bool(cfg.get("strict_require_flow", False))
        flow_dim = self.flow_encoder.num_features

        # Fusion
        fusion_type = str(cfg.get("fusion_type", "gated"))
        fusion_dim = int(cfg.get("fusion_dim", 512))
        attn_nhead = int(cfg.get("attn_nhead", cfg.get("fusion_nhead", 4)))
        self.fusion = _build_fusion(fusion_type, rgb_dim, flow_dim, fusion_dim,
                                    nhead=attn_nhead, dropout=dropout)
        self.fusion_type = fusion_type

        # BiLSTM temporal head
        h = int(cfg.get("bilstm_hidden", 256))
        nl = int(cfg.get("bilstm_layers", 2))
        self.lstm = nn.LSTM(
            fusion_dim, h, num_layers=nl,
            batch_first=True, bidirectional=True,
            dropout=dropout if nl > 1 else 0.0,
        )
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(h * 2, num_classes)

        print(f"[models] TwoStreamFusionBiLSTM: rgb={model_name}({rgb_dim}), "
              f"flow={flow_model}({flow_dim}), fusion={fusion_type}({fusion_dim}), "
              f"bilstm={h}×{nl}")

    def forward(self, x, flow=None):
        B, C, T, H, W = x.shape
        rgb_frames = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        rgb_feats = self.rgb_encoder(rgb_frames).view(B, T, -1)

        if flow is not None:
            if flow.ndim != 5:
                raise ValueError(f"Flow tensor must be [B,3,T,H,W], got shape={tuple(flow.shape)}")
            if flow.shape[0] != B or flow.shape[2] != T:
                raise ValueError(f"Flow B/T mismatch: x={tuple(x.shape)} flow={tuple(flow.shape)}")
            if flow.shape[1] != 3:
                raise ValueError(f"Flow channels mismatch: expected 3, got {flow.shape[1]}")
            flow_frames = flow.permute(0, 2, 1, 3, 4).reshape(B * T, 3, H, W)
        else:
            if self.strict_flow:
                raise RuntimeError(
                    "Flow input is required (strict_require_flow=True), but got None. "
                    "Check CFG.use_flow and flow_frames_root paths."
                )
            flow_frames = rgb_frames.new_zeros(B * T, 3, H, W)
        flow_feats = self.flow_encoder(flow_frames).view(B, T, -1)

        fused = self.fusion(rgb_feats, flow_feats)  # [B, T, fusion_dim]
        out, _ = self.lstm(fused)
        out = self.drop(out)
        logits = self.head(out)
        return logits.permute(0, 2, 1)


class TwoStreamFusionTCN(nn.Module):
    """
    Two-stream RGB + Flow → fusion → TCN → per-frame logits.
    Та же fusion логика, но TCN вместо BiLSTM.
    """
    def __init__(self, model_name: str, num_classes: int = 2, pretrained: bool = True,
                 dropout: float = 0.3, cfg: dict = None, **kwargs):
        super().__init__()
        cfg = cfg or {}
        import timm

        self.rgb_encoder = timm.create_model(
            model_name, pretrained=pretrained,
            num_classes=0, global_pool="avg",
        )
        rgb_dim = self.rgb_encoder.num_features

        flow_model = str(cfg.get("flow_backbone", "efficientnet_b0"))
        flow_pt = bool(cfg.get("flow_pretrained", True))
        self.flow_encoder = timm.create_model(
            flow_model, pretrained=flow_pt,
            num_classes=0, global_pool="avg",
        )
        self.strict_flow = bool(cfg.get("strict_require_flow", False))
        flow_dim = self.flow_encoder.num_features

        fusion_type = str(cfg.get("fusion_type", "gated"))
        fusion_dim = int(cfg.get("fusion_dim", 512))
        attn_nhead = int(cfg.get("attn_nhead", cfg.get("fusion_nhead", 4)))
        self.fusion = _build_fusion(fusion_type, rgb_dim, flow_dim, fusion_dim,
                                    nhead=attn_nhead, dropout=dropout)
        self.fusion_type = fusion_type

        tcn_hidden = int(cfg.get("tcn_hidden", 256))
        tcn_layers = int(cfg.get("tcn_layers", 5))
        tcn_ks = int(cfg.get("tcn_kernel_size", 3))
        self.temporal_head = TCNTemporalHead(
            fusion_dim, num_classes, tcn_hidden, tcn_layers, tcn_ks, dropout
        )

        print(f"[models] TwoStreamFusionTCN: rgb={model_name}({rgb_dim}), "
              f"flow={flow_model}({flow_dim}), fusion={fusion_type}({fusion_dim}), "
              f"tcn={tcn_hidden}×{tcn_layers}")

    def forward(self, x, flow=None):
        B, C, T, H, W = x.shape
        rgb_frames = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        rgb_feats = self.rgb_encoder(rgb_frames).view(B, T, -1)

        if flow is not None:
            if flow.ndim != 5:
                raise ValueError(f"Flow tensor must be [B,3,T,H,W], got shape={tuple(flow.shape)}")
            if flow.shape[0] != B or flow.shape[2] != T:
                raise ValueError(f"Flow B/T mismatch: x={tuple(x.shape)} flow={tuple(flow.shape)}")
            if flow.shape[1] != 3:
                raise ValueError(f"Flow channels mismatch: expected 3, got {flow.shape[1]}")
            flow_frames = flow.permute(0, 2, 1, 3, 4).reshape(B * T, 3, H, W)
        else:
            if self.strict_flow:
                raise RuntimeError(
                    "Flow input is required (strict_require_flow=True), but got None. "
                    "Check CFG.use_flow and flow_frames_root paths."
                )
            flow_frames = rgb_frames.new_zeros(B * T, 3, H, W)
        flow_feats = self.flow_encoder(flow_frames).view(B, T, -1)

        fused = self.fusion(rgb_feats, flow_feats)
        fused = fused.permute(0, 2, 1)  # [B, fusion_dim, T] for TCN
        return self.temporal_head(fused, T)


class SkeletonFusionBiLSTM(nn.Module):
    """
    Two-stream RGB + Skeleton -> fusion -> BiLSTM -> per-frame logits.
    Skeleton expected shape: [B, T, D].
    """
    def __init__(self, model_name: str, num_classes: int = 2, pretrained: bool = True,
                 dropout: float = 0.3, cfg: dict = None, **kwargs):
        super().__init__()
        cfg = cfg or {}
        import timm

        self.rgb_encoder = timm.create_model(
            model_name, pretrained=pretrained,
            num_classes=0, global_pool="avg",
        )
        rgb_dim = self.rgb_encoder.num_features

        self.skeleton_dim = int(cfg.get("skeleton_dim", 406))
        skel_proj_dim = int(cfg.get("skeleton_proj_dim", 256))
        self.strict_skeleton = bool(cfg.get("strict_require_skeleton", False))
        self.skeleton_proj = nn.Sequential(
            nn.LayerNorm(self.skeleton_dim),
            nn.Linear(self.skeleton_dim, skel_proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.use_skeleton_denoise = bool(cfg.get("skeleton_denoise", False))
        if self.use_skeleton_denoise:
            self.skeleton_denoise = SkeletonTemporalDenoise(
                dim=self.skeleton_dim,
                kernel_size=int(cfg.get("skeleton_denoise_kernel", 3)),
                dropout=float(cfg.get("skeleton_denoise_dropout", dropout)),
            )
        else:
            self.skeleton_denoise = nn.Identity()

        self.use_skeleton_reliability_gate = bool(cfg.get("skeleton_reliability_gate", False))
        if self.use_skeleton_reliability_gate:
            self.skeleton_reliability_gate = SkeletonReliabilityGate(
                skeleton_dim=self.skeleton_dim,
                hidden=int(cfg.get("skeleton_reliability_hidden", 16)),
            )
        else:
            self.skeleton_reliability_gate = None

        fusion_type = str(cfg.get("skeleton_fusion_type", cfg.get("fusion_type", "gated")))
        fusion_dim = int(cfg.get("fusion_dim", 512))
        attn_nhead = int(cfg.get("attn_nhead", cfg.get("fusion_nhead", 4)))
        self.fusion = _build_fusion(fusion_type, rgb_dim, skel_proj_dim, fusion_dim,
                                    nhead=attn_nhead, dropout=dropout)
        self.fusion_type = fusion_type

        h = int(cfg.get("bilstm_hidden", 256))
        nl = int(cfg.get("bilstm_layers", 2))
        self.lstm = nn.LSTM(
            fusion_dim, h, num_layers=nl,
            batch_first=True, bidirectional=True,
            dropout=dropout if nl > 1 else 0.0,
        )
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(h * 2, num_classes)

        print(f"[models] SkeletonFusionBiLSTM: rgb={model_name}({rgb_dim}), "
              f"skel={self.skeleton_dim}->{skel_proj_dim}, fusion={fusion_type}({fusion_dim}), "
              f"bilstm={h}x{nl}")

    def forward(self, x, skeleton=None):
        B, C, T, H, W = x.shape
        rgb_frames = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        rgb_feats = self.rgb_encoder(rgb_frames).view(B, T, -1)

        if skeleton is not None:
            if skeleton.ndim != 3:
                raise ValueError(f"Skeleton tensor must be [B,T,D], got shape={tuple(skeleton.shape)}")
            if skeleton.shape[0] != B or skeleton.shape[1] != T:
                raise ValueError(f"Skeleton B/T mismatch: x={tuple(x.shape)} skeleton={tuple(skeleton.shape)}")
            if skeleton.shape[2] != self.skeleton_dim:
                raise ValueError(f"Skeleton D mismatch: expected {self.skeleton_dim}, got {skeleton.shape[2]}")
            gate = self.skeleton_reliability_gate(skeleton) if self.skeleton_reliability_gate is not None else None
            skeleton = self.skeleton_denoise(skeleton)
            skel_feats = self.skeleton_proj(skeleton)
            if gate is not None:
                skel_feats = skel_feats * gate
        else:
            if self.strict_skeleton:
                raise RuntimeError(
                    "Skeleton input is required (strict_require_skeleton=True), but got None. "
                    "Check CFG.use_skeleton and skeleton_root paths."
                )
            skel_zero = rgb_feats.new_zeros(B, T, self.skeleton_dim)
            skel_zero = self.skeleton_denoise(skel_zero)
            skel_feats = self.skeleton_proj(skel_zero)

        fused = self.fusion(rgb_feats, skel_feats)
        out, _ = self.lstm(fused)
        out = self.drop(out)
        logits = self.head(out)
        return logits.permute(0, 2, 1)


class SkeletonFusionTCN(nn.Module):
    """
    Two-stream RGB + Skeleton -> fusion -> TCN -> per-frame logits.
    """
    def __init__(self, model_name: str, num_classes: int = 2, pretrained: bool = True,
                 dropout: float = 0.3, cfg: dict = None, **kwargs):
        super().__init__()
        cfg = cfg or {}
        import timm

        self.rgb_encoder = timm.create_model(
            model_name, pretrained=pretrained,
            num_classes=0, global_pool="avg",
        )
        rgb_dim = self.rgb_encoder.num_features

        self.skeleton_dim = int(cfg.get("skeleton_dim", 406))
        skel_proj_dim = int(cfg.get("skeleton_proj_dim", 256))
        self.strict_skeleton = bool(cfg.get("strict_require_skeleton", False))
        self.skeleton_proj = nn.Sequential(
            nn.LayerNorm(self.skeleton_dim),
            nn.Linear(self.skeleton_dim, skel_proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.use_skeleton_denoise = bool(cfg.get("skeleton_denoise", False))
        if self.use_skeleton_denoise:
            self.skeleton_denoise = SkeletonTemporalDenoise(
                dim=self.skeleton_dim,
                kernel_size=int(cfg.get("skeleton_denoise_kernel", 3)),
                dropout=float(cfg.get("skeleton_denoise_dropout", dropout)),
            )
        else:
            self.skeleton_denoise = nn.Identity()

        self.use_skeleton_reliability_gate = bool(cfg.get("skeleton_reliability_gate", False))
        if self.use_skeleton_reliability_gate:
            self.skeleton_reliability_gate = SkeletonReliabilityGate(
                skeleton_dim=self.skeleton_dim,
                hidden=int(cfg.get("skeleton_reliability_hidden", 16)),
            )
        else:
            self.skeleton_reliability_gate = None

        fusion_type = str(cfg.get("skeleton_fusion_type", cfg.get("fusion_type", "gated")))
        fusion_dim = int(cfg.get("fusion_dim", 512))
        attn_nhead = int(cfg.get("attn_nhead", cfg.get("fusion_nhead", 4)))
        self.fusion = _build_fusion(fusion_type, rgb_dim, skel_proj_dim, fusion_dim,
                                    nhead=attn_nhead, dropout=dropout)
        self.fusion_type = fusion_type

        tcn_hidden = int(cfg.get("tcn_hidden", 256))
        tcn_layers = int(cfg.get("tcn_layers", 5))
        tcn_ks = int(cfg.get("tcn_kernel_size", 3))
        self.temporal_head = TCNTemporalHead(
            fusion_dim, num_classes, tcn_hidden, tcn_layers, tcn_ks, dropout
        )

        print(f"[models] SkeletonFusionTCN: rgb={model_name}({rgb_dim}), "
              f"skel={self.skeleton_dim}->{skel_proj_dim}, fusion={fusion_type}({fusion_dim}), "
              f"tcn={tcn_hidden}x{tcn_layers}")

    def forward(self, x, skeleton=None):
        B, C, T, H, W = x.shape
        rgb_frames = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        rgb_feats = self.rgb_encoder(rgb_frames).view(B, T, -1)

        if skeleton is not None:
            if skeleton.ndim != 3:
                raise ValueError(f"Skeleton tensor must be [B,T,D], got shape={tuple(skeleton.shape)}")
            if skeleton.shape[0] != B or skeleton.shape[1] != T:
                raise ValueError(f"Skeleton B/T mismatch: x={tuple(x.shape)} skeleton={tuple(skeleton.shape)}")
            if skeleton.shape[2] != self.skeleton_dim:
                raise ValueError(f"Skeleton D mismatch: expected {self.skeleton_dim}, got {skeleton.shape[2]}")
            gate = self.skeleton_reliability_gate(skeleton) if self.skeleton_reliability_gate is not None else None
            skeleton = self.skeleton_denoise(skeleton)
            skel_feats = self.skeleton_proj(skeleton)
            if gate is not None:
                skel_feats = skel_feats * gate
        else:
            if self.strict_skeleton:
                raise RuntimeError(
                    "Skeleton input is required (strict_require_skeleton=True), but got None. "
                    "Check CFG.use_skeleton and skeleton_root paths."
                )
            skel_zero = rgb_feats.new_zeros(B, T, self.skeleton_dim)
            skel_zero = self.skeleton_denoise(skel_zero)
            skel_feats = self.skeleton_proj(skel_zero)

        fused = self.fusion(rgb_feats, skel_feats)
        fused = fused.permute(0, 2, 1)
        return self.temporal_head(fused, T)


def _cfg_with_defaults(cfg: dict, **defaults):
    out = dict(cfg or {})
    for k, v in defaults.items():
        out.setdefault(k, v)
    return out


# =====================================================================
#  NOTES:
#  - MoViNet4Violence: TensorFlow only, не интегрируется в PyTorch pipeline
#  - CUE-Net: не готовая модель, а инструкции по модификации UniformerV2
#  - UniFormerV2: нужен клон OpenGVLab/UniFormerV2 + sys.path
# =====================================================================


# =====================================================================
#  build_model() — единая фабрика
# =====================================================================
_BACKBONE_REGISTRY = {
    # Original
    "resnet18":               lambda nc, pt, do, cfg: ResNet18Temporal(nc, pt, do),
    "efficientnet_b0_bilstm": lambda nc, pt, do, cfg: EfficientNetBiLSTM(nc, pt, dropout=do),
    "video_swin_tiny":        lambda nc, pt, do, cfg: VideoSwinTinyTemporal(nc, pt, do),
    "videomae_small":         lambda nc, pt, do, cfg: VideoMAESmallTemporal(nc, pt, do),

    # Extra (original heads)
    "videomae_crime_init":    lambda nc, pt, do, cfg: VideoMAECrimeInitTemporal(nc, pt, do, cfg=cfg),
    "slowfast_r50":           lambda nc, pt, do, cfg: SlowFastR50Temporal(nc, pt, do, cfg=cfg),
    "r3d_18_temporal":        lambda nc, pt, do, cfg: TorchvisionConv3DTemporal("r3d_18", nc, pt, do),
    "r2plus1d_18_temporal":   lambda nc, pt, do, cfg: TorchvisionConv3DTemporal("r2plus1d_18", nc, pt, do),
    "s3d_temporal":           lambda nc, pt, do, cfg: TorchvisionConv3DTemporal("s3d", nc, pt, do),
    "i3d_s3d_temporal":       lambda nc, pt, do, cfg: TorchvisionConv3DTemporal("s3d", nc, pt, do),

    # V2 (BiLSTM temporal heads)
    "r3d_18_temporal_v2":     lambda nc, pt, do, cfg: TorchvisionConv3DTemporalV2("r3d_18", nc, pt, do, cfg=cfg),
    "r2plus1d_18_temporal_v2":lambda nc, pt, do, cfg: TorchvisionConv3DTemporalV2("r2plus1d_18", nc, pt, do, cfg=cfg),
    "s3d_temporal_v2":        lambda nc, pt, do, cfg: TorchvisionConv3DTemporalV2("s3d", nc, pt, do, cfg=cfg),
    "i3d_s3d_temporal_v2":    lambda nc, pt, do, cfg: TorchvisionConv3DTemporalV2("s3d", nc, pt, do, cfg=cfg),
    "slowfast_r50_v2":        lambda nc, pt, do, cfg: SlowFastR50TemporalV2(nc, pt, do, cfg=cfg),
    "videomae_crime_init_v2": lambda nc, pt, do, cfg: VideoMAECrimeInitTemporalV2(nc, pt, do, cfg=cfg),

    # NEW: 2D backbones + BiLSTM (через timm)
    "convnext_tiny_bilstm":   lambda nc, pt, do, cfg: TimmBackboneBiLSTM("convnext_tiny", nc, pt, dropout=do, cfg=cfg),
    "convnext_base_bilstm":   lambda nc, pt, do, cfg: TimmBackboneBiLSTM("convnext_base", nc, pt, dropout=do, cfg=cfg),
    "efficientnet_b3_bilstm": lambda nc, pt, do, cfg: TimmBackboneBiLSTM("efficientnet_b3", nc, pt, dropout=do, cfg=cfg),

    # NEW: I3D R50 + BiLSTM (pytorchvideo)
    "i3d_r50_v2":             lambda nc, pt, do, cfg: I3DR50TemporalV2(nc, pt, do, cfg=cfg),

    # NEW: VideoMAE Base + BiLSTM (HuggingFace, 768 hidden)
    "videomae_base_bilstm":   lambda nc, pt, do, cfg: VideoMAEBaseTemporal(nc, pt, do, cfg=cfg),

    # NEW: VideoMAE v2 Base + BiLSTM
    "videomaev2_base_bilstm": lambda nc, pt, do, cfg: VideoMAEv2BaseTemporal(nc, pt, do, cfg=cfg),

    # NEW V3: Transformer temporal heads
    "convnext_tiny_transformer":   lambda nc, pt, do, cfg: TimmBackboneTransformer("convnext_tiny", nc, pt, dropout=do, cfg=cfg),
    "efficientnet_b0_transformer": lambda nc, pt, do, cfg: TimmBackboneTransformer("efficientnet_b0", nc, pt, dropout=do, cfg=cfg),

    # NEW V3: Multi-scale features
    "convnext_tiny_multiscale":    lambda nc, pt, do, cfg: TimmBackboneMultiScale("convnext_tiny", nc, pt, dropout=do, cfg=cfg),
    "efficientnet_b0_multiscale":  lambda nc, pt, do, cfg: TimmBackboneMultiScale("efficientnet_b0", nc, pt, dropout=do, cfg=cfg),

    # NEW V3: BiLSTM + CRF
    "convnext_tiny_crf":           lambda nc, pt, do, cfg: TimmBackboneBiLSTM_CRF("convnext_tiny", nc, pt, dropout=do, cfg=cfg),
    "efficientnet_b0_crf":         lambda nc, pt, do, cfg: TimmBackboneBiLSTM_CRF("efficientnet_b0", nc, pt, dropout=do, cfg=cfg),

    # NEW V3: Skeleton fusion (needs precomputed skeleton .npy)
    "convnext_tiny_skeleton":      lambda nc, pt, do, cfg: TimmBackboneSkeletonFusion("convnext_tiny", nc, pt, dropout=do, cfg=cfg),

    # NEW V6: Skeleton fusion variants (RGB + skeleton)
    "skeleton_fusion_bilstm":      lambda nc, pt, do, cfg: SkeletonFusionBiLSTM("convnext_tiny", nc, pt, dropout=do, cfg=cfg),
    "skeleton_concat_bilstm":      lambda nc, pt, do, cfg: SkeletonFusionBiLSTM("convnext_tiny", nc, pt, dropout=do, cfg=_cfg_with_defaults(cfg, skeleton_fusion_type="concat")),
    "skeleton_gated_bilstm":       lambda nc, pt, do, cfg: SkeletonFusionBiLSTM("convnext_tiny", nc, pt, dropout=do, cfg=_cfg_with_defaults(cfg, skeleton_fusion_type="gated")),
    "skeleton_attention_bilstm":   lambda nc, pt, do, cfg: SkeletonFusionBiLSTM("convnext_tiny", nc, pt, dropout=do, cfg=_cfg_with_defaults(cfg, skeleton_fusion_type="attention")),
    "skeleton_fusion_tcn":         lambda nc, pt, do, cfg: SkeletonFusionTCN("convnext_tiny", nc, pt, dropout=do, cfg=cfg),
    "skeleton_concat_tcn":         lambda nc, pt, do, cfg: SkeletonFusionTCN("convnext_tiny", nc, pt, dropout=do, cfg=_cfg_with_defaults(cfg, skeleton_fusion_type="concat")),
    "skeleton_gated_tcn":          lambda nc, pt, do, cfg: SkeletonFusionTCN("convnext_tiny", nc, pt, dropout=do, cfg=_cfg_with_defaults(cfg, skeleton_fusion_type="gated")),
    "skeleton_attention_tcn":      lambda nc, pt, do, cfg: SkeletonFusionTCN("convnext_tiny", nc, pt, dropout=do, cfg=_cfg_with_defaults(cfg, skeleton_fusion_type="attention")),

    # NEW V3: Optical flow dual-stream (needs precomputed flow)
    "convnext_tiny_flow":          lambda nc, pt, do, cfg: TimmBackboneFlowFusion("convnext_tiny", nc, pt, dropout=do, cfg=cfg),

    # NEW V4: ConvNeXt-Small (between tiny=768 and base=1024, more layers)
    "convnext_small_bilstm":       lambda nc, pt, do, cfg: TimmBackboneBiLSTM("convnext_small", nc, pt, dropout=do, cfg=cfg),

    # NEW V4: TCN temporal heads
    "convnext_tiny_tcn":           lambda nc, pt, do, cfg: TimmBackboneTCN("convnext_tiny", nc, pt, dropout=do, cfg=cfg),
    "efficientnet_b0_tcn":         lambda nc, pt, do, cfg: TimmBackboneTCN("efficientnet_b0", nc, pt, dropout=do, cfg=cfg),

    # NEW V5: Two-stream fusion (RGB + Optical Flow)
    # BiLSTM temporal head variants
    "twostream_concat_bilstm":     lambda nc, pt, do, cfg: TwoStreamFusionBiLSTM("convnext_tiny", nc, pt, dropout=do, cfg=cfg),
    "twostream_gated_bilstm":      lambda nc, pt, do, cfg: TwoStreamFusionBiLSTM("convnext_tiny", nc, pt, dropout=do, cfg=cfg),
    "twostream_attention_bilstm":  lambda nc, pt, do, cfg: TwoStreamFusionBiLSTM("convnext_tiny", nc, pt, dropout=do, cfg=cfg),
    # TCN temporal head variants
    "twostream_gated_tcn":         lambda nc, pt, do, cfg: TwoStreamFusionTCN("convnext_tiny", nc, pt, dropout=do, cfg=cfg),
    "twostream_attention_tcn":     lambda nc, pt, do, cfg: TwoStreamFusionTCN("convnext_tiny", nc, pt, dropout=do, cfg=cfg),
}

ALL_BACKBONES = sorted(_BACKBONE_REGISTRY.keys())


def build_model(backbone: str = "resnet18", num_classes: int = 2,
                pretrained: bool = True, dropout: float = 0.3,
                cfg: dict = None) -> nn.Module:
    """
    Единая фабрика моделей.

    Args:
        backbone: имя бэкбона (см. ALL_BACKBONES)
        num_classes: число классов
        pretrained: использовать pretrained веса
        dropout: dropout rate
        cfg: словарь конфигурации (нужен для SlowFast, VideoMAE crime, V2 моделей)
    """
    cfg = cfg or {}
    if backbone not in _BACKBONE_REGISTRY:
        raise ValueError(
            f"Unknown backbone: {backbone!r}.\n"
            f"Available: {ALL_BACKBONES}"
        )
    return _BACKBONE_REGISTRY[backbone](num_classes, pretrained, dropout, cfg)


# При импорте как модуль — просто печатаем доступные бэкбоны
if __name__ != "__main__":
    print(f"[models] Available backbones ({len(ALL_BACKBONES)}): {' | '.join(ALL_BACKBONES)}")
