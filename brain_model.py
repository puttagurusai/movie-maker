"""
brain_model.py — architecture matching training notebook (projv1.ipynb).

Modules can be saved/loaded as three packages for inference:
  shared_encoder.pt   — HuBERT + audio_project + emotion_cond + fusion + SharedEncoder
  face_head.pt        — FaceHeadDecoder + lower/upper heads + ARKit index maps
  character_adapter.pt — CharacterAdapter only
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from transformers import HubertModel
except ImportError as e:  # pragma: no cover
    HubertModel = None
    _HUBERT_ERR = e
else:
    _HUBERT_ERR = None

PROJECT_ROOT = Path(__file__).resolve().parent
LOCAL_HUBERT = PROJECT_ROOT / "models" / "brain" / "hubert-base-ls960"

ARKIT_UPPER_INDICES = [
    0, 1, 2, 3, 4,       # Brows
    6, 7,                # Cheek squints
    8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,  # Eyes
    49, 50,              # Nose
]
ARKIT_LOWER_INDICES = [
    5,                   # cheekPuff
    22, 23, 24, 25,      # Jaw
    26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
    51,                  # tongueOut
]

ARKIT_52_NAMES = [
    "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight",
    "cheekPuff", "cheekSquintLeft", "cheekSquintRight",
    "eyeBlinkLeft", "eyeBlinkRight", "eyeLookDownLeft", "eyeLookDownRight",
    "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft", "eyeLookOutRight",
    "eyeLookUpLeft", "eyeLookUpRight", "eyeSquintLeft", "eyeSquintRight",
    "eyeWideLeft", "eyeWideRight",
    "jawForward", "jawLeft", "jawOpen", "jawRight",
    "mouthClose", "mouthDimpleLeft", "mouthDimpleRight", "mouthFrownLeft", "mouthFrownRight",
    "mouthFunnel", "mouthLeft", "mouthLowerDownLeft", "mouthLowerDownRight",
    "mouthPressLeft", "mouthPressRight", "mouthPucker", "mouthRight",
    "mouthRollLower", "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper",
    "mouthSmileLeft", "mouthSmileRight", "mouthStretchLeft", "mouthStretchRight",
    "mouthUpperUpLeft", "mouthUpperUpRight",
    "noseSneerLeft", "noseSneerRight", "tongueOut",
]


def _hubert_source() -> str:
    if (LOCAL_HUBERT / "config.json").is_file():
        return str(LOCAL_HUBERT)
    return "facebook/hubert-base-ls960"


# =====================================================================
# INDEPENDENT MODULES (same as training notebook)
# =====================================================================

class SharedEncoder(nn.Module):
    def __init__(self, embed_dim=384, num_heads=6, pf_dim=1536, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "norm1": nn.LayerNorm(embed_dim),
                "self_attn": nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True),
                "norm2": nn.LayerNorm(embed_dim),
                "cross_attn": nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True),
                "norm3": nn.LayerNorm(embed_dim),
                "ff": nn.Sequential(
                    nn.Linear(embed_dim, pf_dim),
                    nn.GELU(),
                    nn.Linear(pf_dim, embed_dim),
                ),
                "dropout": nn.Dropout(dropout),
            })
            for _ in range(4)
        ])

    def forward(self, x, emotion_emb):
        # x: [B, T, 384], emotion_emb: [B, 384]
        emo_broadcast = emotion_emb.unsqueeze(1).expand(-1, x.size(1), -1)
        for layer in self.layers:
            x_norm = layer["norm1"](x)
            attn_out, _ = layer["self_attn"](x_norm, x_norm, x_norm)
            x = x + layer["dropout"](attn_out)

            x_norm = layer["norm2"](x)
            cross_out, _ = layer["cross_attn"](x_norm, emo_broadcast, emo_broadcast)
            x = x + layer["dropout"](cross_out)

            x_norm = layer["norm3"](x)
            ff_out = layer["ff"](x_norm)
            x = x + layer["dropout"](ff_out)
        return x


class FaceHeadDecoder(nn.Module):
    def __init__(self, embed_dim=384, num_heads=6, pf_dim=1536, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "norm1": nn.LayerNorm(embed_dim),
                "self_attn": nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True),
                "norm2": nn.LayerNorm(embed_dim),
                "cross_attn_encoder": nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True),
                "norm3": nn.LayerNorm(embed_dim),
                "cross_attn_emotion": nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True),
                "norm4": nn.LayerNorm(embed_dim),
                "ff": nn.Sequential(
                    nn.Linear(embed_dim, pf_dim),
                    nn.GELU(),
                    nn.Linear(pf_dim, embed_dim),
                ),
                "dropout": nn.Dropout(dropout),
            })
            for _ in range(4)
        ])

    def _generate_coarticulation_mask(self, sz, device):
        return torch.triu(torch.full((sz, sz), float("-inf"), device=device), diagonal=3)

    def forward(self, x, encoder_out, emotion_emb):
        T = x.size(1)
        coarticulation_mask = self._generate_coarticulation_mask(T, x.device)
        emo_broadcast = emotion_emb.unsqueeze(1).expand(-1, T, -1)
        for layer in self.layers:
            x_norm = layer["norm1"](x)
            attn_out, _ = layer["self_attn"](x_norm, x_norm, x_norm, attn_mask=coarticulation_mask)
            x = x + layer["dropout"](attn_out)

            x_norm = layer["norm2"](x)
            cross_enc_out, _ = layer["cross_attn_encoder"](x_norm, encoder_out, encoder_out)
            x = x + layer["dropout"](cross_enc_out)

            x_norm = layer["norm3"](x)
            cross_emo_out, _ = layer["cross_attn_emotion"](x_norm, emo_broadcast, emo_broadcast)
            x = x + layer["dropout"](cross_emo_out)

            x_norm = layer["norm4"](x)
            ff_out = layer["ff"](x_norm)
            x = x + layer["dropout"](ff_out)
        return x


class CharacterAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(52, 64),
            nn.GELU(),
            nn.Linear(64, 52),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x) * 0.3 + x * 0.7


class SharedEncoderBundle(nn.Module):
    """
    Package saved as shared_encoder.pt:
      hubert + audio_project + emotion_conditioning + fusion + SharedEncoder layers
    """

    def __init__(self, hubert_name: Optional[str] = None):
        super().__init__()
        if HubertModel is None:
            raise ImportError(f"transformers required: {_HUBERT_ERR}")
        hubert_name = hubert_name or _hubert_source()
        self.hubert = HubertModel.from_pretrained(hubert_name)
        for param in self.hubert.feature_extractor.parameters():
            param.requires_grad = False
        for i, layer in enumerate(self.hubert.encoder.layers):
            requires = i >= 6
            for param in layer.parameters():
                param.requires_grad = requires

        self.audio_project = nn.Linear(768 + 3, 768)
        self.emotion_conditioning = nn.Sequential(
            nn.Linear(27, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 384),
            nn.LayerNorm(384),
            nn.GELU(),
        )
        self.fusion_input_project = nn.Linear(768, 384)
        self.shared_encoder = SharedEncoder()

    def encode_audio_features(self, audio_waveform: torch.Tensor, prosody_features: torch.Tensor) -> torch.Tensor:
        """
        HuBERT last hidden → cat prosody → project → 30fps [B, T_30, 768]
        Only last 6 HuBERT layers remain trainable (frozen path still runs full net).
        """
        with torch.no_grad():
            # Full hubert forward; grads disabled for frozen parts automatically if requires_grad False
            hubert_out = self.hubert(audio_waveform).last_hidden_state  # [B, T_50, 768]
        T_50 = hubert_out.size(1)

        prosody = prosody_features.transpose(1, 2)
        prosody = F.interpolate(prosody, size=T_50, mode="linear", align_corners=False)
        prosody = prosody.transpose(1, 2)

        audio_feats = torch.cat([hubert_out, prosody], dim=-1)
        audio_feats = self.audio_project(audio_feats)

        # 50fps → 30fps: T_30 from duration via caller, or scale 50->30 ratio
        T_30 = max(1, int(round(T_50 * 30 / 50)))
        audio_feats = audio_feats.transpose(1, 2)
        audio_feats = F.interpolate(audio_feats, size=T_30, mode="linear", align_corners=False)
        audio_feats = audio_feats.transpose(1, 2)
        return audio_feats  # [B, T_30, 768]

    def forward_fusion(self, audio_feats_768: torch.Tensor, emotion_26d: torch.Tensor, intensity: torch.Tensor):
        """audio [B,T,768] + emo [B,26] + intensity [B] → shared_out [B,T,384], emotion_emb [B,384]"""
        if intensity.dim() == 1:
            intensity = intensity.unsqueeze(-1)
        emotion_input = torch.cat([emotion_26d, intensity], dim=-1)
        emotion_emb = self.emotion_conditioning(emotion_input)
        fusion_input = self.fusion_input_project(audio_feats_768)
        shared_out = self.shared_encoder(fusion_input, emotion_emb)
        return shared_out, emotion_emb


class FaceHeadBundle(nn.Module):
    """Package saved as face_head.pt: decoder + dual heads."""

    def __init__(self):
        super().__init__()
        self.face_head = FaceHeadDecoder()
        self.lower_face_head = nn.Sequential(
            nn.Linear(384, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 29),
            nn.Sigmoid(),
        )
        self.upper_face_linear = nn.Sequential(
            nn.Linear(384, 384),
            nn.GELU(),
            nn.Linear(384, 256),
            nn.GELU(),
            nn.Linear(256, 23),
        )
        self.upper_face_conv = nn.Sequential(
            nn.Conv1d(23, 23, kernel_size=5, padding=2),
            nn.Sigmoid(),
        )
        self.register_buffer(
            "arkit_upper_indices",
            torch.tensor(ARKIT_UPPER_INDICES, dtype=torch.long),
        )
        self.register_buffer(
            "arkit_lower_indices",
            torch.tensor(ARKIT_LOWER_INDICES, dtype=torch.long),
        )

    def forward(self, shared_encoder_out: torch.Tensor, emotion_emb: torch.Tensor) -> torch.Tensor:
        B, T_30, _ = shared_encoder_out.shape
        face_features = self.face_head(shared_encoder_out, shared_encoder_out, emotion_emb)

        lower_out = self.lower_face_head(face_features)
        upper_linear = self.upper_face_linear(face_features)
        upper_linear = upper_linear.transpose(1, 2)
        upper_out = self.upper_face_conv(upper_linear)
        upper_out = upper_out.transpose(1, 2)

        final_arkit = torch.zeros(B, T_30, 52, device=lower_out.device, dtype=lower_out.dtype)
        final_arkit[:, :, self.arkit_lower_indices] = lower_out
        final_arkit[:, :, self.arkit_upper_indices] = upper_out
        return final_arkit


class BrainModel(nn.Module):
    """Full training-time model (single module tree matching notebook)."""

    def __init__(self, hubert_name: Optional[str] = None):
        super().__init__()
        if HubertModel is None:
            raise ImportError(f"transformers required: {_HUBERT_ERR}")
        hubert_name = hubert_name or _hubert_source()
        self.hubert = HubertModel.from_pretrained(hubert_name)
        for param in self.hubert.feature_extractor.parameters():
            param.requires_grad = False
        for i, layer in enumerate(self.hubert.encoder.layers):
            for param in layer.parameters():
                param.requires_grad = i >= 6

        self.audio_project = nn.Linear(768 + 3, 768)
        self.emotion_conditioning = nn.Sequential(
            nn.Linear(27, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 384),
            nn.LayerNorm(384),
            nn.GELU(),
        )
        self.fusion_input_project = nn.Linear(768, 384)
        self.shared_encoder = SharedEncoder()
        self.face_head = FaceHeadDecoder()
        self.lower_face_head = nn.Sequential(
            nn.Linear(384, 256), nn.GELU(),
            nn.Linear(256, 128), nn.GELU(),
            nn.Linear(128, 29), nn.Sigmoid(),
        )
        self.upper_face_linear = nn.Sequential(
            nn.Linear(384, 384), nn.GELU(),
            nn.Linear(384, 256), nn.GELU(),
            nn.Linear(256, 23),
        )
        self.upper_face_conv = nn.Sequential(
            nn.Conv1d(23, 23, kernel_size=5, padding=2),
            nn.Sigmoid(),
        )
        self.arkit_upper_indices = ARKIT_UPPER_INDICES
        self.arkit_lower_indices = ARKIT_LOWER_INDICES
        self.character_adapter = CharacterAdapter()

    def forward(self, audio_waveform, emotion_26d, intensity, prosody_features, target_T30: Optional[int] = None):
        B, S = audio_waveform.shape
        hubert_out = self.hubert(audio_waveform).last_hidden_state
        T_50 = hubert_out.size(1)

        prosody = prosody_features.transpose(1, 2)
        prosody = F.interpolate(prosody, size=T_50, mode="linear", align_corners=False)
        prosody = prosody.transpose(1, 2)

        audio_feats = torch.cat([hubert_out, prosody], dim=-1)
        audio_feats = self.audio_project(audio_feats)

        T_30 = target_T30 if target_T30 is not None else max(1, int(round(T_50 * 30 / 50)))
        audio_feats = audio_feats.transpose(1, 2)
        audio_feats = F.interpolate(audio_feats, size=T_30, mode="linear", align_corners=False)
        audio_feats = audio_feats.transpose(1, 2)

        if intensity.dim() == 1:
            intensity = intensity.unsqueeze(-1)
        emotion_input = torch.cat([emotion_26d, intensity], dim=-1)
        emotion_emb = self.emotion_conditioning(emotion_input)

        fusion_input = self.fusion_input_project(audio_feats)
        shared_encoder_out = self.shared_encoder(fusion_input, emotion_emb)
        face_features = self.face_head(shared_encoder_out, shared_encoder_out, emotion_emb)

        lower_out = self.lower_face_head(face_features)
        upper_linear = self.upper_face_linear(face_features).transpose(1, 2)
        upper_out = self.upper_face_conv(upper_linear).transpose(1, 2)

        final_arkit = torch.zeros(B, T_30, 52, device=lower_out.device, dtype=lower_out.dtype)
        final_arkit[:, :, self.arkit_lower_indices] = lower_out
        final_arkit[:, :, self.arkit_upper_indices] = upper_out

        # Apply character adapter (inference / optional at train time)
        final_arkit = self.character_adapter(final_arkit)
        return final_arkit


def load_full_brain_state_dict(path: Union[str, Path], device: str = "cpu") -> BrainModel:
    """Load monolithic brain_latest.pt / brain.pt state_dict into BrainModel."""
    path = Path(path)
    model = BrainModel()
    state = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    state = {k.replace("module.", ""): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[brain_model] full load missing={len(missing)} unexpected={len(unexpected)}")
    model.to(device)
    model.eval()
    return model
