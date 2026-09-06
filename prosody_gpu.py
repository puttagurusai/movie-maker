"""
prosody_gpu.py

Ultra-fast GPU prosody extractor for Brain input #4.

Channels @ 30 fps, shape [T, 3]:
  0: F0 Pitch (normalized 0–1) via torchcrepe
  1: Energy RMS (normalized 0–1) via Spectrogram
  2: Speaking Rate / Onset Density (normalized 0–1) via Spectral Flux + Gaussian smooth

Requires: torch, torchaudio, torchcrepe
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn.functional as F
import torchaudio

try:
    import torchcrepe
except ImportError as e:  # pragma: no cover
    torchcrepe = None
    _CREPE_ERR = e
else:
    _CREPE_ERR = None

TARGET_FPS = 30
CREPE_HOP = 160          # 10 ms @ 16 kHz → 100 fps
ENERGY_HOP_30 = 533      # ≈ 16000 / 30
FMIN, FMAX = 50.0, 400.0


def normalize_01(tensor: torch.Tensor) -> torch.Tensor:
    """Min-max normalize to [0, 1]."""
    t_min, t_max = tensor.min(), tensor.max()
    if t_max - t_min < 1e-6:
        return torch.zeros_like(tensor)
    return (tensor - t_min) / (t_max - t_min)


def extract_prosody(
    wav_path: str,
    device: Optional[str] = None,
    save_path: Optional[str] = None,
) -> torch.Tensor:
    """
    Extract a 3-channel prosody feature tensor at 30 fps (GPU when available).

    Channels:
      0: F0 Pitch (normalized 0–1) via torchcrepe
      1: Energy RMS (normalized 0–1) via Spectrogram
      2: Speaking Rate / Onset Density (normalized 0–1) via Spectral Flux

    Returns:
      torch.Tensor of shape [T, 3] on the specified device.
    """
    if torchcrepe is None:
        raise ImportError(
            "torchcrepe is required for prosody_gpu.\n"
            "  pip install torchcrepe\n"
            f"Original error: {_CREPE_ERR}"
        )

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    # Keep string for torchcrepe + torch.device for tensors
    device_str = device
    dev = torch.device(device_str)

    # -------------------------------------------------------------------------
    # 1. LOAD AUDIO DIRECTLY TO GPU
    # -------------------------------------------------------------------------
    wav, sr = torchaudio.load(wav_path)
    wav = wav.to(dev)

    # Force Mono
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)

    # Force 16 kHz
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000).to(dev)
        wav = resampler(wav)
        sr = 16000

    # Target frame count at 30 fps
    duration = wav.shape[1] / sr
    target_T = max(1, int(duration * TARGET_FPS))

    # -------------------------------------------------------------------------
    # CHANNEL 0: F0 PITCH (torchcrepe)
    # -------------------------------------------------------------------------
    # Hop length 160 = 10 ms frame step at 16 kHz (100 fps)
    pitch, periodicity = torchcrepe.predict(
        wav,
        sr,
        hop_length=CREPE_HOP,
        fmin=FMIN,
        fmax=FMAX,
        model="tiny",
        device=device_str,
        return_periodicity=True,
    )

    # Fill unvoiced/silent regions with mean pitch
    voiced_mask = periodicity > 0.3
    if voiced_mask.any():
        mean_pitch = pitch[voiced_mask].mean()
        pitch = pitch.clone()
        pitch[~voiced_mask] = mean_pitch
    else:
        pitch = torch.zeros_like(pitch)

    # Resample from 100 fps to exact 30 fps length
    pitch_3d = pitch.unsqueeze(0)  # [1, 1, T_100]
    pitch_30fps = F.interpolate(
        pitch_3d, size=target_T, mode="linear", align_corners=False
    ).squeeze()

    # -------------------------------------------------------------------------
    # CHANNEL 1: ENERGY RMS (Spectrogram)
    # -------------------------------------------------------------------------
    # Hop length 533 ~ 30 fps at 16 kHz (16000 / 30 ≈ 533.33)
    spectrogram_transform = torchaudio.transforms.Spectrogram(
        n_fft=1024,
        hop_length=ENERGY_HOP_30,
        power=2.0,
    ).to(dev)

    spec = spectrogram_transform(wav)  # [1, Freq, T_spec]
    rms_energy = torch.sqrt(spec.mean(dim=1) + 1e-9)  # [1, T_spec]

    # Ensure exact 30 fps frame count
    rms_30fps = F.interpolate(
        rms_energy.unsqueeze(0), size=target_T, mode="linear", align_corners=False
    ).squeeze()

    # -------------------------------------------------------------------------
    # CHANNEL 2: SPEAKING RATE (Spectral Flux / Onset Density)
    # -------------------------------------------------------------------------
    spec_mag = torch.sqrt(spec + 1e-9).squeeze(0)  # [Freq, T_spec]
    spectral_diff = torch.relu(spec_mag[:, 1:] - spec_mag[:, :-1])
    onset_env = spectral_diff.sum(dim=0)
    onset_env = F.pad(onset_env, (1, 0))  # Maintain time shape

    # Smooth using 1D Gaussian kernel on GPU
    kernel_size = 5
    sigma = 1.0
    x_cord = torch.arange(kernel_size, dtype=torch.float32, device=dev) - (kernel_size - 1) / 2
    gaussian_kernel = torch.exp(-(x_cord ** 2) / (2 * sigma ** 2))
    gaussian_kernel = (gaussian_kernel / gaussian_kernel.sum()).view(1, 1, -1)

    smoothed_onset = F.conv1d(
        onset_env.view(1, 1, -1),
        gaussian_kernel,
        padding=kernel_size // 2,
    )

    rate_30fps = F.interpolate(
        smoothed_onset, size=target_T, mode="linear", align_corners=False
    ).squeeze()

    # -------------------------------------------------------------------------
    # NORMALIZE (0.0 to 1.0 Range)
    # -------------------------------------------------------------------------
    # Ensure 1D even for very short clips
    if pitch_30fps.dim() == 0:
        pitch_30fps = pitch_30fps.unsqueeze(0)
        rms_30fps = rms_30fps.unsqueeze(0)
        rate_30fps = rate_30fps.unsqueeze(0)

    f0_norm = normalize_01(pitch_30fps.float())
    energy_norm = normalize_01(rms_30fps.float())
    rate_norm = normalize_01(rate_30fps.float())

    # Align lengths (safety)
    T = min(f0_norm.numel(), energy_norm.numel(), rate_norm.numel(), target_T)
    f0_norm = f0_norm[:T]
    energy_norm = energy_norm[:T]
    rate_norm = rate_norm[:T]

    # -------------------------------------------------------------------------
    # STACK TO [T, 3]
    # -------------------------------------------------------------------------
    prosody_tensor = torch.stack([f0_norm, energy_norm, rate_norm], dim=-1)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(prosody_tensor.detach().cpu(), save_path)
        print(f"[prosody_gpu] Saved {save_path}  shape={tuple(prosody_tensor.shape)}")

    return prosody_tensor


def extract_prosody_batch(
    wav_paths: list[str],
    device: Optional[str] = None,
    out_dir: Optional[Union[str, Path]] = None,
) -> list[torch.Tensor]:
    """Extract prosody for many files; optionally save each as <stem>_prosody.pt."""
    out = []
    out_dir = Path(out_dir) if out_dir else None
    for p in wav_paths:
        save = None
        if out_dir is not None:
            save = str(out_dir / f"{Path(p).stem}_prosody.pt")
        out.append(extract_prosody(p, device=device, save_path=save))
    return out


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python prosody_gpu.py <audio.wav> [out.pt]")
        sys.exit(1)
    path = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else None
    if out is None:
        out = str(Path(path).with_name(Path(path).stem + "_prosody.pt"))
    t = extract_prosody(path, save_path=out)
    print(f"prosody shape={tuple(t.shape)} device={t.device}")
    print(f"  ch0 pitch  mean={t[:, 0].mean():.3f} max={t[:, 0].max():.3f}")
    print(f"  ch1 energy mean={t[:, 1].mean():.3f} max={t[:, 1].max():.3f}")
    print(f"  ch2 rate   mean={t[:, 2].mean():.3f} max={t[:, 2].max():.3f}")
