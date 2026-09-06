# third_party/

Clone optional large dependencies here (not vendored in this pack).

## MoMask (open-vocab body motion)

| | |
|--|--|
| **Code link** | https://github.com/EricGuo5513/momask-codes |
| **Weights link** | https://drive.google.com/file/d/1vXS7SHJBgWPt59wupQ5UUzhFObrnGkQ0/view?usp=sharing |

```powershell
git clone --depth 1 https://github.com/EricGuo5513/momask-codes.git third_party\momask-codes
python -m venv third_party\momask_venv
.\third_party\momask_venv\Scripts\Activate.ps1
pip install -U pip
pip install torch torchvision torchaudio   # match your CUDA from pytorch.org
pip install -r third_party\momask-codes\requirements.txt
powershell -ExecutionPolicy Bypass -File tools\download_momask_models.ps1
```

Weights unpack under `third_party/momask-codes/checkpoints/t2m/`.

Without MoMask, the pipeline still runs using **catalog** body Actions in `body_motion/`.
