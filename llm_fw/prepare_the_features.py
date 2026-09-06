import os
import shutil
import urllib.request
import zipfile
import glob
import warnings
import torch
import librosa
import numpy as np
import soundfile as sf
import kagglehub
from datasets import load_dataset
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

# Suppress Librosa warnings for cleaner console output
warnings.filterwarnings('ignore', category=UserWarning)

# Use a local relative directory
base_dir = os.path.abspath("./corpora")

def download_and_setup_data():
    print("--- STEP 1: DOWNLOADING DATASETS (Local Setup) ---")

    # 1. CREMA-D Setup
    print("🚀 [1/4] Downloading CREMA-D...")
    crema_cache = kagglehub.dataset_download("ejlok1/cremad")
    crema_dest = os.path.join(base_dir, "CREMA-D", "AudioWAV")
    os.makedirs(crema_dest, exist_ok=True)
    for root, dirs, files in os.walk(crema_cache):
        for f in files:
            if f.endswith(".wav") and not os.path.exists(os.path.join(crema_dest, f)):
                shutil.copy2(os.path.join(root, f), os.path.join(crema_dest, f))

    # 2. RAVDESS Setup (Using native Python for Windows compatibility)
    print("🚀 [2/4] Downloading RAVDESS...")
    ravdess_dir = os.path.join(base_dir, "ravdess")
    os.makedirs(ravdess_dir, exist_ok=True)
    ravdess_zip = "ravdess.zip"
    if not os.path.exists(ravdess_zip) and not os.listdir(ravdess_dir):
        urllib.request.urlretrieve("https://zenodo.org/records/1188976/files/Audio_Speech_Actors_01-24.zip?download=1", ravdess_zip)
        with zipfile.ZipFile(ravdess_zip, 'r') as zip_ref:
            zip_ref.extractall(ravdess_dir)
        os.remove(ravdess_zip)

    # 3. TESS Setup
    print("🚀 [3/4] Downloading TESS...")
    tess_cache = kagglehub.dataset_download("ejlok1/toronto-emotional-speech-set-tess")
    tess_root = os.path.join(base_dir, "TESS")
    for root, dirs, files in os.walk(tess_cache):
        for f in files:
            if f.endswith(".wav"):
                parent_folder = os.path.basename(root)
                dest_folder = os.path.join(tess_root, parent_folder)
                os.makedirs(dest_folder, exist_ok=True)
                if not os.path.exists(os.path.join(dest_folder, f)):
                    shutil.copy2(os.path.join(root, f), os.path.join(dest_folder, f))

    # 4. JL Corpus Setup
    print("🚀 [4/4] Downloading JL Corpus...")
    jl_cache = kagglehub.dataset_download("tli725/jl-corpus")
    jl_root = os.path.join(base_dir, "JL_corpus")
    os.makedirs(jl_root, exist_ok=True)
    for root, dirs, files in os.walk(jl_cache):
        for f in files:
            if f.endswith(".wav") and not os.path.exists(os.path.join(jl_root, f)):
                shutil.copy2(os.path.join(root, f), os.path.join(jl_root, f))

    print("\n--- STEP 2: LINKING & VALIDATING PATHS WITH HUGGING FACE ---")
    ds = load_dataset("myned-ai/audio2face-emotion-arkit-teacher")

    CORPUS_ROOTS = {
        
        "cremad": os.path.join(base_dir, "CREMA-D"),
        "ravdess": os.path.join(base_dir, "ravdess"),
        "tess": os.path.join(base_dir, "TESS"),
        "jl_corpus": os.path.join(base_dir, "JL_corpus"),
    }

    print("\n=== FINAL DATA PATH STATUS VERIFICATION ===")
    for source in CORPUS_ROOTS.keys():
        match = next((row for row in ds["train"] if row["source"] == source), None)
        if match:
            # Normalize path for Windows compatibility
            hint = match["audio_path_hint"].replace("/", os.sep)
            full_path = os.path.join(CORPUS_ROOTS[source], hint)
            
            if os.path.exists(full_path):
                audio, sr = sf.read(full_path)
                print(f"✅ {source.upper():<9} -> Linked Successfully! Waveform shape: {audio.shape}")
            else:
                print(f"❌ {source.upper():<9} -> Link Failed! Missing: {full_path}")
                
    return CORPUS_ROOTS

# =====================================================================
# MULTIPROCESSING: PROSODY EXTRACTION
# =====================================================================
def process_single_audio(audio_path):
    try:
        output_path = audio_path.replace(".wav", "_prosody.pt")
        if os.path.exists(output_path):
            return True # Skip if already processed
            
        y, sr = librosa.load(audio_path, sr=16000, mono=True)
        
        # 1. Pitch Extraction (F0)
        f0, _, _ = librosa.pyin(
            y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), 
            sr=16000, frame_length=512
        )
        f0 = np.nan_to_num(f0)
        
        # 2. Energy (RMS)
        rms = librosa.feature.rms(y=y, frame_length=512, hop_length=256)[0]
        
        # 3. Speaking Rate
        onset_env = librosa.onset.onset_strength(y=y, sr=16000, hop_length=256)
        
        # Interpolate to 50 FPS
        target_frames = int((len(y) / 16000) * 50)
        if target_frames == 0: target_frames = 150
            
        t_target = np.linspace(0, 1, target_frames)
        f0_interp = np.interp(t_target, np.linspace(0, 1, len(f0)), f0)
        rms_interp = np.interp(t_target, np.linspace(0, 1, len(rms)), rms)
        rate_interp = np.interp(t_target, np.linspace(0, 1, len(onset_env)), onset_env)
        
        prosody_matrix = np.stack([f0_interp, rms_interp, rate_interp], axis=-1)
        torch.save(torch.tensor(prosody_matrix, dtype=torch.float32), output_path)
        return True
    except Exception as e:
        print(f"Error on {audio_path}: {e}")
        return False

if __name__ == '__main__':
    # 1. Download and verify dataset
    corpus_roots = download_and_setup_data()
    
    # 2. Scrape all downloaded wav files
    audio_files = []
    for source, root_path in corpus_roots.items():
        found = glob.glob(os.path.join(root_path, "**/*.wav"), recursive=True)
        audio_files.extend(found)
        
    print(f"\n--- STEP 3: PRE-COMPUTING LIBROSA FEATURES ON CPU ---")
    print(f"📋 Found {len(audio_files)} audio files. Launching parallel processing...")
    
    # Run heavily parallelized on local CPU cores
    num_cores = max(1, os.cpu_count() - 2) # Leave 2 cores free so PC doesn't freeze
    
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        results = list(tqdm(executor.map(process_single_audio, audio_files), total=len(audio_files)))
        
    successful = sum(1 for r in results if r)
    print(f"\n🎉 Complete! Successfully saved {successful}/{len(audio_files)} prosody tensors.")
    print(f"Next: Zip the '{base_dir}' folder and upload to Google Drive.")