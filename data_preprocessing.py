from pathlib import Path
from typing import List

import tensorflow as tf
import numpy as np
import pandas as pd
import pyarrow as pa
import audiomentations

from config import load_config, Config
from paths import CLIPS_DIR, PREPROC_PRQ_PATH

_PREPROC_DASK_BATCH_SIZE = 1000

SPLIT_FOLDER_TO_SPLIT = {"Training_Set": "train", "Validation_Set": "validation", "Test_Set": "test"}

# Augmentation pipeline (executed on float32 signal in [-1,1])
BACKGROUND_NOISE_DIR = Path("data/01_raw/clips/Training_Set/Negatives")
# Fallback to no augment if directory missing
if BACKGROUND_NOISE_DIR.exists():
    augmenter = audiomentations.Compose([
        audiomentations.AddGaussianSNR(min_snr_db=4.0, max_snr_db=16.0, p=0.3),
        audiomentations.AddBackgroundNoise(
            sounds_path=str(BACKGROUND_NOISE_DIR),
            min_snr_db=4.0,
            max_snr_db=16.0,
            p=0.3,
        ),
    ])
else:
    augmenter = None

# ------------------------------
# Helper functions
# ------------------------------

def load_wav_tf(path: Path, desired_sr: int) -> tuple[np.ndarray, int]:
    """Load a WAV file with TensorFlow and return a mono int16 numpy array."""
    file_contents = tf.io.read_file(str(path))
    audio, sr = tf.audio.decode_wav(file_contents, desired_channels=1)
    audio = tf.squeeze(audio, axis=-1)  # shape [N]

    # Resampling (if needed) – TensorFlow lacks built-in resample.  Assume all
    # clips are already at the desired sample rate.  Otherwise, raise.
    sr_val = int(sr.numpy())
    if sr_val != desired_sr:
        raise ValueError(f"Expected sample rate {desired_sr}, got {sr_val} in {path}")

    # Convert float32 in [-1, 1] ➜ int16
    audio_int16 = tf.cast(tf.clip_by_value(audio, -1.0, 1.0) * 32767.0, tf.int16)
    return audio_int16.numpy(), sr_val


def extract_loudest_slice(array: np.ndarray, sample_rate: int, slice_dur_ms: int) -> np.ndarray:
    """Return a fixed-length slice centred on the max absolute amplitude."""
    slice_samples = int(slice_dur_ms / 1000 * sample_rate)
    if array.shape[0] <= slice_samples:
        return array  # Already short enough

    max_idx = int(np.argmax(np.abs(array)))
    left = slice_samples // 2
    start = max(max_idx - left, 0)
    end = start + slice_samples
    if end > array.shape[0]:
        end = array.shape[0]
        start = end - slice_samples
    return array[start:end]


def run_preprocessing(config: Config):
    desired_sr = config.feature_extraction.sample_rate

    records: List[dict] = []
    clips = list(CLIPS_DIR.rglob("*.wav"))
    for idx, wav_path in enumerate(clips):
        audio_int16, sr_val = load_wav_tf(wav_path, desired_sr)

        # Convert to float32 [-1,1]
        wav_f32 = audio_int16.astype(np.float32) / 32767.0

        # (Optional) loudest slice cropping for bird call localisation
        wav_f32 = extract_loudest_slice(
            wav_f32,
            sr_val,
            config.data_preprocessing.audio_slice_duration_ms,
        )

        # Apply waveform augmentation if pipeline exists
        if augmenter is not None:
            wav_f32 = augmenter(samples=wav_f32, sample_rate=sr_val)

        # Convert back to int16 for storage
        audio_slice = (np.clip(wav_f32, -1.0, 1.0) * 32767.0).astype(np.int16)

        records.append({
            "data": audio_slice.tobytes(),
            "path": str(wav_path),
            "label": np.int32(wav_path.parent.name == "Yellowhammer"),
            "split": SPLIT_FOLDER_TO_SPLIT[wav_path.parents[1].name],
            "sample_rate": np.int32(sr_val),
        })

        if idx < 3:
            print(f"Loaded {wav_path.name}: shape {audio_slice.shape}, dtype {audio_slice.dtype}")

    df = pd.DataFrame(records)

    # Parquet schema
    schema = pa.schema([
        ("data", pa.binary()),
        ("path", pa.string()),
        ("split", pa.string()),
        ("label", pa.int32()),
        ("sample_rate", pa.int32()),
    ])

    PREPROC_PRQ_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(
        PREPROC_PRQ_PATH,
        engine="pyarrow",
        compression="snappy",
        schema=schema,
        index=False,
    )

    print(f"Wrote preprocessed dataset to {PREPROC_PRQ_PATH} (rows={len(df)})")


if __name__ == "__main__":
    config = load_config()
    run_preprocessing(config)
