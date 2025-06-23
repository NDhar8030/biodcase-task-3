from functools import partial
from pathlib import Path
from typing import Iterable

import librosa
import numpy as np
import pandas as pd
import dask.dataframe as dd
import dask
from tqdm.dask import TqdmCallback
import pyarrow as pa

import audiomentations

from config import load_config, Config
from paths import CLIPS_DIR, PREPROC_PRQ_PATH

_PREPROC_DASK_BATCH_SIZE = 1000

SPLIT_FOLDER_TO_SPLIT = {"Training_Set": "train", "Validation_Set": "validation", "Test_Set": "test"}

'''
BACKGROUND_NOISE_DIR = "data/01_raw/clips/Training_Set/Negatives/"
augmenter = audiomentations.Compose([
    audiomentations.AddGaussianSNR(min_snr_db=4.0, max_snr_db=16.0, p=0.3),
    audiomentations.AddBackgroundNoise(
        sounds_path=BACKGROUND_NOISE_DIR,
        min_snr_db=4,
        max_snr_db=16,
        p=0.3
    )
])
'''
 

def extract_loudest_slice(audio_array, sample_rate, audio_slice_duration_ms):
    """Find the max of the audio, then return a slice of duration audio_slice_duration_ms centred on the max.
    Also deal with boundary conditions.

    :param audio_slice_duration_ms:
    :return: slice with duration audio_slice_duration_ms
    """
    slice_n_samples = int(audio_slice_duration_ms / 1000 * sample_rate)
    audio_n_samples = audio_array.shape[0]
    left_edge = slice_n_samples // 2
    right_edge = slice_n_samples - left_edge

    max_index = np.argmax(audio_array)
    start_index = max(max_index - left_edge, 0)
    end_index = min(max_index + right_edge, audio_n_samples)

    # Handle edge cases where the slice would go beyond bounds
    if end_index - start_index < slice_n_samples:
        if start_index == 0:  # Cut at left edge
            end_index = slice_n_samples
        else:  # Cut at right edge
            start_index = audio_n_samples - slice_n_samples
            end_index = audio_n_samples
    return audio_array[start_index:end_index]


def run_preprocessing(config: Config):
    def do_batch(batch: Iterable[Path]):
        slice_data = []
        for i, path in enumerate(batch):
            # Load audio with consistent dtype
            audio_array, sample_rate = librosa.load(path, sr=config.data_preprocessing.sample_rate, mono=True, dtype=np.float32)
            
            # Extract loudest slice while keeping as float32
            slice = extract_loudest_slice(audio_array, sample_rate, config.data_preprocessing.audio_slice_duration_ms)
            
            # Apply augmentation (this returns float32)
            #slice = augmenter(samples=slice, sample_rate=sample_rate)
            
            # Convert to int16 after all processing is done
            slice = (slice * np.iinfo(np.int16).max).astype(np.int16)
            
            slice_data.append({
                "data": slice.tobytes(),  # Store raw bytes directly
                "path": str(path),
                "label": np.int32(path.parent.name == "Yellowhammer"),  # Explicitly use int32
                "split": SPLIT_FOLDER_TO_SPLIT[path.parents[1].name],
                "sample_rate": np.int32(sample_rate)  # Explicitly use int32
            })
            
            if i < 1:
                #print("Successfully augmented slice")
                print(f"Data type: {slice.dtype}, Shape: {slice.shape}")
        
        return pd.DataFrame(slice_data)

    clips = list(CLIPS_DIR.rglob("*.wav"))

    batches = []
    total_batches = (len(clips) + _PREPROC_DASK_BATCH_SIZE - 1) // _PREPROC_DASK_BATCH_SIZE
    for batch_idx in range(total_batches):
        start_idx = batch_idx * _PREPROC_DASK_BATCH_SIZE
        end_idx = min(start_idx + _PREPROC_DASK_BATCH_SIZE, len(clips))
        batches.append(clips[start_idx:end_idx])

    # Define schema explicitly using PyArrow
    schema = pa.schema([
        ('data', pa.binary()),
        ('path', pa.string()),
        ('split', pa.string()),
        ('label', pa.int32()),
        ('sample_rate', pa.int32())
    ])

    with TqdmCallback(desc="Preprocessing clips in batches"):
        dask.config.set({"dataframe.convert-string": False})
        all_data: dd.DataFrame = dd.from_map(
            do_batch, batches, meta=pd.DataFrame({
                "data": pd.Series([], dtype=object),
                "path": pd.Series(dtype="string"),
                "split": pd.Series(dtype="string"),
                "label": pd.Series(dtype="int32"),
                "sample_rate": pd.Series(dtype="int32")
            })
        )
        all_data.to_parquet(
            PREPROC_PRQ_PATH,
            write_index=False,
            engine='pyarrow',
            compression='snappy',
            schema=schema  # Use the explicit schema
        )


if __name__ == "__main__":
    config = load_config()
    run_preprocessing(config)
