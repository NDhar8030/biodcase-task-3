import json
from functools import partial
import faulthandler

import numpy as np
import pandas as pd
import tensorflow as tf
import plotly.graph_objects as go
import pyarrow as pa
from plotly.subplots import make_subplots
from tqdm import tqdm

from config import load_config, Config
from paths import PREPROC_PRQ_PATH, FEATURES_PRQ_PATH, FEATURES_SAMPLE_PLOT_PATH, FEATURES_SHAPE_JSON_PATH


def plot_features_sample(sample: pd.DataFrame, features_shape):
    """Plot a few samples of features
    """
    fmin = sample["features"].apply(lambda x: x.min()).min()
    fmax = sample["features"].apply(lambda x: x.max()).max()

    fig = make_subplots(
        rows=len(sample) + 1,
        cols=2,
        vertical_spacing=0.05,
        subplot_titles=[x.split("/")[-1] for x in list(sample["path"]) for _ in range(2)],
    )
    for i, (idx, row) in enumerate(sample.iterrows()):
        path, audio_data, sample_rate = row["path"], row["data"], row["sample_rate"]
        fig.add_trace(
            go.Heatmap(
                z=np.array(row["features"]).reshape(features_shape).T,
                zmin=fmin,
                zmax=fmax,
                name=path,
                showscale=False,
            ),
            row=i + 1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=list(range(audio_data.shape[0])),
                y=audio_data,
                mode="lines",
                name=path,
                showlegend=False,
            ),
            row=i + 1,
            col=2,
        )
    fig.update_layout(height=500 * len(sample), title_text="Features")
    return fig


def get_features_shape(clip_len_ms, clip_sample_rate, window_len_samples, window_stride_samples, mel_n_channels):
    clip_len_samples = int((clip_len_ms / 1000) * clip_sample_rate)
    return [(clip_len_samples - window_len_samples) // window_stride_samples + 1, mel_n_channels]


def amplitude_to_db(x: tf.Tensor, top_db: float = 80.0) -> tf.Tensor:
    """Convert an amplitude spectrogram to dB-scaled spectrogram."""
    log_spec = 10.0 * tf.math.log(tf.maximum(x, 1e-20)) / tf.math.log(10.0)
    max_val = tf.reduce_max(log_spec)
    return tf.maximum(log_spec, max_val - top_db)


def wav_to_log_mel_spectrogram(
    wav: np.ndarray, params
) -> np.ndarray:
    """Compute log-mel-spectrogram using TensorFlow ops and return as numpy array."""
    # Convert to float32 in [-1, 1]
    wav_f32 = tf.cast(wav, tf.float32) / 32767.0

    # Pad if shorter than desired
    desired_len = params.pad_to_samples
    wav_len = tf.shape(wav_f32)[0]

    def _pad():
        return tf.concat([wav_f32, tf.zeros([desired_len - wav_len])], axis=0)

    def _crop():
        return wav_f32[: desired_len]

    wav_f32 = tf.cond(
        wav_len < desired_len,
        _pad,
        _crop,
    )

    # STFT -> magnitude spectrogram
    stfts = tf.signal.stft(
        wav_f32,
        frame_length=params.frame_length,
        frame_step=params.frame_step,
        fft_length=params.frame_length,
        window_fn=tf.signal.hann_window,
    )
    spectrogram = tf.abs(stfts)

    # Mel scale
    mel_mat = tf.signal.linear_to_mel_weight_matrix(
        params.mel_bins,
        params.frame_length // 2 + 1,
        params.sample_rate,
        lower_edge_hertz=params.fmin,
        upper_edge_hertz=params.fmax,
    )
    mel_spec = tf.matmul(spectrogram, mel_mat)

    # dB scaling
    mel_db = amplitude_to_db(mel_spec, top_db=80.0)

    return mel_db.numpy().astype(np.float32)


def run_feature_extraction(config: Config):
    fe_params = config.feature_extraction

    # Load the pre-processed dataset
    df = pd.read_parquet(PREPROC_PRQ_PATH)

    # Helper to convert raw bytes to int16 numpy array
    def bytes_to_array(b: bytes) -> np.ndarray:
        return np.frombuffer(b, dtype=np.int16)

    # Prepare containers
    feature_vectors: list[np.ndarray] = []

    print("Extracting log-mel spectrograms…")
    for raw in tqdm(df["data"], total=len(df)):
        wav_int16 = bytes_to_array(raw)
        mel_db = wav_to_log_mel_spectrogram(wav_int16, fe_params)
        feature_vectors.append(mel_db.flatten())

    # Determine feature shape from first element
    feature_shape = feature_vectors[0].reshape(-1, fe_params.mel_bins).shape

    df["features"] = feature_vectors

    # Plot a sample of 10 rows for visual inspection (before dropping raw audio)
    sample_df = df.head(10).copy()
    sample_df["features"] = sample_df["features"].apply(lambda x: x.reshape(feature_shape))
    sample_df["data"] = sample_df["data"].apply(bytes_to_array)

    fig = plot_features_sample(sample_df, feature_shape)

    # Drop raw audio bytes now that we're done plotting
    df = df.drop(columns=["data"])  # Drop raw audio bytes

    FEATURES_PRQ_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(
        FEATURES_PRQ_PATH,
        engine="pyarrow",
        compression="snappy",
        schema={"features": pa.list_(pa.float32())},
        index=False,
    )

    FEATURES_SAMPLE_PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(FEATURES_SAMPLE_PLOT_PATH)

    with FEATURES_SHAPE_JSON_PATH.open("w") as f:
        json.dump(feature_shape, f)

    print(f"Saved features to {FEATURES_PRQ_PATH} and plot to {FEATURES_SAMPLE_PLOT_PATH}")


if __name__ == "__main__":
    config = load_config()
    run_feature_extraction(config)