import yaml
from pydantic import BaseModel, field_validator, Field

from paths import PIPELINE_CONFIG_FILE


class DataPreprocessing(BaseModel):
    audio_slice_duration_ms: int
    sample_rate: int


class FeatureExtraction(BaseModel):
    """Parameters for feature extraction.

    The original fixed-point DSP pipeline (window_len, mel_n_channels, …) is kept
    for backward compatibility, but new TensorFlow parameters are added for the
    TF-based preprocessing path.  Any YAML config file may contain either the
    old keys, the new keys, or both.
    """

    # Legacy (fixed-point C pipeline) params – optional now
    window_len: int | None = None
    window_stride: int | None = None
    window_scaling_bits: int | None = None
    mel_n_channels: int | None = None
    mel_low_hz: int | None = None
    mel_high_hz: int | None = None
    mel_post_scaling_bits: int | None = None

    # New TF preprocessing params (defaults follow user request)
    sample_rate: int = 16000
    frame_length: int = 256
    frame_step: int = 192  # 75 % overlap
    mel_bins: int = 64
    fmin: int = 1028
    fmax: int = 8000
    pad_to_samples: int = 32000
    add_channel_dim: bool = True

    # Keep original validator but guard against missing legacy keys
    @field_validator('mel_high_hz')
    @classmethod
    def validate_mel_high_hz(cls, v, values):
        if v is not None and values.data.get('mel_low_hz') is not None:
            if v <= values.data['mel_low_hz']:
                raise ValueError('mel_high_hz must be strictly greater than mel_low_hz')
        return v


class ModelTraining(BaseModel):
    class EarlyStopping(BaseModel):
        patience: int
    seed: int
    n_epochs: int
    shuffle_buff_n: int
    batch_size: int
    early_stopping: EarlyStopping


class EmbeddedCodeGeneration(BaseModel):
    serial_device: str


class Config(BaseModel):
    data_preprocessing: DataPreprocessing
    feature_extraction: FeatureExtraction
    model_training: ModelTraining
    embedded_code_generation: EmbeddedCodeGeneration


def load_config() -> Config:
    with PIPELINE_CONFIG_FILE.open("r") as file:
        yaml_data = yaml.safe_load(file)
    return Config(**yaml_data)
