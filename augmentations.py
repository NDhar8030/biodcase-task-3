import tensorflow as tf
import numpy as np

desired_sr = 16000

@tf.function
def load_wav_16k_mono_tf(filename, label):
    file_contents = tf.io.read_file(filename)
    wav, sample_rate = tf.audio.decode_wav(file_contents, desired_channels=1)
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)

    wav = tf.squeeze(wav, axis=-1)
    label = tf.cast(label, dtype=tf.float32)
    return wav, label

# Helpers
def rand_uniform(low, high):
    # Return a random float in the bounds
    return tf.random.uniform([], low, high)

def apply_with_p(p, fn, x):
    # Apply fn(x) with probability p, else return x.
    return tf.cond(tf.less(tf.random.uniform([], 0, 1), p),
                   lambda: fn(x),
                   lambda: x)

# 1) AddGaussianSNR
def add_gaussian_snr(wav, min_snr=16.0, max_snr=32.0):
    rms_signal = tf.sqrt(tf.reduce_mean(wav**2))
    snr_db = rand_uniform(min_snr, max_snr)
    snr = 10.0**(snr_db / 20.0)
    noise_rms = rms_signal / snr
    noise = tf.random.normal(tf.shape(wav)) * noise_rms
    return wav + noise

# 2) Gain (scalar amplification)
def gain(wav, min_db=-8.0, max_db=24.0):
    db = rand_uniform(min_db, max_db)
    factor = 10.0**(db / 20.0)
    return wav * factor

# 3) TimeStretch (speed perturbation via resampling)
def time_stretch(wav, sr=8000, min_rate=0.9, max_rate=1.112):
    target_length = 32000

    # 1) pick random speed factor
    rate = rand_uniform(min_rate, max_rate)

    # 2) Resample using linear interpolation implemented with pure TF ops
    #    We keep pitch but change tempo by computing samples at `indices` and
    #    performing linear interpolation between neighboring integer indices.
    #    NOTE: `rate` > 1 speeds the audio up (fewer output samples) and
    #    `rate` < 1 slows it down (more output samples).

    orig_len = tf.shape(wav)[0]
    new_len = tf.cast(tf.cast(orig_len, tf.float32) / rate, tf.int32)

    # Build floating-point indices over the original range [0, orig_len-1].
    indices = tf.linspace(0.0, tf.cast(orig_len - 1, tf.float32), new_len)
    indices_floor = tf.cast(tf.floor(indices), tf.int32)
    indices_ceil = tf.minimum(indices_floor + 1, orig_len - 1)

    # Linear-interpolation weights
    weights_ceil = indices - tf.cast(indices_floor, tf.float32)
    weights_floor = 1.0 - weights_ceil

    # Gather and combine – `wav` is expected to be 1-D. The operations stay
    # 1-D to avoid broadcasting to an undesired 2-D shape.
    stretched = (
        tf.gather(wav, indices_floor) * weights_floor +
        tf.gather(wav, indices_ceil) * weights_ceil
    )

    # 4) pad or trim back to original length
    stretched_len = tf.shape(stretched)[0]
    return tf.cond(
        stretched_len < target_length,
        lambda: tf.pad(stretched, [[0, target_length - stretched_len]]),
        lambda: stretched[:target_length]
    )

# 4) PitchShift (via resampling)
def naive_pitch_shift(wav, sample_rate=8000, min_semi=-4.0, max_semi=4.0):
    target_length = 32000
    semi = rand_uniform(min_semi, max_semi)
    rate = 2 ** (semi / 12.0)
    # Convert semitone shift into a resampling factor. Resampling the waveform
    # directly alters both pitch and duration. We compensate duration later by
    # padding/trimming back to `target_length` so the output length stays
    # constant, but the pitch remains shifted.

    # Resample factor: `rate` > 1 raises pitch, < 1 lowers pitch.
    orig_len = tf.shape(wav)[0]
    new_len = tf.cast(tf.cast(orig_len, tf.float32) / rate, tf.int32)

    indices = tf.linspace(0.0, tf.cast(orig_len - 1, tf.float32), new_len)
    indices_floor = tf.cast(tf.floor(indices), tf.int32)
    indices_ceil = tf.minimum(indices_floor + 1, orig_len - 1)

    weights_ceil = indices - tf.cast(indices_floor, tf.float32)
    weights_floor = 1.0 - weights_ceil

    shifted = (
        tf.gather(wav, indices_floor) * weights_floor +
        tf.gather(wav, indices_ceil) * weights_ceil
    )

    stretched_len = tf.shape(shifted)[0]
    return tf.cond(
        stretched_len < target_length,
        lambda: tf.pad(shifted, [[0, target_length - stretched_len]]),
        lambda: shifted[:target_length]
    )

# 5) Shift (roll the signal in time)
def shift(wav, min_shift=-0.5, max_shift=0.5):
    frac = rand_uniform(min_shift, max_shift)
    shift_amt = tf.cast(frac * tf.cast(tf.shape(wav)[0], tf.float32), tf.int32)
    return tf.roll(wav, shift=shift_amt, axis=0)

# 6) AddBackgroundNoise (mix a random bg clip at given SNR)
#   Assumes `bg_files` is a tf.Tensor of strings passed in as a constant.
def add_background_noise(wav, bg_files, min_snr=5.0, max_snr=24.0):
    # Pick one bg file (at graph build time bg_files must be a tf.constant)
    idx = tf.random.uniform([], 0, tf.shape(bg_files)[0], dtype=tf.int32)
    bg_path = bg_files[idx]
    bg_audio, _ = load_wav_16k_mono_tf(bg_path, label=3)
    bg_audio = bg_audio[:tf.shape(wav)[0]]
    bg_audio = tf.pad(bg_audio, [[0, tf.maximum(0, tf.shape(wav)[0] - tf.shape(bg_audio)[0])]])
    # Scale bg to desired SNR
    rms_w = tf.sqrt(tf.reduce_mean(wav**2))
    snr_db = rand_uniform(min_snr, max_snr)
    snr = 10.0**(snr_db / 20.0)
    rms_bg = tf.sqrt(tf.reduce_mean(bg_audio**2))
    scale = rms_w / (snr * (rms_bg + 1e-8))
    return wav + bg_audio * scale

# 7) TimeMask (mask a random time segment to zero)
def time_mask(wav, min_part=0.02, max_part=0.05, min_masks=1, max_masks=4):
    length = tf.shape(wav)[0]
    num_masks = tf.random.uniform([], minval=min_masks, maxval=max_masks + 1, dtype=tf.int32)
    def apply_one_mask(wav, _):
        mask_size = tf.cast(rand_uniform(min_part, max_part) * tf.cast(length, tf.float32), tf.int32)
        start = tf.random.uniform([], 0, length - mask_size, dtype=tf.int32)
        mask = tf.concat([
            tf.ones([start], dtype=wav.dtype),
            tf.zeros([mask_size], dtype=wav.dtype),
            tf.ones([length - start - mask_size], dtype=wav.dtype)
        ], axis=0)
        return wav * mask
    # fold over a range to apply masks
    masked = tf.foldl(apply_one_mask,
                      elems=tf.range(num_masks),
                      initializer=wav)
    return masked

# 8) FrequencyMask (mask a random frequency segment to zero)
def freq_mask(spec, min_part=0.02,max_part=0.05, min_masks=1, max_masks=5):
    freq_len = tf.shape(spec)[1]
    # pick how many masks to apply
    num_masks = tf.random.uniform(
        [], minval=min_masks, maxval=max_masks + 1, dtype=tf.int32)

    def apply_one_mask(s, _):
        # 1) choose mask size in bins
        part = tf.random.uniform([], minval=min_part, maxval=max_part)
        mask_size = tf.cast(part * tf.cast(freq_len, tf.float32), tf.int32)

        # 2) choose start bin
        start = tf.random.uniform([], 0, freq_len - mask_size, dtype=tf.int32)

        # 3) build 1-D mask [freq_len]
        mask1d = tf.concat([
            tf.ones([start], dtype=s.dtype),
            tf.zeros([mask_size], dtype=s.dtype),
            tf.ones([freq_len - start - mask_size], dtype=s.dtype),
        ], axis=0)

        # 4) broadcast to [time, freq]
        mask2d = tf.expand_dims(mask1d, 0)

        # 5) replace masked bins by min value in spec
        min_val = tf.reduce_min(s)
        return s * mask2d + min_val * (1.0 - mask2d)

    # fold over a range to apply masks
    masked = tf.foldl(apply_one_mask,
                      elems=tf.range(num_masks),
                      initializer=spec)
    return masked

# -----------------------------------------------------------------------------
#  SPECTROGRAM-LEVEL AUGMENTATIONS
# -----------------------------------------------------------------------------

def add_gaussian_snr_spec(spec, min_snr=4.0, max_snr=16.0):
    """Add Gaussian noise to a *spectrogram* achieving an SNR in the
    ``[min_snr, max_snr]`` dB range.

    The function is analogous to :pyfunc:`add_gaussian_snr` but operates on a
    2-D (\[time, freq\]) or 3-D (\[time, freq, 1\]) tensor that already
    represents the magnitude / log-magnitude spectrum.

    Parameters
    ----------
    spec : tf.Tensor
        Input spectrogram (``float32``).
    min_snr, max_snr : float
        Bounds for the desired SNR in dB.
    """
    # Compute RMS over *all* bins – identical logic to the waveform version.
    rms_signal = tf.sqrt(tf.reduce_mean(tf.square(spec)))

    snr_db = rand_uniform(min_snr, max_snr)
    snr = 10.0 ** (snr_db / 20.0)

    noise_rms = rms_signal / snr
    noise = tf.random.normal(tf.shape(spec), dtype=spec.dtype) * noise_rms
    return spec + noise


def add_background_noise_spec(spec, bg_specs, min_snr=4.0, max_snr=16.0):
    """Mix a random *background* spectrogram into ``spec`` at a target SNR.

    Assumes ``spec`` has shape ``[T, F]`` *or* ``[T, F, 1]`` and
    ``bg_specs`` is a tensor ``[N, T, F]`` or ``[N, T, F, 1]``.
    """

    # Pick random background spectrogram from pool
    idx = tf.random.uniform([], 0, tf.shape(bg_specs)[0], dtype=tf.int32)
    bg_spec = bg_specs[idx]

    # ----------------------------------------------------------------------------
    # Align TIME dimension – crop then (if needed) pad to match ``spec`` length.
    # ----------------------------------------------------------------------------
    time_len = tf.shape(spec)[0]
    bg_spec = bg_spec[:time_len]  # crop if longer

    bg_len = tf.shape(bg_spec)[0]
    diff = time_len - bg_len  # >= 0 if padding is required

    # If diff>0 we need to pad along the time dimension.
    def _pad():
        rank = tf.rank(spec)
        # paddings: first row is [0, diff] (pad after), others [0,0]
        paddings_first = tf.expand_dims(tf.stack([0, diff]), 0)  # [1,2]
        paddings_rest = tf.zeros(tf.stack([rank - 1, 2]), dtype=tf.int32)
        paddings = tf.concat([paddings_first, paddings_rest], axis=0)
        return tf.pad(bg_spec, paddings)

    bg_spec = tf.cond(diff > 0, _pad, lambda: bg_spec)

    # ----------------------------------------------------------------------------
    # Mix at desired SNR (power domain)
    # ----------------------------------------------------------------------------
    rms_spec = tf.sqrt(tf.reduce_mean(tf.square(spec)))
    rms_bg = tf.sqrt(tf.reduce_mean(tf.square(bg_spec)))

    snr_db = rand_uniform(min_snr, max_snr)
    snr = 10.0 ** (snr_db / 20.0)

    scale = rms_spec / (snr * (rms_bg + 1e-8))
    spec = spec + bg_spec * scale 
    return spec #(spec - tf.reduce_mean(spec)) / (tf.math.reduce_std(spec) + 1e-6) # Normalize