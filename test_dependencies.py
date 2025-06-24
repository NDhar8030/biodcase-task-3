#!/usr/bin/env python3
"""
Test script to verify that all critical dependencies work correctly.
Run this after creating the conda environment to ensure compatibility.
"""

import sys
print("🧪 Testing BioDCASE-Tiny Dependencies...")
print(f"Python version: {sys.version}")

# Test core ML packages
try:
    import numpy as np
    print(f"✅ NumPy {np.__version__}")
except ImportError as e:
    print(f"❌ NumPy import failed: {e}")

try:
    import tensorflow as tf
    print(f"✅ TensorFlow {tf.__version__}")
except ImportError as e:
    print(f"❌ TensorFlow import failed: {e}")

try:
    import tensorflow_model_optimization as tfmot
    print(f"✅ TFMOT {tfmot.__version__}")
except ImportError as e:
    print(f"❌ TFMOT import failed: {e}")

try:
    import keras
    print(f"✅ Keras {keras.__version__}")
except ImportError as e:
    print(f"❌ Keras import failed: {e}")

# Test audio processing
try:
    import librosa
    print(f"✅ Librosa {librosa.__version__}")
except ImportError as e:
    print(f"❌ Librosa import failed: {e}")

try:
    import soundfile as sf
    print(f"✅ SoundFile {sf.__version__}")
except ImportError as e:
    print(f"❌ SoundFile import failed: {e}")

# Test scientific computing
try:
    import scipy
    print(f"✅ SciPy {scipy.__version__}")
except ImportError as e:
    print(f"❌ SciPy import failed: {e}")

try:
    import sklearn
    print(f"✅ Scikit-learn {sklearn.__version__}")
except ImportError as e:
    print(f"❌ Scikit-learn import failed: {e}")

# Test custom utilities
try:
    from numpy_utils import minmax, rms, normalize_minmax
    test_arr = np.array([1, 2, 3, 4, 5])
    min_val, max_val = minmax(test_arr)
    rms_val = rms(test_arr)
    norm_arr = normalize_minmax(test_arr)
    print(f"✅ Custom NumPy utilities working (min={min_val}, max={max_val}, rms={rms_val:.2f})")
except Exception as e:
    print(f"❌ Custom NumPy utilities failed: {e}")

# Test TFMOT compatibility
print("\n🔧 Testing TFMOT Compatibility...")
try:
    # Create a simple Functional model (required for TFMOT)
    from keras import Model, layers
    inputs = layers.Input(shape=(10,))
    x = layers.Dense(8, activation='relu', name='dense1')(inputs)
    outputs = layers.Dense(1, activation='sigmoid', name='dense2')(x)
    model = Model(inputs, outputs, name='test_model')
    
    # Compile the model (required for quantization)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Test quantization-aware training
    import tensorflow_model_optimization as tfmot
    quantize_model = tfmot.quantization.keras.quantize_model
    q_aware_model = quantize_model(model)
    print("✅ Quantization-aware training works")
    
    # Test pruning
    prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
    pruning_params = {
        'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
            initial_sparsity=0.30, final_sparsity=0.50, 
            begin_step=0, end_step=100
        )
    }
    pruned_model = prune_low_magnitude(model, **pruning_params)
    print("✅ Magnitude-based pruning works")
    
    # Test that the models are callable
    import numpy as np
    test_input = np.random.random((1, 10))
    _ = model(test_input)
    _ = q_aware_model(test_input) 
    _ = pruned_model(test_input)
    print("✅ All TFMOT models are functional")
    
except Exception as e:
    print(f"❌ TFMOT compatibility test failed: {e}")

# Test audio processing pipeline
print("\n🎵 Testing Audio Processing Pipeline...")
try:
    import librosa
    import numpy as np
    
    # Generate test audio signal
    sr = 22050
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration))
    test_audio = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
    
    # Test feature extraction
    mfccs = librosa.feature.mfcc(y=test_audio, sr=sr, n_mfcc=13)
    mel_spec = librosa.feature.melspectrogram(y=test_audio, sr=sr)
    stft = librosa.stft(test_audio)
    
    print(f"✅ Audio feature extraction works (MFCC: {mfccs.shape}, Mel: {mel_spec.shape}, STFT: {stft.shape})")
    
except Exception as e:
    print(f"❌ Audio processing test failed: {e}")

print("\n🎯 Environment Test Complete!")
print("If all tests passed, your environment is ready for BioDCASE-Tiny development with TFMOT.") 