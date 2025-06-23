#!/usr/bin/env python3
"""
Setup script to configure TensorFlow Model Optimization environment.
Run this before importing tensorflow or tensorflow_model_optimization.
"""

import os
import sys

def setup_tfmot_environment():
    """Configure environment variables for TFMOT compatibility."""
    
    # Force TensorFlow to use legacy Keras (v2) instead of Keras v3
    os.environ['TF_USE_LEGACY_KERAS'] = '1'
    
    # Disable oneDNN optimizations if causing issues
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    
    # Set memory growth for GPU (if available)
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    
    # Suppress TensorFlow warnings for cleaner output
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    
    print("✅ TFMOT Environment configured:")
    print(f"   - TF_USE_LEGACY_KERAS: {os.environ.get('TF_USE_LEGACY_KERAS')}")
    print(f"   - TF_ENABLE_ONEDNN_OPTS: {os.environ.get('TF_ENABLE_ONEDNN_OPTS')}")
    print(f"   - Python version: {sys.version}")
    
    # Verify TensorFlow and TFMOT versions
    try:
        import tensorflow as tf
        import tensorflow_model_optimization as tfmot
        print(f"   - TensorFlow: {tf.__version__}")
        print(f"   - TFMOT: {tfmot.__version__}")
        print(f"   - Keras: {tf.keras.__version__}")
        
        # Check if we're using the correct Keras version
        if hasattr(tf.keras, '_keras_api_names'):
            print("   - Using TensorFlow's built-in Keras (Legacy)")
        else:
            print("   - Using external Keras")
            
    except ImportError as e:
        print(f"⚠️  Import error: {e}")
        print("   Make sure you've activated the conda environment")
        return False
    
    return True

if __name__ == "__main__":
    setup_tfmot_environment() 