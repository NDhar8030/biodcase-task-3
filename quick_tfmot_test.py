#!/usr/bin/env python3
"""
Quick TFMOT compatibility test for BioDCASE-Tiny
"""

print("🧪 Quick TFMOT Test...")

try:
    # Set up environment BEFORE any imports
    import os
    os.environ['TF_USE_LEGACY_KERAS'] = '1'
    os.environ['KERAS_BACKEND'] = 'tensorflow'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    # Import TensorFlow first
    import tensorflow as tf
    print(f"✅ TensorFlow: {tf.__version__}")
    
    # Check if we're using legacy Keras
    try:
        import tf_keras as keras
        print(f"✅ Using tf-keras (legacy): {keras.__version__}")
        from tf_keras import Model, layers
    except ImportError:
        from keras import Model, layers
        import keras
        print(f"⚠️  Using standard Keras: {keras.__version__}")
    
    # Import TFMOT after setting up Keras properly
    import tensorflow_model_optimization as tfmot
    print(f"✅ TFMOT: {tfmot.__version__}")
    
    # Create simple functional model
    inputs = layers.Input(shape=(10,), name='input')
    x = layers.Dense(8, activation='relu', name='dense1')(inputs)
    outputs = layers.Dense(2, activation='softmax', name='output')(x)
    model = Model(inputs, outputs, name='test_model')
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    
    print("✅ Functional model created")
    
    # Test quantization
    q_model = tfmot.quantization.keras.quantize_model(model)
    print("✅ Quantization works")
    
    # Test pruning
    pruning_params = {
        'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
            initial_sparsity=0.30, final_sparsity=0.50, 
            begin_step=0, end_step=100
        )
    }
    p_model = tfmot.sparsity.keras.prune_low_magnitude(model, **pruning_params)
    print("✅ Pruning works")
    
    # Test inference
    import numpy as np
    test_input = np.random.random((1, 10))
    
    _ = model(test_input)
    _ = q_model(test_input)
    _ = p_model(test_input)
    print("✅ All models working")
    
    print("\n🎯 TFMOT is fully functional!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc() 