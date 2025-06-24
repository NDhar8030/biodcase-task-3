#!/usr/bin/env python3
"""
Simple test to verify knowledge_distill imports work correctly.
"""

import os
# Set environment variables before any imports
os.environ['TF_USE_LEGACY_KERAS'] = '1'
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("🧪 Testing knowledge_distill imports...")

try:
    # Test the imports that were causing issues
    from knowledge_distill import create_student_model, Distiller
    print("✅ Knowledge distill imports successful")
    
    # Test basic functionality
    import tensorflow as tf
    import numpy as np
    
    # Test student model creation
    input_shape = (83, 32, 1)
    student = create_student_model(input_shape)
    print(f"✅ Student model created: {student.count_params():,} parameters")
    
    # Test basic inference
    test_input = np.random.random((1, *input_shape))
    output = student(test_input)
    print(f"✅ Student inference works: {output.shape}")
    
    print("\n🎯 All imports and basic functionality working!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc() 