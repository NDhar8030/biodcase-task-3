"""
Knowledge Distillation, Pruning, and Quantization Pipeline for BioDCASE-Tiny

This module provides a complete pipeline for model compression using:
1. Knowledge distillation from a teacher to student model
2. Magnitude-based pruning 
3. Quantization-aware training (QAT)
4. INT8 TFLite conversion

Integrates seamlessly with existing BioDCASE training infrastructure.
"""

# Configure TFMOT environment before any TensorFlow imports
try:
    from setup_tfmot_env import setup_tfmot_environment
    setup_tfmot_environment()
except ImportError:
    # Fallback if setup script not available
    import os
    os.environ['TF_USE_LEGACY_KERAS'] = '1'
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import json
import tensorflow as tf
print(tf.__version__)
import tensorflow_model_optimization as tfmot
import numpy as np
import pandas as pd

try:
    import tf_keras as legacy_keras
    from tf_keras import Model, layers
    from tf_keras.metrics import AUC
    print("Using tf_keras (legacy Keras v2) for full compatibility")
except ImportError:
    # Fallback to tf.keras if tf_keras package not available
    from tensorflow.keras import Model, layers
    from tensorflow.keras.metrics import AUC
    legacy_keras = tf.keras
    print("tf_keras not found, falling back to tf.keras")

from pathlib import Path

from config import load_config, Config
from paths import FEATURES_PRQ_PATH, FEATURES_SHAPE_JSON_PATH, MODELS_DIR
from model import create_model, train_model
from model_training import make_tf_datasets, get_class_weight, set_seeds, predict_validation, get_median_flattened
from augmentations import add_background_noise_spec, add_gaussian_snr_spec, apply_with_p
from model import NoOpQuantizeConfig, quantize_annotate_layer

# Disable global XLA JIT to avoid optimizer variable issues
try:
    tf.config.optimizer.set_jit(False)
except Exception:
    pass

@tf.keras.saving.register_keras_serializable()
class Distiller(tf.keras.Model):
    """Knowledge distillation wrapper for training student from teacher."""
    
    def __init__(self, student: Model, teacher: Model, name: str = "distiller"):
        super().__init__(name=name)
        self.teacher = teacher
        self.teacher.trainable = False
        self.student = student

    def compile(
        self,
        optimizer,
        metrics=None,
        student_loss_fn=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
        distillation_loss_fn=tf.keras.losses.KLDivergence(),
        alpha: float = 0.1,
        temperature: float = 4.0,
    ):
        super().compile(optimizer=optimizer, metrics=metrics)
        self.student_loss_fn = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn
        self.alpha = alpha
        self.temperature = temperature

    @tf.function
    def train_step(self, data):
        # Unpack data based on whether sample_weight is provided
        if len(data) == 3:
            x, y, sample_weight = data
        else:
            x, y = data
            sample_weight = None

        teacher_preds = self.teacher(x, training=False)
        
        with tf.GradientTape() as tape:
            student_preds = self.student(x, training=True)
            
            # Calculate losses, passing sample_weight
            student_loss = self.student_loss_fn(
                y, student_preds, sample_weight=sample_weight
            )
            
            teacher_soft = tf.nn.softmax(teacher_preds / self.temperature)
            student_soft = tf.nn.softmax(student_preds / self.temperature) 
            distillation_loss = self.distillation_loss_fn(
                teacher_soft, student_soft, sample_weight=sample_weight
            )
            
            total_loss = self.alpha * distillation_loss + (1 - self.alpha) * student_loss

        gradients = tape.gradient(total_loss, self.student.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.student.trainable_variables))
        
        # Update metrics (they will also use sample_weight)
        self.compiled_metrics.update_state(y, student_preds, sample_weight=sample_weight)
        
        # Return a dict mapping metric names to current value.
        # Include the loss in the returned metrics.
        results = {m.name: m.result() for m in self.metrics}
        results.update({
            "loss": total_loss,
            "student_loss": student_loss,
            "distillation_loss": distillation_loss
        })
        return results

    @tf.function
    def test_step(self, data):
        # Unpack data
        if len(data) == 3:
            x, y, sample_weight = data
        else:
            x, y = data
            sample_weight = None
            
        student_preds = self.student(x, training=False)
        
        # Calculate loss
        student_loss = self.student_loss_fn(
            y, student_preds, sample_weight=sample_weight
        )
        
        # Update metrics
        self.compiled_metrics.update_state(y, student_preds, sample_weight=sample_weight)
        
        # Return a dict mapping metric names to current value.
        # Include the loss in the returned metrics.
        results = {m.name: m.result() for m in self.metrics}
        results["loss"] = student_loss
        return results

    def call(self, inputs, training=False):
        return self.student(inputs, training=training)

    def save_student(self, filepath, **kwargs):
        """Save only the student model."""
        self.student.save(filepath, **kwargs)
        
    def export_tflite(self, filepath: str, representative_data=None, quantize=True):
        """Export student to TFLite with optional INT8 quantization."""
        converter = tf.lite.TFLiteConverter.from_keras_model(self.student)
        
        if quantize:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            if representative_data:
                converter.representative_dataset = representative_data
                converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8, tf.lite.OpsSet.SELECT_TF_OPS]
                converter._experimental_lower_tensor_list_ops = False
                converter.inference_input_type = tf.int8
                converter.inference_output_type = tf.int8
        
        tflite_model = converter.convert()
        Path(filepath).write_bytes(tflite_model)
        return len(tflite_model)


def apply_pruning(model: Model, target_sparsity=0.5):
    """Apply magnitude-based pruning to the model."""
    pruning_params = {
        'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
            initial_sparsity=0.0,
            final_sparsity=target_sparsity,
            begin_step=0,
            end_step=2000
        )
    }
    
    model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(
        model,
        pruning_schedule=pruning_params['pruning_schedule']
    )
    
    return model_for_pruning


def apply_qat(model: Model):
    """Wrap BatchNormalization layers with a No-Op quantization config and then
    run `quantize_apply` so the rest of the network is instrumented for QAT.
    """

    def clone_fn(layer):
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            return quantize_annotate_layer(layer, NoOpQuantizeConfig())
        return layer  # keep as-is for automatic quantisation

    annotated_model = tf.keras.models.clone_model(model, clone_function=clone_fn)

    with tfmot.quantization.keras.quantize_scope({
        'NoOpQuantizeConfig': NoOpQuantizeConfig
    }):
        qat_model = tfmot.quantization.keras.quantize_apply(annotated_model)

    return qat_model

from model import _conv_block, _depthwise_conv_block

'''def create_student_model(input_shape, compression_ratio: float = 0.8, n_filters_1=24, n_filters_2=48, dropout=0.05):
    inputs = layers.Input(shape=input_shape)
    x = _conv_block(inputs, filters=n_filters_1, alpha=1, kernel=(10, 4), strides=(5, 2))
    x = _depthwise_conv_block(x, pointwise_conv_filters=n_filters_1, alpha=0.8, block_id=1)
    x = _depthwise_conv_block(x, pointwise_conv_filters=n_filters_2, alpha=0.8, block_id=2)
    x = _depthwise_conv_block(x, pointwise_conv_filters=n_filters_2, alpha=0.8, block_id=3)
    x = layers.Dropout(dropout, name="dropout1")(x)
    x = layers.MaxPooling2D((1,4))(x)
    x = layers.Reshape((x.shape[1],x.shape[2]*x.shape[3]))(x)
    x = layers.GRU(16, return_sequences=True)(x)
    x = layers.Dropout(dropout, name="dropout2")(x)
    x = layers.GlobalAveragePooling1D(keepdims=True)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(16, activation='relu')(x)
    x = layers.Dense(2)(x)
    outputs = layers.Softmax()(x)
    model = Model(inputs, outputs, name="student_serialisable")
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=[AUC(curve="PR", name="average_precision")],
    )
    return model'''

def create_student_model(input_shape, compression_ratio: float = 0.8):
    """Build a light-weight Conv-GRU student using ONLY serialisable layers.

    The architecture avoids MobileNet utility functions (which rely on Lambda
    layers containing TF ops) so that the resulting network can be cloned by
    `keras.models.clone_model` – a hard requirement for TFMOT pruning.
    """

    base_filters = 32  # starting channel count
    filters_1 = max(16, int(base_filters * compression_ratio))
    filters_2 = max(32, int(base_filters * 2 * compression_ratio))
    gru_units = max(24, int(32 * compression_ratio))
    dense_units = max(16, int(32 * compression_ratio))
    dropout = 0.1

    filters_1 = 16
    filters_2 = 32
    gru_units = 16
    dense_units = 16
    dropout = 0.05

    inputs = layers.Input(shape=input_shape)

    # --- Pure Keras layers only ------------------------------------------------
    x = layers.DepthwiseConv2D(kernel_size=(10, 4), strides=(2, 2), padding="same", depth_multiplier=1)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(max_value=6.0)(x)
    x = layers.Conv2D(filters_1, (1, 1), activation="relu")(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.ReLU(max_value=6.0)(x)
    
    x = layers.DepthwiseConv2D(kernel_size=(3, 3), padding="same", depth_multiplier=1)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(max_value=6.0)(x)
    x = layers.Conv2D(filters_2, (1, 1), activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(max_value=6.0)(x)
    
    x = layers.MaxPooling2D((1, 4))(x)  # reduce freq
    x = layers.Dropout(dropout)(x)

    # --- RNN Head --------------------------------------------------------------
    # Flatten frequency dimension into channels so we can run a GRU over time.
    time_steps = x.shape[1]
    feat_dim = x.shape[2] * x.shape[3]
    x = layers.Reshape((time_steps, feat_dim))(x)

    x = layers.GRU(gru_units, return_sequences=True)(x)
    x = layers.GlobalAveragePooling1D()(x)

    x = layers.Dense(dense_units, activation="relu")(x)
    x = layers.Dropout(dropout)(x)

    outputs = layers.Dense(2, activation="softmax")(x)

    model = Model(inputs, outputs, name="student_serialisable")
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=[AUC(curve="PR", name="average_precision")],
    )
    print(model.summary())
    return model


def apply_pruning(model: Model, target_sparsity=0.5):
    """Apply magnitude-based pruning to the model."""
    
    def should_prune(layer):
        # Don't prune certain layer types
        return not isinstance(layer, (
            tf.keras.layers.BatchNormalization,
            tf.keras.layers.Dropout,
            tf.keras.layers.Activation,
            tf.keras.layers.Softmax,
            tf.keras.layers.InputLayer,
            tf.keras.layers.Reshape,
            tf.keras.layers.Flatten,
            tf.keras.layers.GlobalMaxPooling2D,
            tf.keras.layers.GlobalAveragePooling1D,
            tf.keras.layers.MaxPooling2D
        ))
    
    pruning_params = {
        'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
            initial_sparsity=0.0,
            final_sparsity=target_sparsity,
            begin_step=0,
            end_step=2000  # Should cover most training
        )
    }
    
    model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(
        model,
        pruning_schedule=pruning_params['pruning_schedule'],
        layer_pruning_policy=should_prune
    )
    
    return model_for_pruning


def apply_qat(model: Model):
    """Wrap BatchNormalization layers with a No-Op quantization config and then
    run `quantize_apply` so the rest of the network is instrumented for QAT.
    """

    def clone_fn(layer):
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            return quantize_annotate_layer(layer, NoOpQuantizeConfig())
        return layer  # keep as-is for automatic quantisation

    annotated_model = tf.keras.models.clone_model(model, clone_function=clone_fn)

    with tfmot.quantization.keras.quantize_scope({
        'NoOpQuantizeConfig': NoOpQuantizeConfig
    }):
        qat_model = tfmot.quantization.keras.quantize_apply(annotated_model)

    return qat_model


def run_compression_pipeline(teacher_weights_path: str, config: Config):
    """Complete compression pipeline."""
    
    print("🔧 Starting Model Compression Pipeline")
    set_seeds(config.model_training.seed)
    
    with open(FEATURES_SHAPE_JSON_PATH, "r") as f:
        features_shape = json.load(f)
    
    data = pd.read_parquet(FEATURES_PRQ_PATH)
    data["features"] = data["features"].apply(lambda x: x.reshape(features_shape))
    
    # Setup datasets
    neg_train = data[(data["split"] == "train") & (data["label"] == 0)]
    bg_pool_df = neg_train.sample(n=min(len(neg_train), 2048), random_state=config.model_training.seed)
    bg_pool_specs = np.array(bg_pool_df["features"].to_list(), dtype=np.float32).reshape((-1, *features_shape, 1))
    bg_spec_pool = tf.constant(bg_pool_specs)
    
    class_weight = get_class_weight(data[data["split"] == "train"])
    
    # Create datasets with different augmentation intensities
    train_ds_full, valid_ds, reference_ds = make_tf_datasets(
        data, features_shape, bg_spec_pool,
        config.model_training.shuffle_buff_n, 
        config.model_training.seed,
        config.model_training.batch_size,
        apply_augmentation=True,  # Full augmentation for distillation
    )
    
    # Moderate augmentation for pruning (reduce probabilities)
    def create_moderate_augment_ds():
        splits = {}
        for split, group_data in data.groupby("split"):
            features = np.array(group_data["features"].to_list()).reshape((-1, *features_shape, 1))
            one_hot_labels = tf.keras.utils.to_categorical(group_data["label"], num_classes=2)
            dataset = tf.data.Dataset.from_tensor_slices((features, one_hot_labels)).cache()
            
            if split == "train":
                def _moderate_augment(x, y):
                    is_positive = y[1] == 1
                    if is_positive:
                        if tf.reduce_max(x) - get_median_flattened(x) > 400:
                            x = apply_with_p(0.15, lambda w: add_gaussian_snr_spec(w, 4.0, 6.0), x)
                            x = apply_with_p(0.15, lambda w: add_background_noise_spec(w, bg_spec_pool, 4, 6), x)
                        else:
                            x = apply_with_p(0.15, lambda w: add_gaussian_snr_spec(w, 8.0, 16.0), x)
                            x = apply_with_p(0.15, lambda w: add_background_noise_spec(w, bg_spec_pool, 8, 16), x)
                    else:
                        x = apply_with_p(0.0, lambda w: add_gaussian_snr_spec(w, 4.0, 24.0), x)
                        x = apply_with_p(0.0, lambda w: add_background_noise_spec(w, bg_spec_pool, 4, 16), x)
                    return x, y
                
                dataset = dataset.shuffle(buffer_size=config.model_training.shuffle_buff_n, seed=config.model_training.seed)
                dataset = dataset.map(_moderate_augment, num_parallel_calls=tf.data.AUTOTUNE)
                dataset = dataset.batch(config.model_training.batch_size)
            else:
                dataset = dataset.shuffle(buffer_size=config.model_training.shuffle_buff_n, seed=config.model_training.seed).batch(config.model_training.batch_size)
            splits[split] = dataset
        return splits["train"]
    
    # Minimal augmentation for QAT (very light)
    def create_minimal_augment_ds():
        splits = {}
        for split, group_data in data.groupby("split"):
            features = np.array(group_data["features"].to_list()).reshape((-1, *features_shape, 1))
            one_hot_labels = tf.keras.utils.to_categorical(group_data["label"], num_classes=2)
            dataset = tf.data.Dataset.from_tensor_slices((features, one_hot_labels)).cache()
            
            if split == "train":
                def _minimal_augment(x, y):
                    is_positive = y[1] == 1
                    if is_positive:
                        if tf.reduce_max(x) - get_median_flattened(x) > 400:
                            x = apply_with_p(0.05, lambda w: add_gaussian_snr_spec(w, 4.0, 6.0), x)
                            x = apply_with_p(0.05, lambda w: add_background_noise_spec(w, bg_spec_pool, 4, 6), x)
                        else:
                            x = apply_with_p(0.05, lambda w: add_gaussian_snr_spec(w, 8.0, 16.0), x)
                            x = apply_with_p(0.05, lambda w: add_background_noise_spec(w, bg_spec_pool, 8, 16), x)
                    else:
                        x = apply_with_p(0.0, lambda w: add_gaussian_snr_spec(w, 4.0, 24.0), x)
                        x = apply_with_p(0.0, lambda w: add_background_noise_spec(w, bg_spec_pool, 4, 16), x)
                    return x, y
                
                dataset = dataset.shuffle(buffer_size=config.model_training.shuffle_buff_n, seed=config.model_training.seed)
                dataset = dataset.map(_minimal_augment, num_parallel_calls=tf.data.AUTOTUNE)
                dataset = dataset.batch(config.model_training.batch_size)
            else:
                dataset = dataset.shuffle(buffer_size=config.model_training.shuffle_buff_n, seed=config.model_training.seed).batch(config.model_training.batch_size)
            splits[split] = dataset
        return splits["train"]
    
    input_shape = (*features_shape, 1)
    
    # Load teacher
    print("📚 Loading teacher model...")
    try:
        teacher = legacy_keras.models.load_model(teacher_weights_path)
    except Exception as e:
        print("Legacy Keras failed to load teacher, falling back to standalone Keras…")
        import importlib
        try:
            k3 = importlib.import_module('keras')
            teacher = k3.models.load_model(teacher_weights_path, compile=False)
            print("Teacher loaded with standalone Keras")
        except Exception as e2:
            print(f"❌ Could not load teacher model: {e2}")
            raise e
    teacher_params = teacher.count_params()
    
    y_true, y_pred = predict_validation(teacher, valid_ds)
    teacher_ap = tf.keras.metrics.AUC(curve='PR')(y_true, y_pred).numpy()
    print(f"Teacher: {teacher_params:,} params, AP: {teacher_ap:.4f}")
    
    # Knowledge Distillation with FULL augmentation
    print("🎓 Knowledge Distillation (full augmentation)...")
    student = create_student_model(input_shape, compression_ratio=0.5)
    student_params = student.count_params()
    
    # Learning rate schedule for knowledge distillation with heavy augmentation
    initial_lr = 0.001
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=initial_lr,
        decay_steps=25 * len(train_ds_full),  # match 25 training epochs
        alpha=0.1  # Final LR will be 10% of initial
    )
    
    distiller = Distiller(student, teacher)
    distiller.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        alpha=0.5,
        temperature=3.0,
        metrics=[AUC(curve='PR', name='average_precision')]
    )
    
    # Explicitly build the optimizer to prevent the '_unique_id' error
    print("Building optimizer state for distillation...")
    distiller.optimizer.build(student.trainable_variables)
    print("Optimizer state built.")
    
    distilled_weights_path = os.path.join("data", "04_models", "student_distilled.weights.h5")
    distiller.fit(
        train_ds_full.shuffle(config.model_training.shuffle_buff_n).prefetch(tf.data.AUTOTUNE),
        validation_data=valid_ds.cache().prefetch(tf.data.AUTOTUNE),
        epochs=25,
        class_weight=class_weight,
        callbacks=[
            StudentSaver(filepath=distilled_weights_path, monitor="val_average_precision", mode="max")
        ]
    )
    
    # Recreate student architecture and load best weights (no custom objects needed)
    student_distilled = create_student_model(input_shape, compression_ratio=0.5)
    student_distilled.load_weights(distilled_weights_path)
    y_true, y_pred = predict_validation(student_distilled, valid_ds)
    student_ap = tf.keras.metrics.AUC(curve='PR')(y_true, y_pred).numpy()
    print(f"Student: {student_params:,} params, AP: {student_ap:.4f}")
    
    # Pruning with MODERATE augmentation
    print("✂️ Pruning (moderate augmentation)...")
    train_ds_moderate = create_moderate_augment_ds()
    
    pruned_model = apply_pruning(student_distilled, target_sparsity=0.4)
    pruned_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002),
        loss='binary_crossentropy',
        metrics=[AUC(curve='PR', name='average_precision')]
    )
    
    # Explicitly build the optimizer
    print("Building optimizer state for pruning...")
    pruned_model.optimizer.build(pruned_model.trainable_variables)
    print("Optimizer state built.")
    
    pruned_model.fit(
        train_ds_moderate.shuffle(config.model_training.shuffle_buff_n).prefetch(tf.data.AUTOTUNE),
        validation_data=valid_ds.cache().prefetch(tf.data.AUTOTUNE),
        epochs=8,
        callbacks=[tfmot.sparsity.keras.UpdatePruningStep()]
    )
    
    student_pruned = tfmot.sparsity.keras.strip_pruning(pruned_model)
    legacy_keras.models.save_model(student_pruned, os.path.join("data", "04_models", "student_pruned.keras"))
    
    # QAT with MINIMAL augmentation
    print("🔄 Quantization-Aware Training (minimal augmentation)...")
    train_ds_minimal = create_minimal_augment_ds()
    
    qat_model = apply_qat(student_pruned)
    qat_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002),
        loss='binary_crossentropy',
        metrics=[AUC(curve='PR', name='average_precision')]
    )
    
    # Explicitly build the optimizer
    print("Building optimizer state for QAT...")
    qat_model.optimizer.build(qat_model.trainable_variables)
    print("Optimizer state built.")
    
    qat_model.fit(
        train_ds_minimal.shuffle(config.model_training.shuffle_buff_n).prefetch(tf.data.AUTOTUNE),
        validation_data=valid_ds.cache().prefetch(tf.data.AUTOTUNE),
        epochs=6,
        class_weight=class_weight
    )
    
    # Persist both the quantization-aware (for further fine-tuning) *and* a
    # stripped version that contains **no custom objects** so it can be loaded
    # anywhere with ordinary Keras only.

    qat_path = os.path.join("data", "04_models", "student_qat.keras")
    legacy_keras.models.save_model(qat_model, qat_path)

    # --- Produce a portable .h5 ------------------------------------------------
    portable_model = tfmot.quantization.keras.strip_quantization(qat_model)
    portable_h5_path = os.path.join("data", "04_models", "student_portable.h5")
    portable_model.save(portable_h5_path, include_optimizer=False)
    print(f"💾  Portable model saved → {portable_h5_path}  (no custom objects required)")
    
    # INT8 TFLite
    print("📱 INT8 TFLite Conversion...")
    def representative_dataset():
        for batch in reference_ds.take(10):
            yield [batch[0].numpy().astype(np.float32)]
    
    converter = tf.lite.TFLiteConverter.from_keras_model(qat_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    # Support pure INT8 where possible but fall back to Select TF ops to keep GRU
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]
    converter._experimental_lower_tensor_list_ops = False  # keep TensorList for RNNs
    converter.experimental_enable_resource_variables = True

    try:
        tflite_model = converter.convert()
        quant_type = "INT8"
    except Exception as e:
        print("⚠️  Strict INT8 conversion failed, retrying with hybrid mode…", e)
        # Relax: allow float fallback kernels
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                                              tf.lite.OpsSet.SELECT_TF_OPS]
        converter.inference_input_type = tf.float32
        converter.inference_output_type = tf.float32
        tflite_model = converter.convert()
        quant_type = "HYBRID (mixed)"

    tflite_path = os.path.join("data", "04_models", "student_int8.tflite")
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    tflite_size_kb = len(tflite_model) / 1024
    print(f"✅ TFLite {quant_type} model saved → {tflite_path} ({tflite_size_kb:.1f} KB)")
    
    print(f"📊 Results:")
    print(f"Parameters: {teacher_params:,} → {student_params:,} ({student_params/teacher_params:.1%})")
    print(f"AP: {teacher_ap:.4f} → {student_ap:.4f} ({student_ap/teacher_ap:.1%} retention)")
    print(f"TFLite size: {tflite_size_kb:.1f} KB")
    print(f"Augmentation strategy: Full → Moderate → Minimal")
    
    return {
        'teacher_ap': teacher_ap,
        'student_ap': student_ap,
        'teacher_params': teacher_params,
        'student_params': student_params,
        'tflite_size_kb': tflite_size_kb
    }

import os

# -----------------------------------------------------------------------------
# Callback: save only the student weights of the Distiller during training
# -----------------------------------------------------------------------------


class StudentSaver(tf.keras.callbacks.Callback):
    """Callback that saves *only the student weights* from the Distiller.

    By persisting weights instead of the full model we avoid all serialisation
    pitfalls (Lambda layers, untracked resources, custom objects). A fresh
    instance of the student architecture can be recreated later and the saved
    weights loaded back in.
    """

    def __init__(self, filepath: str, monitor: str = "val_average_precision", mode: str = "max"):
        super().__init__()
        self.filepath = filepath  # e.g. ".../student_distilled.weights.h5"
        self.monitor = monitor
        self.mode = mode
        self.best = -np.Inf if mode == "max" else np.Inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = logs.get(self.monitor)
        if current is None:
            return  # metric not available

        if (
            (self.mode == "max" and current > self.best) or
            (self.mode == "min" and current < self.best)
        ):
            self.best = current
            # Save *weights only*
            self.model.student.save_weights(self.filepath)
            print(f"📥  Saved new best student weights at epoch {epoch + 1}: {self.monitor}={current:.4f}")


if __name__ == "__main__":
    config = load_config()
    teacher_path = os.path.join("data", "04_models", "best_mobilegru_slimmed.keras")
    results = run_compression_pipeline(teacher_path, config) 