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
import tensorflow_model_optimization as tfmot
import numpy as np
import pandas as pd
from keras import Model, layers
from keras.src.metrics import AUC, Precision, Recall, F1Score
import keras
from pathlib import Path

from config import load_config, Config
from paths import FEATURES_PRQ_PATH, FEATURES_SHAPE_JSON_PATH, MODELS_DIR
from model import create_model, train_model
from model_training import make_tf_datasets, get_class_weight, set_seeds, predict_validation, get_median_flattened
from augmentations import add_background_noise_spec, add_gaussian_snr_spec, apply_with_p

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

    def train_step(self, data):
        x, y = data
        teacher_preds = self.teacher(x, training=False)
        
        with tf.GradientTape() as tape:
            student_preds = self.student(x, training=True)
            student_loss = self.student_loss_fn(y, student_preds)
            
            teacher_soft = tf.nn.softmax(teacher_preds / self.temperature)
            student_soft = tf.nn.softmax(student_preds / self.temperature) 
            distillation_loss = self.distillation_loss_fn(teacher_soft, student_soft)
            
            total_loss = self.alpha * distillation_loss + (1 - self.alpha) * student_loss

        gradients = tape.gradient(total_loss, self.student.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.student.trainable_variables))
        
        self.compiled_metrics.update_state(y, student_preds)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x, y = data
        student_preds = self.student(x, training=False)
        self.compiled_metrics.update_state(y, student_preds)
        return {m.name: m.result() for m in self.metrics}

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
                converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
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
    """Apply quantization-aware training to the model."""
    qat_model = tfmot.quantization.keras.quantize_model(model)
    return qat_model


def create_student_model(input_shape, compression_ratio=0.5):
    """Create smaller student version of mobilegru_slimmer."""
    from keras.src.applications.mobilenet import _conv_block, _depthwise_conv_block
    
    n_filters_1 = max(24, int(48 * compression_ratio))
    n_filters_2 = max(48, int(96 * compression_ratio))
    gru_units = max(16, int(32 * compression_ratio))
    dense_units = max(16, int(32 * compression_ratio))
    dropout = 0.05
    
    inputs = layers.Input(shape=input_shape)
    x = _conv_block(inputs, filters=n_filters_1, alpha=0.75, kernel=(10, 4), strides=(5, 2))
    x = _depthwise_conv_block(x, pointwise_conv_filters=n_filters_1, alpha=0.75, block_id=1)
    x = _depthwise_conv_block(x, pointwise_conv_filters=n_filters_2, alpha=0.75, block_id=2)
    x = _depthwise_conv_block(x, pointwise_conv_filters=n_filters_2, alpha=0.75, block_id=3)
    x = layers.Dropout(dropout, name="dropout1")(x)
    x = layers.MaxPooling2D((1,4))(x)
    x = layers.Reshape((x.shape[1],x.shape[2]*x.shape[3]))(x)
    x = layers.GRU(gru_units, return_sequences=True)(x)
    x = layers.Dropout(dropout, name="dropout2")(x)
    x = layers.GlobalAveragePooling1D(keepdims=True)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(dense_units, activation='relu')(x)
    x = layers.Dense(2)(x)
    outputs = layers.Softmax()(x)
    
    model = Model(inputs, outputs, name=f"student_cr{compression_ratio}")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss='binary_crossentropy',
        metrics=[
            AUC(curve='PR', name='average_precision'),
            Precision(name='precision', class_id=1),
            Recall(name='recall', class_id=1),
            F1Score(name='f1_score', average='micro')
        ]
    )
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
    """Apply quantization-aware training to the model."""
    
    def should_quantize(layer):
        # Quantize most layers but skip some that can cause issues
        return not isinstance(layer, (
            tf.keras.layers.BatchNormalization,
            tf.keras.layers.Dropout,
            tf.keras.layers.Activation,
            tf.keras.layers.Softmax,
            tf.keras.layers.InputLayer,
            tf.keras.layers.Reshape,
            tf.keras.layers.Flatten
        ))
    
    quantize_model = tfmot.quantization.keras.quantize_model
    qat_model = quantize_model(model)
    
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
    teacher = keras.models.load_model(teacher_weights_path)
    teacher_params = teacher.count_params()
    
    y_true, y_pred = predict_validation(teacher, valid_ds)
    teacher_ap = tf.keras.metrics.AUC(curve='PR')(y_true, y_pred).numpy()
    print(f"Teacher: {teacher_params:,} params, AP: {teacher_ap:.4f}")
    
    # Knowledge Distillation with FULL augmentation
    print("🎓 Knowledge Distillation (full augmentation)...")
    student = create_student_model(input_shape, compression_ratio=0.5)
    student_params = student.count_params()
    
    distiller = Distiller(student, teacher)
    distiller.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        metrics=[
            AUC(curve='PR', name='average_precision'),
            Precision(name='precision', class_id=1),
            Recall(name='recall', class_id=1),
        ],
        alpha=0.3,
        temperature=5.0
    )
    
    distiller.fit(
        train_ds_full.shuffle(config.model_training.shuffle_buff_n).prefetch(tf.data.AUTOTUNE),
        validation_data=valid_ds.cache().prefetch(tf.data.AUTOTUNE),
        epochs=15,
        class_weight=class_weight,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(
                filepath=MODELS_DIR / "student_distilled.keras",
                monitor="val_average_precision",
                save_best_only=True,
                mode="max"
            )
        ]
    )
    
    # Evaluate student
    student_distilled = keras.models.load_model(MODELS_DIR / "student_distilled.keras")
    y_true, y_pred = predict_validation(student_distilled, valid_ds)
    student_ap = tf.keras.metrics.AUC(curve='PR')(y_true, y_pred).numpy()
    print(f"Student: {student_params:,} params, AP: {student_ap:.4f}")
    
    # Pruning with MODERATE augmentation
    print("✂️ Pruning (moderate augmentation)...")
    train_ds_moderate = create_moderate_augment_ds()
    
    pruned_model = apply_pruning(student_distilled, target_sparsity=0.4)
    pruned_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss='binary_crossentropy',
        metrics=[AUC(curve='PR', name='average_precision')]
    )
    
    pruned_model.fit(
        train_ds_moderate.shuffle(config.model_training.shuffle_buff_n).prefetch(tf.data.AUTOTUNE),
        validation_data=valid_ds.cache().prefetch(tf.data.AUTOTUNE),
        epochs=8,
        class_weight=class_weight,
        callbacks=[tfmot.sparsity.keras.UpdatePruningStep()]
    )
    
    student_pruned = tfmot.sparsity.keras.strip_pruning(pruned_model)
    student_pruned.save(MODELS_DIR / "student_pruned.keras")
    
    # QAT with MINIMAL augmentation
    print("🔢 Quantization-Aware Training (minimal augmentation)...")
    train_ds_minimal = create_minimal_augment_ds()
    
    qat_model = apply_qat(student_pruned)
    qat_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002),
        loss='binary_crossentropy',
        metrics=[AUC(curve='PR', name='average_precision')]
    )
    
    qat_model.fit(
        train_ds_minimal.shuffle(config.model_training.shuffle_buff_n).prefetch(tf.data.AUTOTUNE),
        validation_data=valid_ds.cache().prefetch(tf.data.AUTOTUNE),
        epochs=6,
        class_weight=class_weight
    )
    
    qat_model.save(MODELS_DIR / "student_qat.keras")
    
    # INT8 TFLite
    print("📱 INT8 TFLite Conversion...")
    def representative_dataset():
        for batch in reference_ds.take(10):
            yield [batch[0].numpy().astype(np.float32)]
    
    converter = tf.lite.TFLiteConverter.from_keras_model(qat_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    
    tflite_model = converter.convert()
    tflite_path = MODELS_DIR / "student_int8.tflite"
    tflite_path.write_bytes(tflite_model)
    tflite_size_kb = len(tflite_model) / 1024
    
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


if __name__ == "__main__":
    config = load_config()
    teacher_path = MODELS_DIR / "best_mobilegru_slimmed.keras"
    
    if not teacher_path.exists():
        print(f"❌ Teacher model not found: {teacher_path}")
        exit(1)
    
    results = run_compression_pipeline(str(teacher_path), config) 