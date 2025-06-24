import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import Model
from keras.src.metrics import AUC
from keras.src.utils import to_categorical
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from matplotlib import pyplot as plt
import json
import os
from pathlib import Path
from config import Config, load_config
from paths import FEATURES_PRQ_PATH, KERAS_MODEL_PATH, FEATURES_SHAPE_JSON_PATH, MODELS_DIR, REFERENCE_DATASET_PATH, CM_FIG_DIR, PR_CURVE_DIR, PREDS_DIR
from model import create_model, train_model

from augmentations import add_background_noise_spec, add_gaussian_snr_spec, apply_with_p

def set_seeds(seed):
    tf.config.experimental.enable_op_determinism()
    keras.utils.set_random_seed(seed)

def get_median_flattened(tensor):
    """
    Calculates the median of a tensor after flattening it into a 1D tensor.

    Args:
        tensor: The input tensor (e.g., a 3D tensor of shape [83, 32, 1]).

    Returns:
        A scalar tensor representing the median of the flattened tensor.
    """
    # 1. Flatten the tensor into a 1D tensor
    flattened_tensor = tf.reshape(tensor, [-1])

    # 2. Sort the flattened tensor
    sorted_tensor = tf.sort(flattened_tensor)

    # 3. Get the number of elements
    num_elements = tf.shape(sorted_tensor)[0]

    # 4. Calculate middle index
    mid_idx = num_elements // 2

    # 5. Check if number of elements is odd or even
    # Using tf.cond for graph-compatible conditional logic
    # Note: tf.cond expects callables for true_fn and false_fn
    median = tf.cond(
        tf.equal(num_elements % 2, 1),
        true_fn=lambda: tf.cast(sorted_tensor[mid_idx], dtype=tf.float32),
        false_fn=lambda: tf.cast((sorted_tensor[mid_idx - 1] + sorted_tensor[mid_idx]) / 2.0, dtype=tf.float32)
    )
    return median

def make_tf_datasets(
    data: pd.DataFrame,
    features_shape,
    bg_spec_pool,
    buffer_size=10000,
    seed=42,
    batch_size=32,
    apply_augmentation: bool = True,
) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    splits = {}
    for split, group_data in data.groupby("split"):
        # shape (tf backend): batches, rows, cols, channels
        features = np.array(group_data["features"].to_list()).reshape((-1, *features_shape, 1))
        one_hot_labels = to_categorical(group_data["label"], num_classes=2)
        dataset = tf.data.Dataset.from_tensor_slices((features, one_hot_labels)).cache()

        # Dynamic augmentation (SpecAugment) applied **only** to the training split
        if split == "train" and apply_augmentation:
            def _augment(x, y):
                is_positive = y[1] == 1
                if is_positive:
                    if tf.reduce_max(x) - get_median_flattened(x) > 400:
                        x = apply_with_p(0.3, lambda w: add_gaussian_snr_spec(w, 4.0, 6.0), x)
                        x = apply_with_p(0.3, lambda w: add_background_noise_spec(w, bg_spec_pool, 4, 6), x)
                    else:
                        x = apply_with_p(0.3, lambda w: add_gaussian_snr_spec(w, 8.0, 16.0), x)
                        x = apply_with_p(0.3, lambda w: add_background_noise_spec(w, bg_spec_pool, 8, 16), x)
                else:
                    x = apply_with_p(0.0, lambda w: add_gaussian_snr_spec(w, 4.0, 24.0), x)
                    x = apply_with_p(0.0, lambda w: add_background_noise_spec(w, bg_spec_pool, 4, 16), x)
                return x, y

            dataset = dataset.shuffle(buffer_size=buffer_size, seed=seed)
            dataset = dataset.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)
            dataset = dataset.batch(batch_size)
        else:
            dataset = dataset.shuffle(buffer_size=buffer_size, seed=seed).batch(batch_size)
        splits[split] = dataset

    reference_dataset = splits["train"].shuffle(10000).take(100)
    return splits["train"], splits["validation"], reference_dataset


def get_class_weight(train_ds):
    train_labels = train_ds["label"]
    l_counts: dict[str, int] = dict(train_labels.value_counts())
    tot_counts = len(train_labels)
    class_weight = {k: tot_counts / v for k, v in l_counts.items()}
    return class_weight


def predict_validation(model: Model, val_dataset: tf.data.Dataset):
    val_ds = val_dataset.cache().prefetch(tf.data.AUTOTUNE)
    y_true = np.concatenate(list(val_ds.map(lambda x, y: y).as_numpy_iterator()))
    y_pred = model.predict(val_ds)
    return y_true, y_pred

def predict_train(model: Model, train_dataset: tf.data.Dataset):
    train_ds = train_dataset.cache().prefetch(tf.data.AUTOTUNE)
    y_true = np.concatenate(list(train_ds.map(lambda x, y: y).as_numpy_iterator()))
    y_pred = model.predict(train_ds)
    return y_true, y_pred


def get_confusion_matrix(y_true, y_pred, labels: list[str], threshold: float = 0.5):
    # Get probability of positive class
    y_pred_probs = y_pred[:, 1]
    # Apply threshold to get binary predictions
    y_pred_binary = (y_pred_probs >= threshold).astype(int)
    # Convert one-hot y_true to class indices
    y_true_binary = np.argmax(y_true, axis=1)
    
    return ConfusionMatrixDisplay.from_predictions(
        y_true_binary, y_pred_binary,
        display_labels=labels
    ).figure_

def get_pr_curve(y_true, y_pred):
    # Convert one-hot encoded arrays to 1D arrays
    y_true_classes = np.argmax(y_true, axis=1)
    y_pred_probs = y_pred[:, 1]  # Get probability of positive class (Yellowhammer)
    
    precision, recall, thresholds = precision_recall_curve(y_true_classes, y_pred_probs)
    average_precision = average_precision_score(y_true_classes, y_pred_probs)
    
    # Calculate F1 scores for each threshold
    f1_scores = 2 * (precision * recall) / (precision + recall)
    # Handle division by zero
    f1_scores = np.nan_to_num(f1_scores)
    
    # Find the threshold that gives the best F1 score
    best_f1_idx = np.argmax(f1_scores)
    best_f1 = f1_scores[best_f1_idx]
    best_threshold = thresholds[best_f1_idx] if best_f1_idx < len(thresholds) else thresholds[-1]
    best_precision = precision[best_f1_idx]
    best_recall = recall[best_f1_idx]
    
    fig = plt.figure()
    plt.plot(recall, precision, label=f"Average Precision = {average_precision:.4f}")
    
    # Plot the point with best F1 score
    plt.plot(best_recall, best_precision, 'ro', label=(
        f'Best F1: {best_f1:.4f}, Th: {best_threshold:.4f} \n'
        f'Precision: {best_precision:.4f}\n'
        f'Recall: {best_recall:.4f}'
    ))
    
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    return fig, best_threshold

def run_model_training(config: Config):
    set_seeds(config.model_training.seed)  # for reproducibility
    with open(FEATURES_SHAPE_JSON_PATH, "r") as f:
        features_shape = json.load(f)

    data = pd.read_parquet(FEATURES_PRQ_PATH)
    data["features"] = data["features"].apply(lambda x: x.reshape(features_shape))  # got flattened when writing parquet, restore shape now

    # ------------------------------------------------------------------
    # Build *spectrogram* background pool from negative training samples
    # ------------------------------------------------------------------
    neg_train = data[(data["split"] == "train") & (data["label"] == 0)]
    # Limit pool size to avoid excessive memory usage
    max_pool_size = 2048
    bg_pool_df = neg_train.sample(n=min(len(neg_train), max_pool_size), random_state=config.model_training.seed)

    bg_pool_specs = np.array(bg_pool_df["features"].to_list(), dtype=np.float32).reshape((-1, *features_shape, 1))
    bg_spec_pool = tf.constant(bg_pool_specs)

    class_weight = get_class_weight(data[data["split"] == "train"])
    train_ds, valid_ds, reference_ds = make_tf_datasets(
        data,
        features_shape,
        bg_spec_pool,
        config.model_training.shuffle_buff_n, 
        config.model_training.seed,
        config.model_training.batch_size,
        apply_augmentation=True,
    )
    model = create_model((*features_shape, 1))
    print(model.summary())
    model = train_model(model, train_ds, valid_ds, config, class_weight)
    loaded_model = keras.models.load_model(MODELS_DIR / f"best_{model.name}.keras")   
    y_true, y_pred = predict_validation(loaded_model, valid_ds)
    # Convert one-hot encoded arrays to 1D class arrays
    y_true_classes = np.argmax(y_true, axis=1)
    y_pred_probs = y_pred[:, 1]  # Get probability of positive class (Yellowhammer)
    preds_df = pd.DataFrame({'true_label': y_true_classes, 'prediction_probability': y_pred_probs})
    preds_df.to_csv(PREDS_DIR / f"best_{model.name}_preds.csv", index=False)
    pr_fig, best_threshold = get_pr_curve(y_true, y_pred)
    cm_fig = get_confusion_matrix(y_true, y_pred, labels=["Other", "Yellowhammer"], threshold=best_threshold)
    cm_fig.savefig(CM_FIG_DIR / f"best_{model.name}.png")
    pr_fig.savefig(PR_CURVE_DIR / f"best_{model.name}.png")

    y_true, y_pred = predict_train(loaded_model, train_ds)
    # Convert one-hot encoded arrays to 1D class arrays
    y_true_classes = np.argmax(y_true, axis=1)
    y_pred_probs = y_pred[:, 1]  # Get probability of positive class (Yellowhammer)
    preds_df = pd.DataFrame({'true_label': y_true_classes, 'prediction_probability': y_pred_probs})
    preds_df.to_csv(PREDS_DIR / f"best_{model.name}_preds_train.csv", index=False)
    pr_fig, best_threshold = get_pr_curve(y_true, y_pred)
    cm_fig = get_confusion_matrix(y_true, y_pred, labels=["Other", "Yellowhammer"], threshold=best_threshold)
    cm_fig.savefig(CM_FIG_DIR / f"train_best_{model.name}.png")
    pr_fig.savefig(PR_CURVE_DIR / f"train_best_{model.name}.png")
    
    model.save(KERAS_MODEL_PATH)
    reference_ds.save(str(REFERENCE_DATASET_PATH))

if __name__ == "__main__":
    config = load_config()
    run_model_training(config)