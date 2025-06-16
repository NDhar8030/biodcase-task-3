from keras import Model, layers
from keras.src.applications.mobilenet import _conv_block, _depthwise_conv_block
from keras.src.callbacks import History, EarlyStopping, TensorBoard, ModelCheckpoint
from keras.src.metrics import AUC, Precision, Recall, F1Score
import tensorflow as tf

from paths import TENSORBOARD_LOGS_PATH
from config import Config

TRAIN_CHECKPOINT_PATH = "data//04_models"

@tf.keras.utils.register_keras_serializable(package="Custom")
class GRUInt8Cell(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.state_size = units

    def build(self, input_shape):
        input_dim = input_shape[-1]

        # Kernel: [input_dim, 3 * units] -> z, r, h̃
        self.kernel = self.add_weight(
            shape=(input_dim, 3 * self.units),
            initializer="glorot_uniform",
            name="kernel")

        self.recurrent_kernel = self.add_weight(
            shape=(self.units, 3 * self.units),
            initializer="orthogonal",
            name="recurrent_kernel")

        self.bias = self.add_weight(
            shape=(3 * self.units,),
            initializer="zeros",
            name="bias")

        super().build(input_shape)

    @tf.function
    def call(self, inputs, states):
        h_tm1 = states[0]

        x_proj = tf.matmul(inputs, self.kernel)
        h_proj = tf.matmul(h_tm1, self.recurrent_kernel)

        x_z, x_r, x_h = tf.split(x_proj, 3, axis=1)
        h_z, h_r, h_h = tf.split(h_proj, 3, axis=1)
        b_z, b_r, b_h = tf.split(self.bias, 3, axis=0)

        z = tf.sigmoid(x_z + h_z + b_z)
        r = tf.sigmoid(x_r + h_r + b_r)
        h_candidate = tf.tanh(x_h + r * h_h + b_h)

        h = (1.0 - z) * h_candidate + z * h_tm1
        return h, [h]
    
@tf.keras.utils.register_keras_serializable(package="Custom")
class GRUInt8(tf.keras.layers.Layer):
    def __init__(self, units, return_sequences=False, go_backwards=False, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.return_sequences = return_sequences
        self.go_backwards = go_backwards
        self.rnn = tf.keras.layers.RNN(
            GRUInt8Cell(units),
            return_sequences=return_sequences,
            go_backwards=go_backwards
        )

    def call(self, inputs):
        return self.rnn(inputs)

    def get_config(self):
        config = super().get_config()
        config.update({
            "units": self.units,
            "return_sequences": self.return_sequences,
            "go_backwards": self.go_backwards
        })
        return config

def create_model(input_shape, n_filters_1=32, n_filters_2=64, dropout=0.02) -> Model:
    inputs = layers.Input(shape=input_shape)
    x = _conv_block(inputs, filters=n_filters_1, alpha=1, kernel=(10, 4), strides=(5, 2))
    x = _depthwise_conv_block(x, pointwise_conv_filters=n_filters_1, alpha=1, block_id=1)
    x = layers.GlobalMaxPooling2D(keepdims=True)(x)
    x = layers.Dropout(dropout, name="dropout1")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(2)(x)
    outputs = layers.Softmax()(x)
    model = Model(inputs, outputs, name="mobilenet_slimmed")
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=[AUC(curve='PR', name='average_precision'),
                 Precision(name='precision', class_id=1),
                 Recall(name='recall', class_id=1),
                 F1Score(name='f1_score', average='micro')]
    )
    return model

def create_model(input_shape,dropout=0.05) -> Model:
    inputs = layers.Input(shape=input_shape)
    x = inputs

    x = layers.SeparableConv2D(8, (2,2), activation='relu', dilation_rate=(4,4), padding='same')(x)
    x = layers.SeparableConv2D(16, (2,2), activation='relu', dilation_rate=(3,3), padding='same')(x)
    x = layers.SeparableConv2D(32, (2,2), activation='relu', dilation_rate=(2,2), padding='same')(x)

    x = layers.AveragePooling2D((2,6))(x)

    x = layers.Reshape((x.shape[1],x.shape[2]*x.shape[3]))(x)

    x = GRUInt8(32, return_sequences=True, go_backwards=False)(x)
    x = layers.Dropout(dropout,name="dropout1")(x)

    x = layers.GlobalAveragePooling1D()(x)

    x = layers.Dense(16, activation='relu')(x)

    outputs = layers.Dense(2, activation='softmax')(x)
    model = Model(inputs, outputs, name="convgru_2")
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=[AUC(curve='PR', name='average_precision'),
                 Precision(name='precision', class_id=1),
                 Recall(name='recall', class_id=1),
                 ]
    )
    return model

def train_model(model: Model, train_ds, valid_ds, config: Config, class_weight) -> Model:
    tr_cfg = config.model_training
    train_ds = train_ds.cache().shuffle(tr_cfg.shuffle_buff_n).prefetch(tf.data.AUTOTUNE)
    valid_ds = valid_ds.cache().prefetch(tf.data.AUTOTUNE)
    model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=tr_cfg.n_epochs,
        class_weight=class_weight,
        callbacks=[
            ModelCheckpoint(
                filepath=f"{TRAIN_CHECKPOINT_PATH}//best_{model.name}.keras",
                monitor="val_average_precision",
                save_best_only=True,
                save_weights_only=False,
                mode="max",
                verbose=1,
            ),
            EarlyStopping(
                patience=tr_cfg.early_stopping.patience,
                monitor="val_average_precision",
                mode="max",
                verbose=1,
            ),
            TensorBoard(TENSORBOARD_LOGS_PATH, update_freq=1)
        ]
    )
    return model