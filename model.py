from keras import Model, layers, regularizers
from keras.src.applications.mobilenet import _conv_block, _depthwise_conv_block
from keras.src.callbacks import History, EarlyStopping, TensorBoard, ModelCheckpoint, ReduceLROnPlateau
from keras.src.metrics import AUC, Precision, Recall, F1Score
import tensorflow as tf

from paths import TENSORBOARD_LOGS_PATH
from config import Config

TRAIN_CHECKPOINT_PATH = "data//04_models"

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

    x = layers.GRU(32, return_sequences=True)(x)
    x = layers.Dropout(dropout,name="dropout1")(x)

    x = layers.GlobalAveragePooling1D()(x)

    x = layers.Dense(16, activation='relu')(x)

    outputs = layers.Dense(2, activation='softmax')(x)
    model = Model(inputs, outputs, name="official_ConvGRU")
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=[AUC(curve='PR', name='average_precision'),
                 Precision(name='precision', class_id=1),
                 Recall(name='recall', class_id=1),
                 ]
    )
    return model

def create_model(input_shape,dropout=0.15) -> Model:
    inputs = layers.Input(shape=input_shape)
    x = inputs

    x = layers.SeparableConv2D(8, (2,2), activation='relu', dilation_rate=(4,4), padding='same', depthwise_regularizer=regularizers.L2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.SeparableConv2D(16, (2,2), activation='relu', dilation_rate=(3,3), padding='same', depthwise_regularizer=regularizers.L2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.SeparableConv2D(32, (2,2), activation='relu', dilation_rate=(2,2), padding='same', depthwise_regularizer=regularizers.L2(1e-4))(x)
    x = layers.BatchNormalization()(x)

    x = layers.AveragePooling2D((2,6))(x)

    x = layers.Reshape((x.shape[1],x.shape[2]*x.shape[3]))(x)

    x = layers.GRU(32, return_sequences=True,
                   kernel_regularizer=regularizers.L2(1e-4))(x)
    x = layers.Dropout(dropout)(x)

    x = layers.GlobalAveragePooling1D()(x)

    x = layers.Dense(16, activation='relu',
                     kernel_regularizer=regularizers.L2(1e-4))(x)

    outputs = layers.Dense(2, activation='softmax')(x)
    model = Model(inputs, outputs, name="official_regularized_ConvGRU")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0008),
        loss='binary_crossentropy',
        metrics=[AUC(curve='PR', name='average_precision'),
                 Precision(name='precision', class_id=1),
                 Recall(name='recall', class_id=1),
                 ]
    )
    return model

def create_model(input_shape,dropout=0.15) -> Model:
    inputs = layers.Input(shape=input_shape)
    x = inputs

    x = layers.Reshape((x.shape[1],x.shape[2]*x.shape[3]))(x)

    x = layers.GRU(32, return_sequences=True,
                   kernel_regularizer=regularizers.L2(1e-4))(x)
    x = layers.Dropout(dropout)(x)

    x = layers.GlobalAveragePooling1D()(x)

    x = layers.Dense(16, activation='relu',
                     kernel_regularizer=regularizers.L2(1e-4))(x)

    outputs = layers.Dense(2, activation='softmax')(x)
    model = Model(inputs, outputs, name="official_regularized_GRU")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0008),
        loss='binary_crossentropy',
        metrics=[AUC(curve='PR', name='average_precision'),
                 Precision(name='precision', class_id=1),
                 Recall(name='recall', class_id=1),
                 ]
    )
    return model

def create_model(input_shape,dropout=0.10) -> Model:
    inputs = layers.Input(shape=input_shape)
    x = inputs

    x = layers.Reshape((x.shape[1],x.shape[2]*x.shape[3]))(x)

    x = layers.GRU(32, return_sequences=True)(x)
    x = layers.Dropout(dropout)(x)

    x = layers.GlobalAveragePooling1D()(x)

    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dropout(dropout)(x)

    outputs = layers.Dense(2, activation='softmax')(x)
    model = Model(inputs, outputs, name="official_GRU_dynamic_augment")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss='binary_crossentropy',
        metrics=[AUC(curve='PR', name='average_precision'),
                 Precision(name='precision', class_id=1),
                 Recall(name='recall', class_id=1),
                 ]
    )
    return model

def create_model(input_shape,dropout=0.15) -> Model:
    inputs = layers.Input(shape=input_shape)
    x = inputs

    x = layers.SeparableConv2D(8, (2,2), activation='relu', dilation_rate=(4,4), padding='same', depthwise_regularizer=regularizers.L2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.SeparableConv2D(16, (2,2), activation='relu', dilation_rate=(3,3), padding='same', depthwise_regularizer=regularizers.L2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.SeparableConv2D(32, (2,2), activation='relu', dilation_rate=(2,2), padding='same', depthwise_regularizer=regularizers.L2(1e-4))(x)
    x = layers.BatchNormalization()(x)

    x = layers.AveragePooling2D((2,6))(x)

    x = layers.Reshape((x.shape[1],x.shape[2]*x.shape[3]))(x)

    x = layers.GRU(32, return_sequences=True,
                   kernel_regularizer=regularizers.L2(1e-4))(x)
    x = layers.Dropout(dropout)(x)

    x = layers.GlobalAveragePooling1D()(x)

    x = layers.Dense(16, activation='relu',
                     kernel_regularizer=regularizers.L2(1e-4))(x)

    outputs = layers.Dense(2, activation='softmax')(x)
    model = Model(inputs, outputs, name="official_regularized_ConvGRU_dynamic_augment")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0008),
        loss='binary_crossentropy',
        metrics=[AUC(curve='PR', name='average_precision'),
                 Precision(name='precision', class_id=1),
                 Recall(name='recall', class_id=1),
                 ]
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

    x = layers.GRU(32, return_sequences=True)(x)
    x = layers.Dropout(dropout,name="dropout1")(x)

    x = layers.GlobalAveragePooling1D()(x)

    x = layers.Dense(16, activation='relu')(x)

    outputs = layers.Dense(2, activation='softmax')(x)
    model = Model(inputs, outputs, name="official_ConvGRU_dynamic_augment")
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=[AUC(curve='PR', name='average_precision'),
                 Precision(name='precision', class_id=1),
                 Recall(name='recall', class_id=1),
                 ]
    )
    return model

def create_model(input_shape,dropout=0.10) -> Model:
    inputs = layers.Input(shape=input_shape)
    x = inputs

    x = layers.Reshape((x.shape[1],x.shape[2]*x.shape[3]))(x)

    x = layers.GRU(32, return_sequences=True, kernel_regularizer=regularizers.L2(1e-8))(x)
    x = layers.Dropout(dropout)(x)

    x = layers.GlobalAveragePooling1D()(x)

    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dropout(dropout)(x)

    outputs = layers.Dense(2, activation='softmax')(x)
    model = Model(inputs, outputs, name="official_GRU_test")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss='binary_crossentropy',
        metrics=[AUC(curve='PR', name='average_precision'),
                 Precision(name='precision', class_id=1),
                 Recall(name='recall', class_id=1),
                 ]
    )
    return model

def create_model(input_shape,dropout=0.10) -> Model:
    inputs = layers.Input(shape=input_shape)
    x = inputs

    x = layers.Reshape((x.shape[1],x.shape[2]*x.shape[3]))(x)

    x = layers.GRU(32, return_sequences=True)(x)
    x = layers.Dropout(dropout)(x)

    x = layers.GlobalAveragePooling1D()(x)

    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dropout(dropout)(x)

    outputs = layers.Dense(2, activation='softmax')(x)
    model = Model(inputs, outputs, name="official_focal_GRU_dynamic_augment")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.CategoricalFocalCrossentropy(alpha=0.25, gamma=3.0),
        metrics=[AUC(curve='PR', name='average_precision'),
                 Precision(name='precision', class_id=1),
                 Recall(name='recall', class_id=1),
                 ]
    )
    return model


def create_model(input_shape,dropout=0.05) -> Model:
    inputs = layers.Input(shape=input_shape)
    x = inputs

    x = layers.SeparableConv2D(32, (2,2), activation='relu', dilation_rate=(4,4), padding='same')(x)
    x = layers.SeparableConv2D(32, (2,2), activation='relu', dilation_rate=(3,3), padding='same')(x)
    #x = layers.SeparableConv2D(32, (2,2), activation='relu', dilation_rate=(2,2), padding='same')(x)

    x = layers.AveragePooling2D((2,6))(x)

    x = layers.Reshape((x.shape[1],x.shape[2]*x.shape[3]))(x)

    x = layers.GRU(32, return_sequences=True)(x)
    x = layers.Dropout(dropout,name="dropout1")(x)

    x = layers.GlobalAveragePooling1D()(x)

    x = layers.Dense(16, activation='relu')(x)

    outputs = layers.Dense(2, activation='softmax')(x)
    model = Model(inputs, outputs, name="official_ConvGRU_test")
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=[AUC(curve='PR', name='average_precision'),
                 Precision(name='precision', class_id=1),
                 Recall(name='recall', class_id=1),
                 ]
    )
    return model

def create_model(input_shape,dropout=0.05) -> Model:
    inputs = layers.Input(shape=input_shape)
    x = inputs

    x = layers.SeparableConv2D(16, (3,3), activation='relu', padding='same')(x)
    x = layers.SeparableConv2D(32, (5,3), activation='relu', padding='same')(x)
    x = layers.SeparableConv2D(32, (3,3), activation='relu', dilation_rate=(3,1), padding='same')(x)

    x = layers.SeparableConv2D(32, (1,32), activation='relu', padding='valid')(x)

    x = layers.AveragePooling2D((2,1))(x)

    x = layers.Reshape((x.shape[1],x.shape[2]*x.shape[3]))(x)

    x = layers.GRU(32, return_sequences=True)(x)
    x = layers.Dropout(dropout,name="dropout1")(x)

    x = layers.GlobalAveragePooling1D()(x)

    x = layers.Dense(16, activation='relu')(x)

    outputs = layers.Dense(2, activation='softmax')(x)
    model = Model(inputs, outputs, name="weird_ConvGRU_test")
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=[AUC(curve='PR', name='average_precision'),
                 Precision(name='precision', class_id=1),
                 Recall(name='recall', class_id=1),
                 ]
    )
    return model

def create_model(input_shape,dropout=0.05) -> Model:
    inputs = layers.Input(shape=input_shape)
    x = inputs

    x = layers.SeparableConv2D(16, (3,3), activation='relu', padding='same')(x)
    x = layers.SeparableConv2D(32, (5,3), activation='relu', padding='same')(x)
    x = layers.SeparableConv2D(32, (3,3), activation='relu', dilation_rate=(3,1), padding='same')(x)

    x = layers.SeparableConv2D(32, (1,32), activation='relu', padding='valid')(x)
    x = layers.Dropout(dropout)(x)

    x = layers.AveragePooling2D((2,1))(x)

    x = layers.Reshape((x.shape[1],x.shape[2]*x.shape[3]))(x)

    x = layers.GRU(32, return_sequences=True)(x)
    x = layers.Dropout(dropout,name="dropout1")(x)

    x = layers.GlobalAveragePooling1D()(x)

    x = layers.Dense(16, activation='relu')(x)

    outputs = layers.Dense(2, activation='softmax')(x)
    model = Model(inputs, outputs, name="multi_ConvGRU_test")
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=[AUC(curve='PR', name='average_precision'),
                 Precision(name='precision', class_id=1),
                 Recall(name='recall', class_id=1),
                 ]
    )
    return model

def create_model(input_shape,dropout=0.05) -> Model:
    inputs = layers.Input(shape=input_shape)
    x = inputs

    x = layers.SeparableConv2D(32, (2,2), activation='relu', dilation_rate=(4,4), padding='same')(x)
    x = layers.SeparableConv2D(32, (2,2), activation='relu', dilation_rate=(3,3), padding='same')(x)
    #x = layers.SeparableConv2D(32, (2,2), activation='relu', dilation_rate=(2,2), padding='same')(x)

    x = layers.AveragePooling2D((2,6))(x)
 
    x = layers.Reshape((x.shape[1],x.shape[2]*x.shape[3]))(x)

    x = layers.GRU(32, return_sequences=True)(x)
    x = layers.Dropout(dropout,name="dropout1")(x)

    x = layers.GlobalAveragePooling1D()(x)

    x = layers.Dense(16, activation='relu')(x)

    outputs = layers.Dense(2, activation='softmax')(x)
    model = Model(inputs, outputs, name="official_ConvGRU_test")
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=[AUC(curve='PR', name='average_precision'),
                 Precision(name='precision', class_id=1),
                 Recall(name='recall', class_id=1),
                 ]
    )
    return model

def create_model(input_shape,dropout=0.05) -> Model:
    inputs = layers.Input(shape=input_shape)
    # === Conv stack (replace old one) ===
    y = layers.Conv2D(16, (3,3), padding='same', activation='relu')(inputs)
    y = layers.BatchNormalization()(y)

    y = layers.Conv2D(24, (3,3), dilation_rate=(3,1),
                      padding='same', activation='relu')(y)
    y = layers.BatchNormalization()(y)

    y = layers.Conv2D(32, (1, input_shape[1]), padding='valid',
                      activation='relu')(y)   # collapse F
    y = layers.BatchNormalization()(y)

    y = layers.MaxPooling2D((2,1))(y)    # halve time ⇒ 41 steps

    # === LOCAL head ===
    loc = layers.Reshape((y.shape[1], y.shape[3]))(y)  # (T, C)
    loc = layers.GRU(32, return_sequences=True)(loc)
    loc = layers.GlobalAveragePooling1D()(loc)

    # === GLOBAL head ===
    glob = layers.GlobalAveragePooling2D()(y)
    glob = layers.Dense(32, activation='relu')(glob)

    z = layers.concatenate([loc, glob])
    z = layers.Dense(32, activation='relu')(z)
    z = layers.Dropout(dropout)(z)

    outputs = layers.Dense(2, activation='softmax')(z)
    model = Model(inputs, outputs, name="dualhead_ConvGRU_test")
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=[AUC(curve='PR', name='average_precision'),
                 Precision(name='precision', class_id=1),
                 Recall(name='recall', class_id=1),
                 ]
    )
    return model

def create_model(input_shape,dropout=0.10) -> Model:
    inputs = layers.Input(shape=input_shape)
    x = inputs

    x = layers.Reshape((x.shape[1],x.shape[2]*x.shape[3]))(x)

    x = layers.GRU(32, return_sequences=True, kernel_regularizer=regularizers.L2(1e-8))(x)
    x = layers.Dropout(dropout)(x)

    x = layers.GlobalAveragePooling1D()(x)

    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dropout(dropout)(x)

    outputs = layers.Dense(2, activation='softmax')(x)
    model = Model(inputs, outputs, name="official_daugment_GRU_test")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss='binary_crossentropy',
        metrics=[AUC(curve='PR', name='average_precision'),
                 Precision(name='precision', class_id=1),
                 Recall(name='recall', class_id=1),
                 ]
    )
    return model

def create_model(input_shape,dropout=0.05) -> Model:
    inputs = layers.Input(shape=input_shape)
    x = inputs

    x = layers.SeparableConv2D(32, (2,2), activation='relu', dilation_rate=(4,4), padding='same')(x)
    x = layers.SeparableConv2D(32, (2,2), activation='relu', dilation_rate=(3,3), padding='same')(x)
    x = layers.SeparableConv2D(32, (2,2), activation='relu', dilation_rate=(2,2), padding='same')(x)

    x = layers.AveragePooling2D((2,6))(x)

    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(32, activation='relu')(x)

    outputs = layers.Dense(2, activation='softmax')(x)
    model = Model(inputs, outputs, name="official_Conv2D_test")
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=[AUC(curve='PR', name='average_precision'),
                 Precision(name='precision', class_id=1),
                 Recall(name='recall', class_id=1),
                 ]
    )
    return model

def create_model(input_shape,dropout=0.05) -> Model:
    inputs = layers.Input(shape=input_shape)
    x = inputs

    x = layers.SeparableConv2D(32, (2,2), activation='relu', dilation_rate=(4,4), padding='same')(x)
    x = layers.SeparableConv2D(32, (2,2), activation='relu', dilation_rate=(3,3), padding='same')(x)
    #x = layers.SeparableConv2D(32, (2,2), activation='relu', dilation_rate=(2,2), padding='same')(x)

    x = layers.AveragePooling2D((2,6))(x)
 
    x = layers.Reshape((x.shape[1],x.shape[2]*x.shape[3]))(x)

    x = layers.GRU(32, return_sequences=True)(x)
    x = layers.Dropout(dropout,name="dropout1")(x)

    x = layers.GlobalAveragePooling1D()(x)

    x = layers.Dense(32, activation='relu')(x)

    outputs = layers.Dense(2, activation='softmax')(x)
    model = Model(inputs, outputs, name="official_daugment_ConvGRU_test")
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=[AUC(curve='PR', name='average_precision'),
                 Precision(name='precision', class_id=1),
                 Recall(name='recall', class_id=1),
                 ]
    )
    return model

def create_model(input_shape, n_filters_1=32, n_filters_2=64, dropout=0.02) -> Model:
    inputs = layers.Input(shape=input_shape)
    x = _conv_block(inputs, filters=n_filters_1, alpha=1, kernel=(10, 4), strides=(5, 2))
    x = _depthwise_conv_block(x, pointwise_conv_filters=n_filters_1, alpha=1, block_id=1)
    x = layers.GlobalMaxPooling2D(keepdims=True)(x)
    x = layers.Dropout(dropout, name="dropout1")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(2)(x)
    outputs = layers.Softmax()(x)
    model = Model(inputs, outputs, name="proposed_baseline")
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=[AUC(curve='PR', name='average_precision'),
                 Precision(name='precision', class_id=1),
                 Recall(name='recall', class_id=1),
                 F1Score(name='f1_score', average='micro')]
    )
    return model

def create_model(input_shape, n_filters_1=32, n_filters_2=64, dropout=0.02) -> Model:
    inputs = layers.Input(shape=input_shape)
    x = _conv_block(inputs, filters=n_filters_1, alpha=1, kernel=(10, 4), strides=(5, 2))
    x = _depthwise_conv_block(x, pointwise_conv_filters=n_filters_1, alpha=1, block_id=1)
    x = layers.GlobalMaxPooling2D(keepdims=True)(x)
    x = layers.Dropout(dropout, name="dropout1")(x)
    x = layers.Reshape((x.shape[1],x.shape[2]*x.shape[3]))(x)
    x = layers.GRU(32, go_backwards=True, return_sequences=True)(x)
    x = layers.Dropout(dropout, name="dropout2")(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(16, activation='relu')(x)
    x = layers.Dense(2)(x)
    outputs = layers.Softmax()(x)
    model = Model(inputs, outputs, name="gru_plus_proposed_baseline")
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=[AUC(curve='PR', name='average_precision'),
                 Precision(name='precision', class_id=1),
                 Recall(name='recall', class_id=1),
                 F1Score(name='f1_score', average='micro')]
    )
    return model

def create_model(input_shape, n_filters_1=32, n_filters_2=64, dropout=0.02) -> Model:
    inputs = layers.Input(shape=input_shape)
    x = _conv_block(inputs, filters=n_filters_1, alpha=1, kernel=(10, 4), strides=(5, 2))
    x = _depthwise_conv_block(x, pointwise_conv_filters=n_filters_1, alpha=1, block_id=1)
    x = _depthwise_conv_block(x, pointwise_conv_filters=n_filters_2, alpha=1, block_id=2)
    x = _depthwise_conv_block(x, pointwise_conv_filters=n_filters_2, alpha=1, block_id=3)
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

def create_model(input_shape, n_filters_1=24, n_filters_2=48, dropout=0.15) -> Model:
    inputs = layers.Input(shape=input_shape)
    x = _conv_block(inputs, filters=n_filters_1, alpha=1, kernel=(10, 4), strides=(5, 2))
    x = _depthwise_conv_block(x, pointwise_conv_filters=n_filters_1, alpha=1, block_id=1)
    x = _depthwise_conv_block(x, pointwise_conv_filters=n_filters_2, alpha=1, block_id=2)
    x = layers.MaxPooling2D((1,4))(x)
    x = _depthwise_conv_block(x, pointwise_conv_filters=n_filters_2, alpha=1, block_id=3)
    x = layers.GlobalMaxPooling2D(keepdims=True)(x)
    x = layers.Dropout(dropout, name="dropout1")(x)
    x = layers.Reshape((x.shape[1],x.shape[2]*x.shape[3]))(x)
    x = layers.GRU(32, return_sequences=True, kernel_regularizer=regularizers.L2(1e-8))(x)
    x = layers.Dropout(dropout, name="dropout2")(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(16, activation='relu')(x)
    x = layers.Dense(2)(x)
    outputs = layers.Softmax()(x)
    model = Model(inputs, outputs, name="mobilegru_slimmed")
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=[AUC(curve='PR', name='average_precision'),
                 Precision(name='precision', class_id=1),
                 Recall(name='recall', class_id=1),
                 F1Score(name='f1_score', average='micro')]
    )
    return model

def create_model(input_shape) -> Model:
    inputs = layers.Input(shape=input_shape)
    flat_dim = input_shape[0] * input_shape[1] * input_shape[2]
    x = layers.Reshape((flat_dim,))(inputs)
    x = layers.Dense(16, activation='relu')(x)
    outputs = layers.Dense(2, activation='softmax')(x)  
    model = Model(inputs, outputs, name="dense_baseline")
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=[AUC(curve='PR', name='average_precision'),
                 Precision(name='precision', class_id=1),
                 Recall(name='recall', class_id=1),
                 F1Score(name='f1_score', average='micro')]
    )
    return model

def create_model(input_shape, n_filters_1=32, n_filters_2=64, dropout=0.05) -> Model:
    inputs = layers.Input(shape=input_shape)
    x = _conv_block(inputs, filters=n_filters_1, alpha=1, kernel=(10, 4), strides=(5, 2))
    x = _depthwise_conv_block(x, pointwise_conv_filters=n_filters_1, alpha=1, block_id=1)
    x = _depthwise_conv_block(x, pointwise_conv_filters=n_filters_2, alpha=1, block_id=2)
    x = _depthwise_conv_block(x, pointwise_conv_filters=n_filters_2, alpha=1, block_id=3)
    x = layers.GlobalMaxPooling2D(keepdims=True)(x)
    x = layers.Dropout(dropout, name="dropout1")(x)
    x = layers.Reshape((x.shape[1],x.shape[2]*x.shape[3]))(x)
    x = layers.GRU(32, return_sequences=True)(x)
    x = layers.Dropout(dropout, name="dropout2")(x)
    x = layers.GlobalAveragePooling1D(keepdims=True)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(2)(x)
    outputs = layers.Softmax()(x)
    model = Model(inputs, outputs, name="mobilegru_regularized_slimmed")
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=[AUC(curve='PR', name='average_precision'),
                 Precision(name='precision', class_id=1),
                 Recall(name='recall', class_id=1),
                 F1Score(name='f1_score', average='micro')]
    )
    return model

def create_model(input_shape, n_filters_1=32, n_filters_2=64, dropout=0.02) -> Model:
    inputs = layers.Input(shape=input_shape)
    x = _conv_block(inputs, filters=n_filters_1, alpha=1, kernel=(10, 4), strides=(5, 2))
    x = _depthwise_conv_block(x, pointwise_conv_filters=n_filters_1, alpha=1, block_id=1)
    x = layers.GlobalMaxPooling2D(keepdims=True)(x)
    x = layers.Dropout(dropout, name="dropout1")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(2)(x)
    outputs = layers.Softmax()(x)
    model = Model(inputs, outputs, name="proposed_oldpreproc_noaugment_baseline")
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=[AUC(curve='PR', name='average_precision'),
                 Precision(name='precision', class_id=1),
                 Recall(name='recall', class_id=1),
                 F1Score(name='f1_score', average='micro')]
    )
    return model

def create_model(input_shape, n_filters_1=32, n_filters_2=64, dropout=0.15) -> Model:
    inputs = layers.Input(shape=input_shape)
    x = _conv_block(inputs, filters=n_filters_1, alpha=1, kernel=(10, 4), strides=(5, 2))
    x = _depthwise_conv_block(x, pointwise_conv_filters=n_filters_1, alpha=1.0, block_id=1)
    x = _depthwise_conv_block(x, pointwise_conv_filters=n_filters_2, alpha=1.0, block_id=2)
    x = _depthwise_conv_block(x, pointwise_conv_filters=n_filters_2, alpha=1.0, block_id=3)
    x = layers.Dropout(dropout, name="dropout1")(x)
    x = layers.MaxPooling2D((1,4))(x)
    x = layers.Reshape((x.shape[1],x.shape[2]*x.shape[3]))(x)
    x = layers.GRU(32, return_sequences=True)(x)
    x = layers.Dropout(dropout, name="dropout2")(x)
    x = layers.GlobalAveragePooling1D(keepdims=True)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dense(2)(x)
    outputs = layers.Softmax()(x)
    model = Model(inputs, outputs, name="mobilegru_slimmed")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss='binary_crossentropy',
        metrics=[AUC(curve='PR', name='average_precision'),
                 Precision(name='precision', class_id=1),
                 Recall(name='recall', class_id=1),
                 F1Score(name='f1_score', average='micro')]
    )
    return model

def create_model(input_shape, n_filters_1=24, n_filters_2=48, dropout=0.05) -> Model:
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
    model = Model(inputs, outputs, name="mobilegru_slimmer")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss='binary_crossentropy',
        metrics=[AUC(curve='PR', name='average_precision'),
                 Precision(name='precision', class_id=1),
                 Recall(name='recall', class_id=1),
                 F1Score(name='f1_score', average='micro')]
    )
    return model

def train_model(model: Model, train_ds, valid_ds, config: Config, class_weight) -> Model:
    tr_cfg = config.model_training
    train_ds = train_ds.shuffle(tr_cfg.shuffle_buff_n, reshuffle_each_iteration=True).prefetch(tf.data.AUTOTUNE)
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
            #ReduceLROnPlateau(
            #    monitor="val_loss",
            #    factor=0.1,
            #    patience=3,
            #    verbose=1,
            #),
            #EarlyStopping(
            #    patience=tr_cfg.early_stopping.patience,
            #    monitor="val_loss",
            #    mode="min",
            #    verbose=1,
            #),
            TensorBoard(TENSORBOARD_LOGS_PATH, update_freq=1)
        ]
    )
    return model

# === Knowledge-Distillation helper =============================================================
class Distiller(tf.keras.Model):
    """Keras Model wrapper that distils a *student* from a frozen *teacher*.

    The forward pass always returns the student predictions so `model.fit()`,
    callbacks, and metric tracking continue to work as before, while an
    internal loss combines standard supervision with a soft-label KLDivergence
    term taken from the teacher logits.
    """
    def __init__(self, student: Model, teacher: Model, name: str = "distiller"):
        super().__init__(name=name)
        self.teacher = teacher
        self.teacher.trainable = False  # freeze teacher weights
        self.student = student

    def compile(
        self,
        optimizer,
        metrics,
        student_loss_fn=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
        distillation_loss_fn=tf.keras.losses.KLDivergence(),
        alpha: float = 0.1,
        temperature: float = 4.0,
        **kwargs,
    ):
        """Configure the distiller.

        alpha         – weight given to the distillation (soft) loss
        temperature   – higher → softer probability distribution
        """
        super().compile(optimizer=optimizer, metrics=metrics, **kwargs)
        self.student_loss_fn = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn
        self.alpha = alpha
        self.temperature = temperature

    # --------------------------------------------------------
    # Overridden train / test steps (Keras custom-training loop)
    # --------------------------------------------------------
    def train_step(self, data):
        x, y = data

        # Forward pass of teacher (no gradient tracking)
        teacher_preds = self.teacher(x, training=False)

        with tf.GradientTape() as tape:
            # Student forward pass
            student_preds = self.student(x, training=True)

            # Hard-label loss (ground truth)
            student_loss = self.student_loss_fn(y, student_preds)

            # Soft-label (distillation) loss
            softened_teacher = tf.nn.softmax(teacher_preds / self.temperature, axis=1)
            softened_student = tf.nn.softmax(student_preds / self.temperature, axis=1)
            distill_loss = self.distillation_loss_fn(softened_teacher, softened_student)

            # Blended loss
            loss = self.alpha * distill_loss + (1.0 - self.alpha) * student_loss

        # Back-prop into *student* only
        grads = tape.gradient(loss, self.student.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.student.trainable_variables))

        # Update standard metrics using student predictions
        self.compiled_metrics.update_state(y, student_preds)
        results = {m.name: m.result() for m in self.metrics}
        results["loss"] = loss
        return results

    def test_step(self, data):
        x, y = data
        student_preds = self.student(x, training=False)
        student_loss = self.student_loss_fn(y, student_preds)
        self.compiled_metrics.update_state(y, student_preds)
        results = {m.name: m.result() for m in self.metrics}
        results["loss"] = student_loss
        return results

    # Keras will call this for inference / .predict()
    def call(self, inputs, training=False):
        return self.student(inputs, training=training)

    def get_config(self):
        """Return serialisable config so the distiller can be cloned / saved."""
        return {
            **super().get_config(),
            "alpha": self.alpha,
            "temperature": self.temperature,
            "student": self.student.name,
            "teacher": self.teacher.name,
        }

    @classmethod
    def from_config(cls, config, custom_objects=None):
        # NOTE: We cannot recreate student/teacher from just their names;
        # this helper will be invoked only when the user passes the actual
        # models in `custom_objects`.
        student = custom_objects["student"]
        teacher = custom_objects["teacher"]
        return cls(student=student, teacher=teacher, name=config.get("name", "distiller"))

    # -------------------------
    # Convenience I/O helpers
    # -------------------------
    def save_student(self, filepath, **kwargs):
        """Save *only* the distilled student network (small file, H5/Keras)."""
        self.student.save(filepath, **kwargs)

    def to_tflite(self, filepath, representative_data=None, quantize=True):
        """Export the student to (optionally quantized) TFLite.

        representative_data – generator that yields one float32 batch so the
                               converter can calibrate INT8 ranges.
        """
        converter = tf.lite.TFLiteConverter.from_keras_model(self.student)
        if quantize:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            if representative_data is not None:
                converter.representative_dataset = representative_data
        tflite_model = converter.convert()
        with open(filepath, "wb") as f:
            f.write(tflite_model)
        return filepath

# Helper that plugs Distiller into the existing training utility ----------------

def train_distiller(
    student: Model,
    teacher: Model,
    train_ds,
    valid_ds,
    config: Config,
    class_weight,
    *,
    alpha: float = 0.1,
    temperature: float = 4.0,
):
    """Convenience wrapper to train *student* via knowledge-distillation.

    It reuses the original `train_model` so you keep checkpointing, TensorBoard
    logging, etc., unchanged.
    """
    distiller = Distiller(student=student, teacher=teacher)
    distiller.compile(
        optimizer=tf.keras.optimizers.Adam(),
        metrics=[
            AUC(curve="PR", name="average_precision"),
            Precision(name="precision", class_id=1),
            Recall(name="recall", class_id=1),
        ],
        alpha=alpha,
        temperature=temperature,
    )
    # Uses the same training loop + callbacks defined above
    return train_model(distiller, train_ds, valid_ds, config, class_weight)

# --------------------------------------------------------------------------------