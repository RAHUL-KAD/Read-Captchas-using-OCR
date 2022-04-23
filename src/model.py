"""
In this file
    1. We are going to build a model 
    2. Train the model
"""

from src import preprocessing
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class CTCLayer(layers.Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.loss_fn = keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred):
        # Compute the training time loss value and add it to the layer
        # using self.add_loss()
        batch_len = tf.cast(tf.shape(y_true)[0], dtype='int64')
        input_length = tf.cast(tf.shape(y_pred)[1], dtype='int64')
        label_length = tf.cast(tf.shape(y_true)[1], dtype='int64')

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype='int64')
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype='int64')

        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)

        # at test time, just return the computed predictions
        return y_pred


def build_model():
    # input to the model
    input_img = layers.Input(
        shape=(preprocessing.IMAGE_WIDTH, preprocessing.IMAGE_HEIGHT, 1), name='image', dtype='float32'
    )
    labels = layers.Input(name='label', shape=(None, ), dtype='float32')

    # First conv block
    x = layers.Conv2D(
        32,
        (3, 3),
        activation='relu',
        kernel_initializer='he_normal',
        padding='same',
        name='conv1'
    )(input_img)
    x = layers.MaxPooling2D((2, 2), name='pool1')(x)

    # Second Conv block
    x = layers.Conv2D(
        64,
        (3, 3),
        activation='relu',
        kernel_initializer='he_normal',
        padding='same',
        name='conv2'
    )(x)
    x = layers.MaxPooling2D((2, 2), name='pool2')(x)

    # we have used two max pool with size and stride 2.
    # Hence, downsampled featured maps are 4x smaller. The number of 
    # filters in last layer is 64. 
    # Reshape accordingly before passing the output to the RNN part of the model.
    new_shape = ((preprocessing.IMAGE_WIDTH // 4), (preprocessing.IMAGE_HEIGHT // 4) * 64)
    x = layers.Reshape(target_shape=new_shape, name='reshape')(x)
    x = layers.Dense(64, activation='relu', name='dense1')(x)
    x = layers.Dropout(0.2)(x)

    # RNNs 
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.25))(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.25))(x)

    # Output layer
    x = layers.Dense(
        len(preprocessing.int_to_char.get_vocabulary()) + 1, activation='softmax', name='dense2'
    )(x)

    # Add CTC layer for calculiting CTC loss at each step
    output = CTCLayer(name='ctc_loss')(labels, x)

    # Define the Model
    model = keras.models.Model(
        inputs=[input_img, labels], outputs=output, name='ocr_model_v1'
    )

    # optimizer
    opt = keras.optimizers.Adam()

    # compile the model and return
    model.compile(optimizer = opt)
    return model

# get the model
model = build_model()
model.summary()

# Training
epochs = 100
early_stopping_patience = 10
# add early stopping
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss', patience = early_stopping_patience, restore_best_weights=True
)

# Train the model
history = model.fit(
    preprocessing.train_dataset,
    validation_data = preprocessing.validation_dataset,
    epochs=epochs,
    callbacks=[early_stopping]
)
