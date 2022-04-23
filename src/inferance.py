from src import preprocessing, model, data
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


# Get the prediction model by extracting layers till the output layer
prediction_model = keras.models.Model(
    model.model.get_layer(name='image').input, model.model.get_layer(name='dense2').output
)
prediction_model.summary()

# A utility function to decode the output of the network
def decode_batch_predictions(pred):
    input_len = np.ones(pred.shape[0] * pred.shape[1])
    # Use greedy search. For complex tasks, you can use bean search
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][
        :, :data.max_length
    ]
    # Iterate over the results and get back the text
    output_texts = []
    for res in results:
        res = tf.strings.reduce_join(preprocessing.int_to_char(res)).numpy().astype('utf-8')
        output_texts.append(res)
    return output_texts

# Let's check results on some validation samples
for batch in preprocessing.validation_dataset.take(1):
    batch_images = batch["image"]
    batch_labels = batch["label"]

    preds = prediction_model.predict(batch_images)
    pred_texts = decode_batch_predictions(preds)

    orig_texts = []
    for label in batch_labels:
        label = tf.strings.reduce_join(num_to_char(label)).numpy().decode("utf-8")
        orig_texts.append(label)

    _, ax = plt.subplots(4, 4, figsize=(15, 5))
    for i in range(len(pred_texts)):
        img = (batch_images[i, :, :, 0] * 255).numpy().astype(np.uint8)
        img = img.T
        title = f"Prediction: {pred_texts[i]}"
        ax[i // 4, i % 4].imshow(img, cmap="gray")
        ax[i // 4, i % 4].set_title(title)
        ax[i // 4, i % 4].axis("off")
plt.savefig('/ocr/Read-Captchas-using-OCR/tmp/prediction.png')
plt.show()