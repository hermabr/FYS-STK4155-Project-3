import tensorflow as tf
from tensorflow.keras import layers, models


def get_model():
    model = models.Sequential()
    #  model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=(32, 32, 2)))
    model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=(64, 64, 2)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dense(2))

    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    return model
