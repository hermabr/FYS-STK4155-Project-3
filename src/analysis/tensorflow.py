import numpy as np
from tqdm import tqdm
from data import FallData
import matplotlib.pyplot as plt

from cnn.tensorflow import get_model


def main():
    for test_size in tqdm(np.arange(0.1, 1, 0.1)):

        data = FallData(test_size=test_size, resize=(32, 32), for_tensorflow=True)

        #  train_dataset = tf.data.Dataset.from_tensor_slices((data.X_train, data.y_train))
        #  test_dataset = tf.data.Dataset.from_tensor_slices((data.X_test, data.y_test))
        #
        #  BATCH_SIZE = 4
        #  SHUFFLE_BUFFER_SIZE = 6
        #
        #  train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
        #  test_dataset = test_dataset.batch(BATCH_SIZE)
        model = get_model()

        model.fit(
            data.train_dataset,
            epochs=30,
            validation_data=data.test_dataset,
            verbose=False,
        )

        #  plt.plot(history.history["accuracy"], label="accuracy")
        #  plt.plot(history.history["val_accuracy"], label="val_accuracy")
        #  plt.xlabel("Epoch")
        #  plt.ylabel("Accuracy")
        #  plt.ylim([0.5, 1.01])
        #  plt.legend(loc="lower right")
        #  plt.show()

        test_loss, test_acc = model.evaluate(data.test_dataset, verbose=False)
        tqdm.write(f"{test_size*100:.0f}%: {test_acc*100:.0f}%")
