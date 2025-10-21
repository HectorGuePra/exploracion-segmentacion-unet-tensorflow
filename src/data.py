import tensorflow as tf
import tensorflow_datasets as tfds

# Default constants
IMG_SIZE = (128, 128)
BATCH_SIZE = 64
BUFFER_SIZE = 1000


def normalize(input_image, input_mask):
    input_image = tf.cast(input_image, tf.float32) / 255.0
    input_mask -= 1
    return input_image, input_mask

def load_image(datapoint):
  input_image = tf.image.resize(datapoint['image'], (128, 128))
  input_mask = tf.image.resize(datapoint['segmentation_mask'], (128, 128))

  input_image, input_mask = normalize(input_image, input_mask)

  return input_image, input_mask


class Augment(tf.keras.layers.Layer):
    def __init__(self, seed: int = 42):
        super().__init__()
        self.augment_inputs = tf.keras.layers.RandomFlip(mode="horizontal", seed=seed)
        self.augment_labels = tf.keras.layers.RandomFlip(mode="horizontal", seed=seed)

    def call(self, inputs, labels):
        inputs = self.augment_inputs(inputs)
        labels = self.augment_labels(labels)
        return inputs, labels


def make_datasets(dataset_name: str = 'oxford_iiit_pet:4.0.0',
                  batch_size: int = BATCH_SIZE,
                  buffer_size: int = BUFFER_SIZE,
                  with_info: bool = True):
    dataset, info = tfds.load(dataset_name, with_info=with_info)
    train_images = dataset['train'].map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    test_images = dataset['test'].map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    train_batches = (
        train_images
        .cache()
        .shuffle(buffer_size)
        .batch(batch_size)
        .repeat()
        .map(Augment())
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )
    
    test_batches = test_images.batch(batch_size)
    return train_batches, test_batches, info
