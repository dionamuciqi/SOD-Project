import os
import glob
from typing import Tuple, List

import tensorflow as tf
from sklearn.model_selection import train_test_split

IMG_SIZE = 128
BATCH_SIZE = 8
AUTOTUNE = tf.data.AUTOTUNE
SEED = 42


# Match images with masks

def _collect_image_mask_pairs(img_dir: str, mask_dir: str) -> List[Tuple[str, str]]:
    img_files = sorted(glob.glob(os.path.join(img_dir, "*")))
    pairs = []

    for img_path in img_files:
        fname = os.path.basename(img_path)
        name_no_ext = os.path.splitext(fname)[0]

        candidates = glob.glob(os.path.join(mask_dir, name_no_ext + ".*"))
        if not candidates:
            continue
        mask_path = candidates[0]
        pairs.append((img_path, mask_path))

    return pairs


# Preprocessing function

def load_image(image_path: tf.Tensor, mask_path: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:

    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = tf.cast(image, tf.float32) / 255.0

 
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_image(mask, channels=1, expand_animations=False)
    mask = tf.image.resize(mask, (IMG_SIZE, IMG_SIZE))
    mask = tf.cast(mask, tf.float32) / 255.0

    return image, mask



# Data Augmentation

def augment(image: tf.Tensor, mask: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:

    # Horizontal flip
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        mask = tf.image.flip_left_right(mask)

    # Brightness variation (only on image)
    if tf.random.uniform(()) > 0.5:
        image = tf.image.adjust_brightness(image, delta=0.1)

    # Random crop
    if tf.random.uniform(()) > 0.5:
        stacked = tf.concat([image, mask], axis=-1)
        crop_size = int(IMG_SIZE * 0.9)

        stacked = tf.image.random_crop(
            stacked,
            size=[crop_size, crop_size, 4]
        )

        stacked = tf.image.resize(stacked, (IMG_SIZE, IMG_SIZE))
        image = stacked[..., :3]
        mask = stacked[..., 3:]

    return image, mask
