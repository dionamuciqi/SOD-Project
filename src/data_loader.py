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

# Build full dataset pipeline

def create_datasets(img_dir: str, mask_dir: str):
    pairs = _collect_image_mask_pairs(img_dir, mask_dir)

    image_paths = [p[0] for p in pairs]
    mask_paths = [p[1] for p in pairs]

    train_img, valtest_img, train_mask, valtest_mask = train_test_split(
        image_paths, mask_paths, test_size=0.30, random_state=SEED 
    )

    val_img, test_img, val_mask, test_mask = train_test_split(
        valtest_img, valtest_mask, test_size=0.50, random_state=SEED
    )

    def to_dataset(imgs, masks, augment_flag=False):
        ds = tf.data.Dataset.from_tensor_slices((imgs, masks))
        ds = ds.map(load_image, num_parallel_calls=AUTOTUNE)

        if augment_flag:
            ds = ds.map(augment, num_parallel_calls=AUTOTUNE)

        ds = ds.batch(BATCH_SIZE)
        ds = ds.prefetch(AUTOTUNE)
        return ds 
    
    def to_dataset(imgs, masks, augment_flag=False):
        ds = tf.data.Dataset.from_tensor_slices((imgs, masks))
        ds = ds.map(load_image, num_parallel_calls=AUTOTUNE)

        if augment_flag:
            ds = ds.map(augment, num_parallel_calls=AUTOTUNE)

            ds = ds.batch(BATCH_SIZE)
            ds = ds.prefetch(AUTOTUNE)
            return ds
    train_ds = to_dataset(train_img, train_mask, augment_flag=True)
    val_ds = to_dataset(val_img,val_mask, augment_flag=False)
    test_ds = to_dataset(test_img, test_mask, augment_flag=False)

    return train_ds, val_ds, test_ds