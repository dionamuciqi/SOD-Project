import os
import glob
from typing import Tuple, List

import tensorflow as tf
from sklearn.model_selection import train_test_split


IMG_SIZE = 128

BATCH_SIZE = 8
AUTOTUNE = tf.data.AUTOTUNE
SEED = 42


#  Match image - mask 
def _collect_image_mask_pairs(img_dir: str, mask_dir: str) -> List[Tuple[str, str]]:
    img_files = sorted(glob.glob(os.path.join(img_dir, "*")))
    pairs = []

    for img_path in img_files:
        fname = os.path.basename(img_path)
        name_no_ext = os.path.splitext(fname)[0]

        mask_candidates = glob.glob(os.path.join(mask_dir, name_no_ext + ".*"))
        if not mask_candidates:
            continue

        pairs.append((img_path, mask_candidates[0]))

    return pairs


#  Load + preprocess
def load_image(image_path: tf.Tensor, mask_path: tf.Tensor):
    # Load image
    img = tf.io.read_file(image_path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    img = tf.cast(img, tf.float32) / 255.0

    # Load mask
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_image(mask, channels=1, expand_animations=False)
    mask = tf.image.resize(mask, (IMG_SIZE, IMG_SIZE))
    mask = tf.cast(mask, tf.float32) / 255.0

    return img, mask


# Augmentations 
def augment(img: tf.Tensor, mask: tf.Tensor):

    # Horizontal flip
    if tf.random.uniform(()) > 0.5:
        img = tf.image.flip_left_right(img)
        mask = tf.image.flip_left_right(mask)

    # Brightness 
    if tf.random.uniform(()) > 0.5:
        img = tf.image.adjust_brightness(img, delta=0.1)

    # Random 
    if tf.random.uniform(()) > 0.5:
        stacked = tf.concat([img, mask], axis=-1)
        crop = int(IMG_SIZE * 0.90)

        stacked = tf.image.random_crop(stacked, size=[crop, crop, 4])
        stacked = tf.image.resize(stacked, (IMG_SIZE, IMG_SIZE))

        img = stacked[..., :3]
        mask = stacked[..., 3:]

    return img, mask


#  Prepare datasets (train/val/test)
def create_datasets(img_dir: str, mask_dir: str):

    
    pairs = _collect_image_mask_pairs(img_dir, mask_dir)

    img_paths = [p[0] for p in pairs]
    mask_paths = [p[1] for p in pairs]

    # 70% train, 30% val+test
    train_img, valtest_img, train_mask, valtest_mask = train_test_split(
        img_paths, mask_paths,
        test_size=0.30,
        random_state=SEED
    )

    # 50/50 split e val/test => 15% + 15%
    val_img, test_img, val_mask, test_mask = train_test_split(
        valtest_img, valtest_mask,
        test_size=0.50,
        random_state=SEED
    )

    # Build TensorFlow pipelines
    def to_dataset(imgs, masks, aug=False):
        ds = tf.data.Dataset.from_tensor_slices((imgs, masks))
        ds = ds.map(load_image, num_parallel_calls=AUTOTUNE)

        if aug:
            ds = ds.map(augment, num_parallel_calls=AUTOTUNE)

        ds = ds.batch(BATCH_SIZE)
        ds = ds.prefetch(AUTOTUNE)
        return ds

    # Final sets
    train_ds = to_dataset(train_img, train_mask, aug=True)
    val_ds = to_dataset(val_img, val_mask, aug=False)
    test_ds = to_dataset(test_img, test_mask, aug=False)

    return train_ds, val_ds, test_ds
