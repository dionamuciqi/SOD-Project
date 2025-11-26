import tensorflow as tf
from sod_model import build_sod_model
from data_loader import create_datasets

# IoU metric
def iou_metric(y_true, y_pred, smooth=1e-6):
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    y_true = tf.cast(y_true, tf.float32)
    inter = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - inter
    return (inter + smooth) / (union + smooth)

bce = tf.keras.losses.BinaryCrossentropy()

def sod_loss(y_true, y_pred):
    return bce(y_true, y_pred) + 0.5 * (1 - iou_metric(y_true, y_pred))

print("Loading dataset...")
train_ds, val_ds, test_ds = create_datasets(
    r"C:\Users\PC-STYLE\Desktop\SOD-Project\data\ECSSD\train",
    r"C:\Users\PC-STYLE\Desktop\SOD-Project\data\ECSSD\train_mask"
)

print("Building U-Net++ model...")
model = build_sod_model()

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss=sod_loss,
    metrics=[iou_metric]
)

ckpt_cb = tf.keras.callbacks.ModelCheckpoint(
    "sod_model.h5",
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=7,
    restore_best_weights=True
)

print("Training started...")
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=25,
    callbacks=[ckpt_cb, early_stop]
)

print("Training finished.")
