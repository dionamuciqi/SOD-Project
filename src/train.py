import tensorflow as tf
from sod_model import build_sod_model
from data_loader import create_datasets

# IoU METRIC
def iou_metric(y_true, y_pred, smooth=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])

    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - intersection

    iou = (intersection + smooth) / (union + smooth)
    return iou


# BCE LOSS
bce_loss_fn = tf.keras.losses.BinaryCrossentropy()

# Combined Loss
def sod_loss(y_true, y_pred):
    bce = bce_loss_fn(y_true, y_pred)
    iou = iou_metric(y_true, y_pred)
    return bce + 0.5 * (1 - iou)


# Load datasets
train_ds, val_ds, test_ds = create_datasets(
    r"C:\Users\PC-STYLE\Desktop\SOD-Project\data\ECSSD\train",
    r"C:\Users\PC-STYLE\Desktop\SOD-Project\data\ECSSD\train_mask"
)

# Build model
model = build_sod_model()

# Compile model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=sod_loss,
    metrics=[iou_metric]
)

print("Model compiled successfully.")

# EarlyStopping
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

# ModelCheckpoint 
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath="sod_model.h5",
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)

# Training Loop
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20,
    callbacks=[early_stopping, checkpoint]
)

print("Training complete.")
model.save("sod_model.h5")
print("Final model saved.")
