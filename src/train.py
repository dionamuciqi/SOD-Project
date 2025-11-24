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

    return (intersection + smooth) / (union + smooth)



# BCE LOSS

bce_loss_fn = tf.keras.losses.BinaryCrossentropy()


# Combined SOD Loss: BCE + 0.5 * (1 - IoU)

def sod_loss(y_true, y_pred):
    bce = bce_loss_fn(y_true, y_pred)
    iou = iou_metric(y_true, y_pred)
    return bce + 0.5 * (1 - iou)



train_ds, val_ds, test_ds = create_datasets(
    r"C:\Users\PC-STYLE\Desktop\SOD-Project\data\ECSSD\train",
    r"C:\Users\PC-STYLE\Desktop\SOD-Project\data\ECSSD\train_mask"
)


# Build SOD model

model = build_sod_model()


optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)


# Compile model

model.compile(
    optimizer=optimizer,
    loss=sod_loss,
    metrics=[iou_metric]
)

print("Model compiled successfully.")



# BONUS FEATURE: Pause & Resume Training 

ckpt = tf.train.Checkpoint(
    step=tf.Variable(1),
    optimizer=optimizer,
    model=model
)

ckpt_manager = tf.train.CheckpointManager(
    ckpt,
    directory="./checkpoints",
    max_to_keep=3
)

# Resume if possible
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print(f"Resuming training from checkpoint at step {int(ckpt.step.numpy())}")
else:
    print("Starting training from scratch.")


# EarlyStopping callback

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)


# Save best model (ModelCheckpoint)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath="sod_model.h5",
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)


# Custom Training Loop (Required for BONUS)

EPOCHS = 20
current_step = int(ckpt.step.numpy())

for epoch in range(current_step, EPOCHS):

    print(f"\n----- Epoch {epoch + 1} / {EPOCHS} -----")

    # Train for ONE epoch
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=1,
        callbacks=[early_stopping, checkpoint]
    )

    
    ckpt.step.assign_add(1)

    
    ckpt_manager.save()
    print("Checkpoint saved.")

    
    if early_stopping.stopped_epoch > 0:
        print("Early stopping triggered.")
        break


