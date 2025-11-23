import tensorflow as tf
from tensorflow.keras import layers, models

def build_sod_model(input_shape=(128, 128, 3)):
    inputs = layers.Input(shape=input_shape)

    # Encoder Block 1
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
    x = layers.MaxPooling2D(2)(x) #128 -> 64

    # Encoder Block 2
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D(2)(x) #64 -> 32

    # Encoder Block 3
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D(2)(x) #32 -> 16

    # Bottleneck
    x = layers.Conv2D(256, 3, padding="same", activation="relu")(x)
  
    # Decoder Block 1 (Up 16 -> 32)
    x = layers.Conv2DTranspose(128, 3, strides=2, padding="same", activation="relu")(x)
    
    # Decoder Block 2 (Up 32 -> 64)
    x = layers.Conv2DTranspose(64, 3, strides=2, padding="same", activation="relu")(x)
  
    # Decoder Block 3 (Up 64 -> 128)
    x = layers.Conv2DTranspose(32, 3, strides=2, padding="same", activation="relu")(x)
   
    # Output Layer
    outputs = layers.Conv2D(
        1,
        kernel_size=1,
        padding="same",
        activation="sigmoid",
        name="saliency_mask"
    )(x)

    return models.Model(inputs, outputs)
  