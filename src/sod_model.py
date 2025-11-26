import tensorflow as tf
from tensorflow.keras import layers, models

# Conv block 
def conv_block(x, filters, name):
    x = layers.Conv2D(filters, 3, padding="same",
                      kernel_initializer="he_normal",
                      name=name+"_conv1")(x)
    x = layers.BatchNormalization(name=name+"_bn1")(x)
    x = layers.ReLU(name=name+"_relu1")(x)

    x = layers.Conv2D(filters, 3, padding="same",
                      kernel_initializer="he_normal",
                      name=name+"_conv2")(x)
    x = layers.BatchNormalization(name=name+"_bn2")(x)
    x = layers.ReLU(name=name+"_relu2")(x)

    return x

#  U-Net++ model 
def build_sod_model(input_shape=(128,128,3)):
    inputs = layers.Input(shape=input_shape)

    # Encoder
    x00 = conv_block(inputs, 32, "x00")
    p0 = layers.MaxPooling2D()(x00)

    x10 = conv_block(p0, 64, "x10")
    p1 = layers.MaxPooling2D()(x10)

    x20 = conv_block(p1, 128, "x20")
    p2 = layers.MaxPooling2D()(x20)

    x30 = conv_block(p2, 256, "x30")
    p3 = layers.MaxPooling2D()(x30)

    x40 = conv_block(p3, 512, "x40")

    # Decoder dense skip connections (U-Net++)
    x01 = conv_block(
        layers.Concatenate()([x00, layers.UpSampling2D()(x10)]),
        32, "x01"
    )

    x11 = conv_block(
        layers.Concatenate()([x10, layers.UpSampling2D()(x20)]),
        64, "x11"
    )

    x02 = conv_block(
        layers.Concatenate()([x00, x01, layers.UpSampling2D()(x11)]),
        32, "x02"
    )

    x21 = conv_block(
        layers.Concatenate()([x20, layers.UpSampling2D()(x30)]),
        128, "x21"
    )

    x12 = conv_block(
        layers.Concatenate()([x10, x11, layers.UpSampling2D()(x21)]),
        64, "x12"
    )

    x03 = conv_block(
        layers.Concatenate()([x00, x01, x02,
                              layers.UpSampling2D()(x12)]),
        32, "x03"
    )

    x31 = conv_block(
        layers.Concatenate()([x30, layers.UpSampling2D()(x40)]),
        256, "x31"
    )

    x22 = conv_block(
        layers.Concatenate()([x20, x21,
                              layers.UpSampling2D()(x31)]),
        128, "x22"
    )

    x13 = conv_block(
        layers.Concatenate()([x10, x11, x12,
                              layers.UpSampling2D()(x22)]),
        64, "x13"
    )

    x04 = conv_block(
        layers.Concatenate()([x00, x01, x02, x03,
                              layers.UpSampling2D()(x13)]),
        32, "x04"
    )

    output = layers.Conv2D(1, 1, activation="sigmoid")(x04)

    return models.Model(inputs, output)
