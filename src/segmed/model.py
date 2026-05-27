from __future__ import annotations


def build_unet(
    input_size: tuple[int, int] = (256, 256),
    base_filters: int = 32,
    batch_norm: bool = True,
    dropout: float = 0.1,
):
    from tensorflow.keras import Model
    from tensorflow.keras.layers import BatchNormalization, Conv2D, Conv2DTranspose, Dropout, Input, MaxPooling2D, concatenate

    def conv_block(inputs, filters: int):
        x = Conv2D(filters, (3, 3), activation="relu", padding="same")(inputs)
        if batch_norm:
            x = BatchNormalization()(x)
        x = Conv2D(filters, (3, 3), activation="relu", padding="same")(x)
        if batch_norm:
            x = BatchNormalization()(x)
        return x

    inputs = Input((input_size[0], input_size[1], 1))
    c1 = conv_block(inputs, base_filters)
    p1 = Dropout(dropout)(MaxPooling2D((2, 2))(c1))
    c2 = conv_block(p1, base_filters * 2)
    p2 = Dropout(dropout)(MaxPooling2D((2, 2))(c2))
    c3 = conv_block(p2, base_filters * 4)
    p3 = Dropout(dropout * 2)(MaxPooling2D((2, 2))(c3))
    c4 = conv_block(p3, base_filters * 8)
    p4 = Dropout(dropout * 2)(MaxPooling2D((2, 2))(c4))
    c5 = conv_block(p4, base_filters * 16)

    u6 = concatenate([Conv2DTranspose(base_filters * 8, (2, 2), strides=(2, 2), padding="same")(c5), c4], axis=3)
    c6 = conv_block(u6, base_filters * 8)
    u7 = concatenate([Conv2DTranspose(base_filters * 4, (2, 2), strides=(2, 2), padding="same")(c6), c3], axis=3)
    c7 = conv_block(u7, base_filters * 4)
    u8 = concatenate([Conv2DTranspose(base_filters * 2, (2, 2), strides=(2, 2), padding="same")(c7), c2], axis=3)
    c8 = conv_block(u8, base_filters * 2)
    u9 = concatenate([Conv2DTranspose(base_filters, (2, 2), strides=(2, 2), padding="same")(c8), c1], axis=3)
    c9 = conv_block(u9, base_filters)

    outputs = Conv2D(1, (1, 1), activation="sigmoid")(c9)
    return Model(inputs=[inputs], outputs=[outputs])


def build_legacy_unet(input_size: tuple[int, int] = (256, 256)):
    from tensorflow.keras import Model
    from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Input, MaxPooling2D, concatenate

    inputs = Input((input_size[0], input_size[1], 1))
    conv1 = Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
    conv1 = Conv2D(32, (3, 3), activation="relu", padding="same")(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(64, (3, 3), activation="relu", padding="same")(pool1)
    conv2 = Conv2D(64, (3, 3), activation="relu", padding="same")(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(128, (3, 3), activation="relu", padding="same")(pool2)
    conv3 = Conv2D(128, (3, 3), activation="relu", padding="same")(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(256, (3, 3), activation="relu", padding="same")(pool3)
    conv4 = Conv2D(256, (3, 3), activation="relu", padding="same")(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    conv5 = Conv2D(512, (3, 3), activation="relu", padding="same")(pool4)
    conv5 = Conv2D(512, (3, 3), activation="relu", padding="same")(conv5)
    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding="same")(conv5), conv4], axis=3)
    conv6 = Conv2D(256, (3, 3), activation="relu", padding="same")(up6)
    conv6 = Conv2D(256, (3, 3), activation="relu", padding="same")(conv6)
    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding="same")(conv6), conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), activation="relu", padding="same")(up7)
    conv7 = Conv2D(128, (3, 3), activation="relu", padding="same")(conv7)
    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding="same")(conv7), conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), activation="relu", padding="same")(up8)
    conv8 = Conv2D(64, (3, 3), activation="relu", padding="same")(conv8)
    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding="same")(conv8), conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), activation="relu", padding="same")(up9)
    conv9 = Conv2D(32, (3, 3), activation="relu", padding="same")(conv9)
    outputs = Conv2D(1, (1, 1), activation="sigmoid")(conv9)
    return Model(inputs=[inputs], outputs=[outputs])


def build_model(architecture: str, input_size: tuple[int, int], base_filters: int, batch_norm: bool, dropout: float):
    if architecture == "legacy_unet":
        return build_legacy_unet(input_size)
    if architecture == "modern_unet":
        return build_unet(input_size, base_filters, batch_norm, dropout)
    raise ValueError(f"Unknown architecture: {architecture}")
