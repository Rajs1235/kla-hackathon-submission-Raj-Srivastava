from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import ReLU


def residual_block(x):

    shortcut = x

    x = Conv2D(
        64,
        (3,3),
        padding="same"
    )(x)

    x = BatchNormalization()(x)

    x = ReLU()(x)

    x = Conv2D(
        64,
        (3,3),
        padding="same"
    )(x)

    x = BatchNormalization()(x)

    x = Add()([shortcut, x])

    x = ReLU()(x)

    return x