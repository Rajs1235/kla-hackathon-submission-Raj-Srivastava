import tensorflow as tf

from tensorflow.keras.layers import (
    Input,
    Conv2D,
    BatchNormalization,
    ReLU,
    UpSampling2D
)

from models.residual_block import residual_block

# ==========================
# INPUT
# ==========================

inputs = Input(shape=(128,128,1))

# ==========================
# FEATURE EXTRACTION
# ==========================

x = Conv2D(
    64,
    (3,3),
    padding="same"
)(inputs)

x = BatchNormalization()(x)

x = ReLU()(x)

# ==========================
# RESIDUAL BLOCKS
# ==========================

for _ in range(8):
    x = residual_block(x)

# ==========================
# UPSAMPLING
# ==========================

x = UpSampling2D(
    size=(2,2),
    interpolation="bilinear"
)(x)

# ==========================
# RECONSTRUCTION
# ==========================

x = Conv2D(
    32,
    (3,3),
    padding="same"
)(x)

x = BatchNormalization()(x)

x = ReLU()(x)

outputs = Conv2D(
    1,
    (3,3),
    activation="sigmoid",
    padding="same"
)(x)

# ==========================
# MODEL
# ==========================

model = tf.keras.Model(
    inputs,
    outputs,
    name="ResidualSRCNN"
)

model.compile(

    optimizer=tf.keras.optimizers.Adam(

        learning_rate=1e-3

    ),

    loss="mae"

)

model.summary()