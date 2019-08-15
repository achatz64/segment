from tensorflow.python.keras.layers import SeparableConv2D, BatchNormalization, Convolution2D, Input
from tensorflow.python.keras import Model

def sep_conv(filter_in: int, filter_out: int, repetition: int = 1, kernel: tuple = (2, 2),
             depth_multiplier: int = 1, batch_normalization: bool = False):
    """
    Simple separable convolutions cascade model.
    """

    input_tensor = Input(shape=(None, None, filter_in))

    # contracting immediately, no depth multiplication here
    x = SeparableConv2D(filters=filter_out, kernel_size=kernel)(input_tensor)

    if batch_normalization:
        x = BatchNormalization()(x)

    # rest is repetition
    for i in range(repetition-1):
        x = SeparableConv2D(filters=filter_out, kernel_size=kernel, depth_multiplier=depth_multiplier)(x)

        if batch_normalization:
            x = BatchNormalization()(x)

    return Model(input_tensor, x)

