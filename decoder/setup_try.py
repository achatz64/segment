from tensorflow.python.keras.layers import SeparableConv2D, BatchNormalization, Input, Concatenate, Lambda
from tensorflow.python.keras.engine.training import Model as ModelType
from tensorflow.python.framework.ops import Tensor as TensorType
import tensorflow as tf
from tensorflow.python.keras import Model

from typing import List, Dict, Union, Tuple, Callable


def sep_conv_block(arg_list: List[Dict[str, Union[int, float, Tuple[Union[int, None]]]]],
                   batch_normalization: bool = False) -> Callable[[TensorType], TensorType]:

    """
    Simple separable convolutions cascade model.
    """
    def f(x):
        for a in arg_list:
            x = SeparableConv2D(**a)(x)

        if batch_normalization:
            x = BatchNormalization()(x)

        return x

    return f

def decoder_from_skeleton(skeleton: List[Dict[str, Union[int, Callable[[TensorType], TensorType]]]]) -> ModelType:
    """
    Decoder from Skeleton.

    skeleton = [{"e/d-conn": x, "channels": 120}, {"e-conn": y, "d-conn": z, "channels": 250}, ...]
    """

    # input layers
    inputs = [Input(shape=(None, None, channels)) for channels in [bone["channels"] for bone in skeleton]]
    image_shape_layer = Input(shape=(2,), dtype='int32')

    # first step
    f = skeleton[0]["e/d-conn"](inputs[0])

    # intermediate steps
    for i, bone in enumerate(skeleton[1:], 1):

        input_here = inputs[i]

        # connection from encoder
        from_encoder = bone["e-conn"](input_here)

        # connection from decoder
        if "up" in bone:
            f = bone["up"](f)

        # resizing for concatenation (unclear why wrapping in Lambda in necessary in 2.0 but not in 1.14)
        # f = Lambda(lambda x: tf.compat.v1.image.resize(f, tf.compat.v1.shape(x)[1:3]))(input_here)
        f = Lambda(lambda x: tf.image.resize(f, tf.shape(x)[1:3]))(input_here)


        concat = Concatenate()([f, from_encoder])

        # process concatenation

        f = bone["d-conn"](concat)

    # resizing to image dimensions (unclear why wrapping in Lambda in necessary in 2.0 but not in 1.14)
    #output = Lambda(lambda x: tf.compat.v1.image.resize(f, x[0]))(image_shape_layer)
    output = Lambda(lambda x: tf.image.resize(f, x[0]))(image_shape_layer)

    return Model((inputs, image_shape_layer), output)









