from tensorflow.python.keras.layers import SeparableConv2D, BatchNormalization, Input
from tensorflow.python.keras.engine.training import Model as ModelType
from tensorflow.python.framework.ops import Tensor as TensorType
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

