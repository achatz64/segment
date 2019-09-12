from tensorflow.python.keras.layers import SeparableConv2D, BatchNormalization, Input, Concatenate, Lambda, Conv2D
from tensorflow.python.keras.engine.training import Model as ModelType
from tensorflow.python.framework.ops import Tensor as TensorType
from tensorflow.image import resize
from tensorflow import shape, map_fn, int32
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
        f = Lambda(lambda x: resize(x[1], shape(x[0])[1:3]))([input_here, f])

        concat = Concatenate()([f, from_encoder])

        # process concatenation

        f = bone["d-conn"](concat)

    # resizing to image dimensions (unclear why wrapping in Lambda in necessary in 2.0 but not in 1.14)
    #output = Lambda(lambda x: tf.compat.v1.image.resize(f, x[0]))(image_shape_layer)
    output = Lambda(lambda x: resize(x[1], x[0][0]))([image_shape_layer, f])

    return Model((inputs, image_shape_layer), output)


def baseline_skeleton(channels: List[Dict[str, int]],
                      batch_normalization=True,
                      kernel=(3, 3)):
    """
    Constructs a primitive decoder skeleton.

    channels is of the form:
    [{'e-channel': x, 'd-channel': y}, {'e-channel': xx, 'd-channel': yy}, ...]
    """

    assert len(channels) > 0, "Empty list of channels!"

    def squeeze(to_channels, activation="relu"):

        def out(tensor):
            tensor = SeparableConv2D(to_channels, kernel, padding="same", activation=activation)(tensor)

            if batch_normalization:
                tensor = BatchNormalization()(tensor)

            return tensor

        return out

    skeleton = [{"e/d-conn": squeeze(channels[0]['d-channel']),
                 "channels": channels[0]['e-channel']}]

    for channel_pair in channels[1:-1]:
        new = {"channels": channel_pair["e-channel"],
               "e-conn": Lambda(lambda x: x),
               "d-conn": squeeze(channel_pair["d-channel"])}
        skeleton.append(new)

    last = {"channels": channels[-1]["e-channel"],
            "e-conn": Lambda(lambda x: x),
            "d-conn": squeeze(channels[-1]["d-channel"], activation='softmax')}

    skeleton.append(last)

    return skeleton

def zero_skeleton(channels : List[int]):

    def edconn(f: TensorType):
        for c in channels[1:]:
            f = Conv2D(c, (1, 1), activation='relu')(f)
        return f

    return [{"channels": channels[0],
             "e/d-conn": edconn}]


def attach(encoder: ModelType,
           decoder: ModelType,
           use_layers: List[int]):

    # large layers first
    use_layers = sorted(use_layers, key=lambda x: -x)

    inputs = encoder.input
    layers = [encoder.layers[i] for i in use_layers]

    for l in layers:
        print(l.output_shape)

    out_decoder = decoder(
        [l.output for l in layers] + [Lambda(lambda x: map_fn(lambda y: shape(y)[0:2], x, dtype=int32))(inputs)])

    return Model(inputs, out_decoder)








