from .layer import inbound_layers
from tensorflow.python.keras.engine.training import Model as ModelType
from typing import Tuple, List


def output_shapes(model: ModelType, test_input: Tuple[int] = (None, 2056, 2056, 3)) -> List[Tuple[int]]:
    """
    Will not work with concat and add layers...

    Attempts to compute the output shapes of the layers in model.layers for the test_input.
    """

    if model.input_shape == (None, None, None, 3):
        input_shape = test_input
    else:
        shapes_list = []
        for layer in model.layers:
            output_shape = layer.output_shape

            if isinstance(output_shape, list):
                output_shape = output_shape[0]

            shapes_list.append(output_shape)

        return shapes_list

    layers = model.layers
    shapes = {}
    shapes_list = []

    for layer in layers:

        inbound: list = inbound_layers(layer)

        if not inbound:
            shapes[id(layer)] = input_shape
            shapes_list.append(input_shape)
        else:
            for inbound_layer in inbound:
                inbound_layer_shape = shapes[id(inbound_layer)]
                try:
                    suggested = layer.compute_output_shape(inbound_layer_shape)
                except:
                    suggested = inbound_layer_shape

            shapes[id(layer)] = suggested
            shapes_list.append(suggested)

    return shapes_list
