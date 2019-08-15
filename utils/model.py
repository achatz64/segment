from utils.layer import inbound_layers

def output_shapes(model, test_input=(None, 2056, 2056, 3)):
    """
    Will not work with concat and add layers...
    """

    if model.input_shape == (None, None, None, 3):
        input_shape = test_input
    else:
        return [layer.output_shape for layer in model.layers]

    layers = model.layers
    shapes = {}
    shapes_list = []

    for layer in layers:

        suggested_output = None

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

                if suggested_output is None or suggested == suggested_output:
                    suggested_output = suggested
                else:
                    raise ValueError("Cannot compute shapes")

            shapes[id(layer)] = suggested_output
            shapes_list.append(suggested_output)

    return shapes_list
