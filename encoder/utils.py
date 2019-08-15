import utils


def find_poolpoints(model):
    output_dims = utils.output_shapes(model)
    breakpoints = {}

    input_shape = output_dims[0]

    for i, shape in enumerate(output_dims):
        if (input_shape[1] + 1) // 2 >= shape[1]:
            breakpoints[input_shape[1:3]] = i - 1
        input_shape = shape

    # last layer
    if len(input_shape) == 4:
        breakpoints[input_shape[1:3]] = i
    else:
        breakpoints[(1, 1)] = i

    return breakpoints

