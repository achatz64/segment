def inbound_layers(layer):
    """
    Not sure this works.
    """
    result = []
    for node in layer._inbound_nodes:
        result += node.inbound_layers
    return result
