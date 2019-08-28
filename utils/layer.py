from tensorflow.python.keras.engine.base_layer import Layer
from typing import List

def inbound_layers(layer: Layer) -> List[Layer]:
    """
    Not sure this works.
    """
    result = []
    for node in layer._inbound_nodes:
        inbound = node.inbound_layers
        if not isinstance(inbound, list):
            inbound = [inbound]

        result += inbound
    return result
