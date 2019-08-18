from tensorflow.python.keras.engine.base_layer import Layer
from typing import List

def inbound_layers(layer: Layer) -> List[Layer]:
    """
    Not sure this works.
    """
    result = []
    for node in layer._inbound_nodes:
        result += node.inbound_layers
    return result
