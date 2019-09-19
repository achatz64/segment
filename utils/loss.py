import tensorflow.keras.backend as K

def focal_loss(y_true, y_pred):

  epsilon = K.epsilon()
  y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

  return -y_true * K.pow(1 - y_pred, 2) * K.log(y_pred)