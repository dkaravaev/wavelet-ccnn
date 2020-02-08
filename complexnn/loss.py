import keras.backend as K


def cmse(y_true, y_pred):
    return K.mean(K.square(y_pred[:, :, 0] - y_true[:, :, 0]) +
                  K.square(y_pred[:, :, 1] - y_true[:, :, 1]), axis=1)


def cmae(y_true, y_pred):
    return K.mean(K.sqrt(K.square(y_pred[:, :, 0] - y_true[:, :, 0]) +
                         K.square(y_pred[:, :, 1] - y_true[:, :, 1])), axis=1)