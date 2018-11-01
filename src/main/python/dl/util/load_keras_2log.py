# -*- coding: utf-8 -*-
"""
describe : 
author : yu_wei
created on : 2018/11/1
version :
refer :
-- load keras model and then save to log
https://stackoverflow.com/questions/42112260/how-do-i-use-the-tensorboard-callback-of-keras
"""
from keras.models import load_model
import keras
import keras.backend as K


def alpha_crossentropy(y_true, y_pred):
    alpha = 0.15
    y_pred = K.exp(y_pred - alpha * y_true)
    y_pred /= K.sum(y_pred, axis=1, keepdims=True)
    return K.categorical_crossentropy(y_true, y_pred)


if __name__ == '__main__':
    h5_file_path = "model.h5"
    model = load_model("resources/dcm/dcm_model.h5", custom_objects={'alpha_crossentropy': alpha_crossentropy})
    # keras.callbacks.TensorBoard(log_dir='', histogram_freq=0, write_graph=True, write_images=True)
    tbCallBack = keras.callbacks.TensorBoard(log_dir='', histogram_freq=0,
                                             write_graph=True, write_images=True)
    tbCallBack.set_model(model)
