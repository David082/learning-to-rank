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
import tensorflow as tf


# https://stackoverflow.com/questions/45466020/how-to-export-keras-h5-to-tensorflow-pb
def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    from tensorflow.python.framework.graph_util import convert_variables_to_constants
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        print("freeze_var_names ======>", freeze_var_names)
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        print("input_graph_def ======>", input_graph_def.node)
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(session, input_graph_def, output_names, freeze_var_names)
        return frozen_graph


def alpha_crossentropy(y_true, y_pred):
    alpha = 0.15
    y_pred = K.exp(y_pred - alpha * y_true)
    y_pred /= K.sum(y_pred, axis=1, keepdims=True)
    return K.categorical_crossentropy(y_true, y_pred)


def keras_h5_to_tensorflow_pb(h5_file_path, pb_file_path):
    model = load_model(h5_file_path, custom_objects={'alpha_crossentropy': alpha_crossentropy})
    model.summary()
    frozen_graph = freeze_session(K.get_session(), output_names=[out.op.name for out in model.outputs])
    tf.train.write_graph(frozen_graph, "", pb_file_path, as_text=False)
    print('saved the freezed graph (ready for inference) at: ', pb_file_path)


if __name__ == '__main__':
    h5_file_path = "model.h5"
    model = load_model("resources/dcm/dcm_model.h5", custom_objects={'alpha_crossentropy': alpha_crossentropy})
    # keras.callbacks.TensorBoard(log_dir='', histogram_freq=0, write_graph=True, write_images=True)
    tbCallBack = keras.callbacks.TensorBoard(log_dir='', histogram_freq=0,
                                             write_graph=True, write_images=True)
    tbCallBack.set_model(model)

    # ------ h5 to pb
    keras_h5_to_tensorflow_pb("/home/jovyan/work/jattention/dcm/dcm_model_v2.1.h5", "dcm_model_v2.1.pb")
