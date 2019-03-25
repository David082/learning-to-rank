# -*- coding: utf-8 -*-
"""
describe : 
author : yu_wei
created on : 2018/11/27
version : 
"""
import tensorflow as tf
from keras.models import load_model
import keras.backend as K


def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.

    https://stackoverflow.com/questions/45466020/how-to-export-keras-h5-to-tensorflow-pb

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


def keras_h5_to_tensorflow_pb(h5_file_path, pb_file_path):
    model = load_model(h5_file_path)  # add: custom_objects
    model.summary()
    frozen_graph = freeze_session(K.get_session(), output_names=[out.op.name for out in model.outputs])
    tf.train.write_graph(frozen_graph, "", pb_file_path, as_text=False)
    print('saved the freezed graph (ready for inference) at: ', pb_file_path)


def pb2log(model_filename, log_dir):
    """Load pb file to session and save to log.
    For tensorboard
    """
    with tf.Session() as sess:
        # model_filename ='PATH_TO_PB.pb'
        with tf.gfile.FastGFile(model_filename, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def)
    # LOGDIR='YOUR_LOG_LOCATION'
    train_writer = tf.summary.FileWriter(log_dir)
    train_writer.add_graph(sess.graph)


if __name__ == '__main__':
    keras_h5_to_tensorflow_pb("dcm_model_minmax.h5", "dcm_model_v2.pb")
    pb2log("lr.pb", "")
