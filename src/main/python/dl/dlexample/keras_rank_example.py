# -*- coding: utf-8 -*-
"""
describe : 
author : yu_wei
created on : 2018/7/3
version :
refer :
"""
import pandas as pd
import keras
from keras.layers import Input, Dense, Lambda
from keras.regularizers import l2
from keras.models import Model
from keras import optimizers
import tensorflow as tf
from keras.callbacks import TensorBoard
from keras.engine.topology import InputSpec
from keras.engine.topology import Layer

FEATURES = ["hotel_brand", "hotel_zone", "hotel_star", "hotel_goldstar", "hotel_fromtrain", "hotel_fromcitycenter",
            "hotel_fromairport", "hotel_score", "hotel_novoters", "hotel_fitmentyear_diffdays",
            "hotel_ctr_total",
            "hotel_uv_list_avg_7d", "hotel_ciireceivable_sum_7d", "hotel_pv_list_avg_7d", "hotel_order_ratio_zone",
            "hotel_order_cnt_7d", "delta_hotel_self_price", "fh_price", "hotel_zone_avg_order_cnt_7d",
            "dynamic_price_down",
            "delta_hotel_zone_price", "delta_hotel_city_price", "hotel_fh_price_avg_7d",
            "hotel_uv_ratio_zone", "hotel_mainscore",
            "hotel_ctr_is_dist_qry_1", "hotel_ctr_is_dist_qry_0",
            "hotel_zone_avg_order_cnt_31d", "hotel_tag_60_on", "hotel_avg_picturerank", "hotel_tag_xiuxiandujia_r",
            "hotel_ctr_is_price_qry_1",
            "hotel_tag_20_30"]


class KerasPipeline:
    def __init__(self, train=None):
        self.train = train

    def fit_min_max(self):
        from sklearn.preprocessing import MinMaxScaler
        min_max = MinMaxScaler()
        min_max.fit(self.train.loc[:, FEATURES])
        return min_max.data_min_, min_max.data_max_

    def feature_minmax_dict(self):
        min_values, max_values = self.fit_min_max()
        min_f = [f + "_min" for f in FEATURES]
        max_f = [f + "_max" for f in FEATURES]
        min_params = dict(zip(min_f, min_values))
        max_params = dict(zip(max_f, max_values))
        return min_params, max_params


class MinMaxScaler(Layer):
    def __init__(self, min_value=0, max_value=1, **kwargs):
        self.axis = 1
        self.min_value = min_value
        self.max_value = max_value
        super(MinMaxScaler, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        super(MinMaxScaler, self).build(input_shape)
        self.built = True

    def call(self, x, mask=None):
        if self.min_value == self.max_value:
            return (x - x)
        else:
            output = (x - self.min_value) / (self.max_value - self.min_value)
            return output


class SimpleNet:
    def __init__(self, learning_rate=0.001, epochs=10, batch_size=10000, l2_lambda=0.01,
                 n_hidden_1=20, n_hidden_2=10,
                 num_features=33, num_classes=1,
                 log_dir='logs',
                 min_values=None, max_values=None
                 ):
        # Hyper Parameters
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.l2_lambda = l2_lambda
        # Preprocessing
        self.min_values = min_values
        self.max_values = max_values
        # Network Parameters
        self.num_features = num_features
        self.num_classes = num_classes
        self.n_hidden_1 = n_hidden_1
        self.n_hidden_2 = n_hidden_2
        # TensorBoard log
        self.log_dir = log_dir

    def build_network(self):
        def input_feature(name):
            return Input(shape=(1,), name=name)

        def minmax(x, min, max):
            return (x - min) / (max - min)

        def dense_layer(units, activation, name):
            return Dense(units, activation=activation, name=name)

        def dense_regularizer_layer(units, activation, name):
            return Dense(units, activation=activation, kernel_regularizer=l2(self.l2_lambda), name=name)

        # input
        input_f = [input_feature(f) for f in FEATURES]
        # minmax layer
        minmax = [Lambda(minmax, arguments={'min': self.min_values[i], 'max': self.max_values[i]},
                         name=f + "_minmax")(input_f[i]) for (i, f) in enumerate(FEATURES)]
        # concat layer
        input_x = keras.layers.concatenate(minmax)
        # hidden1
        hidden1 = Dense(self.n_hidden_1, activation='relu', kernel_regularizer=l2(self.l2_lambda),
                        name='hidden1')(input_x)
        # hidden2
        hidden2 = Dense(self.n_hidden_2, activation='relu', kernel_regularizer=l2(self.l2_lambda),
                        name='hidden2')(hidden1)
        # output
        output = Dense(self.num_classes, activation='sigmoid', kernel_regularizer=l2(self.l2_lambda),
                       name='output')(hidden2)
        # model
        model = Model(inputs=input_f, outputs=output)
        sgd = optimizers.SGD(lr=self.learning_rate)
        model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])

        return model

    def train_network(self, train_x, train_y, h5_file_path):
        model = self.build_network()
        # summarize layers
        print(model.summary())
        model.fit(train_x, train_y,
                  validation_split=0.3, epochs=self.epochs, batch_size=self.batch_size,
                  verbose=1,
                  callbacks=[TensorBoard(log_dir=self.log_dir)])
        # save model
        print("[============================>] - save h5")
        model.save(h5_file_path)
        return model


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
    from keras import backend as K
    from keras.models import load_model

    model = load_model(h5_file_path)
    frozen_graph = freeze_session(K.get_session(), output_names=[out.op.name for out in model.outputs])
    tf.train.write_graph(frozen_graph, "", pb_file_path, as_text=False)
    print('saved the freezed graph (ready for inference) at: ', pb_file_path)


def h5_to_pb(h5_file_path, pb_file_path, keep_var_names=None, output_names=None, clear_devices=True):
    from keras import backend as K
    from tensorflow.python.framework import graph_util
    from tensorflow.python.framework import graph_io
    from keras.models import load_model

    sess = K.get_session()
    graph = sess.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = graph_util.convert_variables_to_constants(sess, input_graph_def, output_names, freeze_var_names)

    # lodel keras model
    K.set_learning_phase(0)
    model = load_model(h5_file_path)
    graph_io.write_graph(frozen_graph, "", pb_file_path, as_text=False)
    print('saved the freezed graph (ready for inference) at: ', pb_file_path)


def view_graph(pb_file_path, log_dir):
    with tf.Session() as sess:
        # model_filename ='PATH_TO_PB.pb'
        with tf.gfile.FastGFile(pb_file_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def)
    # LOGDIR='YOUR_LOG_LOCATION'
    train_writer = tf.summary.FileWriter(log_dir)
    train_writer.add_graph(sess.graph)


if __name__ == "__main__":
    # =============================== Read in data
    file_path = "D:/personalizedRanking/resources/data/data_train_test_sample"
    data = pd.read_csv(file_path, sep="\t")
    label = "booking_bool"

    train = data[(data['d'] != "2018-05-11")]
    test = data[(data['d'] == "2018-05-11")]

    # train set
    train_x = [train[f] for f in FEATURES]
    train_y = train[label]
    # test set
    test_x = [test[f] for f in FEATURES]
    test_y = test[label]

    # Fit to get min max from train set.
    kpipe = KerasPipeline(train)
    min_values, max_values = kpipe.fit_min_max()

    # =============================== train
    sn = SimpleNet(min_values=min_values, max_values=max_values)
    model = sn.train_network(train_x, train_y, "lr.h5")

    # =============================== udf
    # keras to tensorflow
    keras_h5_to_tensorflow_pb("lr.h5", "graph.pb")
    # view graph
    view_graph("graph.pb", "")
