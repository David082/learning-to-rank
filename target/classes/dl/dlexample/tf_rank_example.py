# -*- coding: utf-8 -*-
"""
describe : 
author : yu_wei
created on : 2018/6/13
version :
"""
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.python.framework import graph_util
from sklearn.metrics import roc_auc_score

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


class TensorPipeline:
    def __init__(self):
        pass

    def df_to_tensor(self, df):
        cols_dict = {k: tf.constant(df[k].values, dtype=tf.float32, name=k) for k in FEATURES}
        # https://www.tensorflow.org/api_docs/python/tf/concat
        cols_to_tensor = tf.concat([tf.reshape(cols_dict.get(k), shape=(-1, 1)) for k in FEATURES], axis=1, name="cols")
        return cols_to_tensor

    def fit_min_max(self, train):
        from sklearn.preprocessing import MinMaxScaler
        min_max = MinMaxScaler()
        min_max.fit(train.loc[:, FEATURES])
        return min_max.data_min_, min_max.data_max_


class SimpleNet:
    def __init__(self, learning_rate=0.1, l2_reg=0.0001, epochs=100, batch_size=10000,
                 display_step=10,
                 n_hidden_1=20, n_hidden_2=10,
                 num_features=33, num_classes=1,
                 min_values=None, max_values=None,
                 model=None):
        # Hyper Parameters
        self.learning_rate = learning_rate
        self.l2_reg = l2_reg
        self.epochs = epochs
        self.batch_size = batch_size
        self.display_step = display_step
        self.model = model  # l: logistic regression; m: multi layer

        # Network Parameters
        self.n_hidden_1 = n_hidden_1  # 1st layer number of neurons
        self.n_hidden_2 = n_hidden_2  # 2nd layer number of neurons
        self.num_features = num_features
        self.num_classes = num_classes  # classes

        # Preprocessing
        self.min_values = min_values
        self.max_values = max_values

    def build_network(self):
        cols_dict = {k: tf.placeholder(tf.float32, shape=[None, 1], name=k) for k in FEATURES}

        def min_max_func(x, min_x, max_x):
            if (min_x == max_x):
                return 0
            else:
                return (x - min_x) / (max_x - min_x)

        # ------ input
        x = tf.concat([(cols_dict.get(k) - self.min_values[i]) / (self.max_values[i] - self.min_values[i]) for (i, k) in
                       enumerate(FEATURES)], axis=1, name="input")  # input
        y = tf.placeholder(tf.float32, shape=[None, self.num_classes], name='labels_placeholder')

        def weight_variable(shape, name="weights"):
            initial = tf.random_normal(shape)
            return tf.Variable(initial, name=name)

        def bias_variable(shape, name="biases"):
            initial = tf.random_normal(shape)
            return tf.Variable(initial, name=name)

        def fc(input, w, b):
            return tf.matmul(input, w) + b

        # logistic regression
        with tf.name_scope('lr') as scope:
            weights = weight_variable([self.num_features, self.num_classes])
            biases = bias_variable([self.num_classes])
            output_lr = fc(x, weights, biases)

        # fc1
        with tf.name_scope('fc1') as scope:
            kernel = weight_variable([self.num_features, self.n_hidden_1])
            biases = bias_variable([self.n_hidden_1])
            output_fc1 = tf.nn.relu(fc(x, kernel, biases), name=scope)
        # fc2
        with tf.name_scope('fc2') as scope:
            kernel = weight_variable([self.n_hidden_1, self.n_hidden_2])
            biases = bias_variable([self.n_hidden_2])
            output_fc2 = tf.nn.relu(fc(output_fc1, kernel, biases), name=scope)
        # fc3
        with tf.name_scope('fc3') as scope:
            kernel = weight_variable([self.n_hidden_2, self.num_classes])
            biases = bias_variable([self.num_classes])
            output_fc3 = fc(output_fc2, kernel, biases)

        prob = None
        cost = None
        if self.model == "l":
            # ====== logistic regression
            prob = tf.nn.sigmoid(output_lr, name='output')
            cost = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=output_lr)
            ) + self.l2_reg * tf.nn.l2_loss(weights)
        elif self.model == "m":
            # ====== multi layer
            prob = tf.nn.sigmoid(output_fc3, name='output')
            vars = tf.trainable_variables()
            # l2_regularizer
            l2 = tf.add_n([tf.nn.l2_loss(v) for v in vars if 'bias' not in v.name]) * self.l2_reg
            cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=output_fc3)) + l2

        optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(cost)

        graph_dict = cols_dict
        graph_dict['y'] = y
        graph_dict['cost'] = cost
        graph_dict['prob'] = prob
        graph_dict['optimizer'] = optimizer

        return graph_dict

    def input_feed_dict(self, graph, df):
        feed_dict = {graph[k]: np.array(df[k].values, dtype="float32").reshape(-1, 1) for k in FEATURES}
        return feed_dict

    def train_network(self, graph, total_batch, pb_file_path):
        init = tf.global_variables_initializer()
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
            sess.run(init)
            for epoch in range(self.epochs):
                avg_cost = 0.
                for i in range(total_batch):
                    batch_x = train_x[self.batch_size * i: self.batch_size * (i + 1)]
                    batch_y = train_y[self.batch_size * i:self.batch_size * (i + 1)].reshape(-1, 1)

                    feed_dict = self.input_feed_dict(graph, batch_x)
                    feed_dict[graph['y']] = batch_y

                    _, c = sess.run([graph['optimizer'], graph['cost']], feed_dict=feed_dict)
                    avg_cost += c

                # feed dict
                train_feed_dict = self.input_feed_dict(graph, train_x)
                train_feed_dict[graph['y']] = train_y.reshape(-1, 1)

                test_feed_dict = self.input_feed_dict(graph, test_x)
                test_feed_dict[graph['y']] = test_y.reshape(-1, 1)

                # eval
                prob_train = sess.run([graph['prob']], feed_dict=train_feed_dict)
                print("AUC of train :", roc_auc_score(train_y[: len(prob_train[0])], prob_train[0]))
                prob_test = sess.run([graph['prob']], feed_dict=test_feed_dict)
                print("AUC of test :", roc_auc_score(test_y[: len(prob_test[0])], prob_test[0]))

                avg_cost = avg_cost / total_batch
                # Display logs per epoch step
                if (epoch + 1) % self.display_step == 0:
                    print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.3f}".format(avg_cost))
            # Save model to pb file
            constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ["output"])
            with tf.gfile.FastGFile(pb_file_path, mode='wb') as f:
                f.write(constant_graph.SerializeToString())
            # Save log for tensorboard
            tf.summary.FileWriter("logs", sess.graph)


def load_pb_model(pb_file_path, test_x_df):
    with tf.gfile.FastGFile(pb_file_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

    # graph weight and bias
    with tf.Session() as sess:
        # extract weights from pb file
        graph_nodes = [n for n in graph_def.node]
        wts = [n for n in graph_nodes if n.op == 'Const']
        from tensorflow.python.framework import tensor_util
        for n in wts:
            print("Name of the node - %s" % n.name)
            print("Value - ")
            print(tensor_util.MakeNdarray(n.attr['value'].tensor))

        # input_x = sess.graph.get_tensor_by_name("input")
        input_dict = {
            sess.graph.get_tensor_by_name(k + ":0"): np.array(test_x_df[k].values, dtype="float32").reshape(-1, 1)
            for k in FEATURES
            }
        prob = sess.graph.get_tensor_by_name("output:0")
        predict = sess.run(prob, feed_dict=input_dict)
        return predict


if __name__ == "__main__":
    # =============================== Read in data
    file_path = "D:/personalizedRanking/resources/data/data_train_test_sample"
    data = pd.read_csv(file_path, sep="\t")
    label = "booking_bool"

    train = data[(data['d'] != "2018-05-11")]
    test = data[(data['d'] == "2018-05-11")]

    # Fit to get min max from train set.
    tp = TensorPipeline()
    min_values, max_values = tp.fit_min_max(train)

    # train_x = np.array(train_x, dtype="float32")
    train_x = train.loc[:, FEATURES]
    train_y = np.array(train[label], dtype="float32")

    # test_x = np.array(test_x, dtype="float32")
    test_x = test.loc[:, FEATURES]
    test_y = np.array(test[label], dtype="float32")

    # =============================== Graph train
    sn = SimpleNet(min_values=min_values, max_values=max_values, model="m")
    total_batch = len(train_y) // sn.batch_size
    g = sn.build_network()
    # feed_dict = sn.graph_feed_dict(g, test)
    # feed_dict.get(g['hotel_brand'])
    # len(feed_dict)
    sn.train_network(graph=g, total_batch=total_batch, pb_file_path="lr.pb")

    # =============================== Load pb file
    scores = load_pb_model("lr.pb", test_x)
    scores = scores.reshape(-1)
    pre = test_x
    pre[label] = test[label]
    pre['scores'] = scores
    roc_auc_score(pre.loc[:, label], pre.loc[:, "scores"])
    pre.iloc[0:1000, ].to_csv("pre.csv", sep=",", index=False)
