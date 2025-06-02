import os
import numpy as np
import sklearn.metrics as skmetrics
import tensorflow as tf
import timeit
import tensorflow.contrib.metrics as contrib_metrics
import tensorflow.contrib.slim as contrib_slim
import nn
import logging

logger = logging.getLogger("default_log")

class TinySleepNetCNN(object):
    def __init__(self, config, output_dir="./output", testing=False, use_best=False):
        self.config = config
        self.output_dir = output_dir
        self.checkpoint_path = os.path.join(self.output_dir, "checkpoint")
        self.best_ckpt_path = os.path.join(self.output_dir, "best_ckpt")
        self.weights_path = os.path.join(self.output_dir, "weights")
        self.log_dir = os.path.join(self.output_dir, "log")

        # # Placeholders
        # with tf.variable_scope("placeholders") as scope:
        #     self.signals = tf.placeholder(dtype=tf.float32, shape=(None, self.config["input_size"], 1, 1), name='signals')
        #     self.labels = tf.placeholder(dtype=tf.int32, shape=(None,), name='labels')
        #     self.is_training = tf.placeholder(dtype=tf.bool, shape=(), name='is_training')

        #Placeholders
        with tf.variable_scope("placeholders") as scope:
            self.signals = tf.placeholder(dtype=tf.float32, shape=(None, self.config["input_size"], 1, 1), name='signals')
            self.labels = tf.placeholder(dtype=tf.int32, shape=(None,), name='labels')
            self.is_training = tf.placeholder(dtype=tf.bool, shape=(), name='is_training')
            self.class_weights = tf.placeholder(dtype=tf.float32, shape=(self.config["n_classes"],), name='class_weights')

        # Global step and epoch
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.global_epoch = tf.Variable(0, trainable=False, name='global_epoch')

        # Build CNN
        net = self.build_cnn()

        # Softmax linear
        net = nn.fc("softmax_linear", net, self.config["n_classes"], bias=0.0)

        # Outputs
        self.logits = net
        self.preds = tf.argmax(self.logits, axis=1)

        # # Loss
        # self.loss_per_sample = tf.nn.sparse_softmax_cross_entropy_with_logits(
        #     labels=self.labels, logits=self.logits, name="loss_ce_per_sample")
        # self.loss_ce = tf.reduce_mean(self.loss_per_sample)
        # self.reg_losses = self.regularization_loss()
        # self.loss = self.loss_ce + self.reg_losses

        # Loss
        self.loss_per_sample = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.labels, logits=self.logits, name="loss_ce_per_sample")
        sample_weights = tf.reduce_sum(
            tf.multiply(
                tf.one_hot(indices=self.labels, depth=self.config["n_classes"]),
                self.class_weights
            ), 1
        )
        self.loss_per_sample = tf.multiply(self.loss_per_sample, sample_weights)
        self.loss_ce = tf.reduce_mean(self.loss_per_sample)
        self.reg_losses = self.regularization_loss()
        self.loss = self.loss_ce + self.reg_losses

        # Metrics
        with tf.variable_scope("stream_metrics") as scope:
            self.metric_value_op, self.metric_update_op = contrib_metrics.aggregate_metric_map({
                "loss": tf.metrics.mean(values=self.loss),
                "accuracy": tf.metrics.accuracy(labels=self.labels, predictions=self.preds),
                "precision": tf.metrics.precision(labels=self.labels, predictions=self.preds),
                "recall": tf.metrics.recall(labels=self.labels, predictions=self.preds),
            })
            metric_vars = contrib_slim.get_local_variables(scope=scope.name)
            self.metric_init_op = tf.variables_initializer(metric_vars)

        # Training and test outputs
        self.train_outputs = {
            "global_step": self.global_step,
            "train/loss": self.loss,
            "train/preds": self.preds,
            "train/stream_metrics": self.metric_update_op,
        }
        self.test_outputs = {
            "global_step": self.global_step,
            "test/loss": self.loss,
            "test/preds": self.preds,
        }

        # Session
        config_sess = tf.ConfigProto()
        config_sess.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config_sess)
        if not testing:
            self.train_writer = tf.summary.FileWriter(os.path.join(self.log_dir, "train"))
            self.train_writer.add_graph(self.sess.graph)
            logger.info("Saved tensorboard graph to {}".format(self.train_writer.get_logdir()))

        # Optimizer
        if not testing:
            self.lr = tf.constant(self.config["learning_rate"], dtype=tf.float32)
            with tf.variable_scope("optimizer") as scope:
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(update_ops):
                    self.train_step_op, self.grad_op = nn.adam_optimizer(
                        loss=self.loss,
                        training_variables=tf.trainable_variables(),
                        global_step=self.global_step,
                        learning_rate=self.lr,
                        beta1=self.config["adam_beta_1"],
                        beta2=self.config["adam_beta_2"],
                        epsilon=self.config["adam_epsilon"],
                    )

        # Initializer
        with tf.variable_scope("initializer") as scope:
            self.init_global_op = tf.variables_initializer(tf.global_variables())
            self.init_local_op = tf.variables_initializer(tf.local_variables())

        # Saver
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)
        self.best_saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)

        # Initialize and restore
        self.run([self.init_global_op, self.init_local_op])
        is_restore = False
        if use_best and os.path.exists(self.best_ckpt_path) and os.path.isfile(os.path.join(self.best_ckpt_path, "checkpoint")):
            latest_checkpoint = tf.train.latest_checkpoint(self.best_ckpt_path)
            self.saver.restore(self.sess, latest_checkpoint)
            logger.info("Best model restored from {}".format(latest_checkpoint))
            is_restore = True
        elif os.path.exists(self.checkpoint_path) and os.path.isfile(os.path.join(self.checkpoint_path, "checkpoint")):
            latest_checkpoint = tf.train.latest_checkpoint(self.checkpoint_path)
            self.saver.restore(self.sess, latest_checkpoint)
            logger.info("Model restored from {}".format(latest_checkpoint))
            is_restore = True
        if not is_restore:
            logger.info("Model started from random weights")

    def build_cnn(self):
        first_filter_size = int(self.config["sampling_rate"] / 2.0)
        first_filter_stride = int(self.config["sampling_rate"] / 16.0)

        with tf.variable_scope("cnn") as scope:
            net = nn.conv1d("conv1d_1", self.signals, 128, first_filter_size, first_filter_stride)
            net = nn.batch_norm("bn_1", net, self.is_training)
            net = tf.nn.relu(net, name="relu_1")
            net = nn.max_pool1d("maxpool1d_1", net, 8, 8)
            net = tf.layers.dropout(net, rate=0.5, training=self.is_training, name="drop_1")

            net = nn.conv1d("conv1d_2_1", net, 128, 8, 1)
            net = nn.batch_norm("bn_2_1", net, self.is_training)
            net = tf.nn.relu(net, name="relu_2_1")
            net = nn.conv1d("conv1d_2_2", net, 128, 8, 1)
            net = nn.batch_norm("bn_2_2", net, self.is_training)
            net = tf.nn.relu(net, name="relu_2_2")
            net = nn.conv1d("conv1d_2_3", net, 128, 8, 1)
            net = nn.batch_norm("bn_2_3", net, self.is_training)
            net = tf.nn.relu(net, name="relu_2_3")

            # Дополнительные слои для компенсации отсутствия RNN
            net = nn.conv1d("conv1d_3_1", net, 256, 8, 1)
            net = nn.batch_norm("bn_3_1", net, self.is_training)
            net = tf.nn.relu(net, name="relu_3_1")
            net = nn.conv1d("conv1d_3_2", net, 256, 8, 1)
            net = nn.batch_norm("bn_3_2", net, self.is_training)
            net = tf.nn.relu(net, name="relu_3_2")

            net = nn.max_pool1d("maxpool1d_2", net, 4, 4)
            net = tf.layers.flatten(net, name="flatten_2")
        

            # Дополнительный полносвязный слой
            net = nn.fc("fc_1", net, 512)
            net = tf.nn.relu(net, name="relu_fc_1")
            net = tf.layers.dropout(net, rate=0.5, training=self.is_training, name="drop_2")

        return net

    def regularization_loss(self):
        reg_losses = []
        list_vars = [
            "cnn/conv1d_1/conv2d/kernel:0",
            "cnn/conv1d_2_1/conv2d/kernel:0",
            "cnn/conv1d_2_2/conv2d/kernel:0",
            "cnn/conv1d_2_3/conv2d/kernel:0",
            "cnn/conv1d_3_1/conv2d/kernel:0",
            "cnn/conv1d_3_2/conv2d/kernel:0",
        ]
        for v in tf.trainable_variables():
            if any(v.name in s for s in list_vars):
                reg_losses.append(tf.nn.l2_loss(v))
        if len(reg_losses):
            reg_losses = tf.multiply(tf.add_n(reg_losses, name="l2_loss"), self.config["l2_weight_decay"])
        else:
            reg_losses = 0
        return reg_losses

    # def train(self, minibatches):
    #     self.run(self.metric_init_op)
    #     start = timeit.default_timer()
    #     preds = []
    #     trues = []
    #     for x, y, *_ in minibatches:
    #         feed_dict = {
    #             self.signals: x,
    #             self.labels: y,
    #             self.is_training: True,
    #         }
    #         _, outputs = self.run([self.train_step_op, self.train_outputs], feed_dict=feed_dict)
    #         preds.extend(outputs["train/preds"])
    #         trues.extend(y)
    #     acc = skmetrics.accuracy_score(y_true=trues, y_pred=preds)
    #     f1_score = skmetrics.f1_score(y_true=trues, y_pred=preds, average="macro")
    #     cm = skmetrics.confusion_matrix(y_true=trues, y_pred=preds, labels=[0,1,2,3,4])
    #     stop = timeit.default_timer()
    #     duration = stop - start
    #     outputs.update({
    #         "train/trues": trues,
    #         "train/preds": preds,
    #         "train/accuracy": acc,
    #         "train/f1_score": f1_score,
    #         "train/cm": cm,
    #         "train/duration": duration,
    #     })
    #     return outputs

    def train(self, minibatches):
        self.run(self.metric_init_op)
        start = timeit.default_timer()
        preds = []
        trues = []
        for x, y, *_ in minibatches:
            feed_dict = {
                self.signals: x,
                self.labels: y,
                self.is_training: True,
                self.class_weights: self.config["class_weights"],
            }
            _, outputs = self.run([self.train_step_op, self.train_outputs], feed_dict=feed_dict)
            preds.extend(outputs["train/preds"])
            trues.extend(y)
        acc = skmetrics.accuracy_score(y_true=trues, y_pred=preds)
        f1_score = skmetrics.f1_score(y_true=trues, y_pred=preds, average="macro")
        cm = skmetrics.confusion_matrix(y_true=trues, y_pred=preds, labels=[0,1,2,3,4])
        logger.info("Train Confusion Matrix:\n{}".format(cm))
        stop = timeit.default_timer()
        duration = stop - start
        outputs.update({
            "train/trues": trues,
            "train/preds": preds,
            "train/accuracy": acc,
            "train/f1_score": f1_score,
            "train/cm": cm,
            "train/duration": duration,
        })
        return outputs


    # def evaluate(self, minibatches):
    #     start = timeit.default_timer()
    #     losses = []
    #     preds = []
    #     trues = []
    #     for x, y, *_ in minibatches:
    #         feed_dict = {
    #             self.signals: x,
    #             self.labels: y,
    #             self.is_training: False,
    #         }
    #         outputs = self.run(self.test_outputs, feed_dict=feed_dict)
    #         losses.append(outputs["test/loss"])
    #         preds.extend(outputs["test/preds"])
    #         trues.extend(y)
    #     loss = np.mean(losses)
    #     acc = skmetrics.accuracy_score(y_true=trues, y_pred=preds)
    #     f1_score = skmetrics.f1_score(y_true=trues, y_pred=preds, average="macro")
    #     cm = skmetrics.confusion_matrix(y_true=trues, y_pred=preds, labels=[0,1,2,3,4])
    #     stop = timeit.default_timer()
    #     duration = stop - start
    #     outputs = {
    #         "test/trues": trues,
    #         "test/preds": preds,
    #         "test/loss": loss,
    #         "test/accuracy": acc,
    #         "test/f1_score": f1_score,
    #         "test/cm": cm,
    #         "test/duration": duration,
    #     }
    #     return outputs

    def evaluate(self, minibatches):
        start = timeit.default_timer()
        losses = []
        preds = []
        trues = []
        for x, y, *_ in minibatches:
            feed_dict = {
                self.signals: x,
                self.labels: y,
                self.is_training: False,
                self.class_weights: self.config["class_weights"],
            }
            outputs = self.run(self.test_outputs, feed_dict=feed_dict)
            losses.append(outputs["test/loss"])
            preds.extend(outputs["test/preds"])
            trues.extend(y)
        loss = np.mean(losses)
        acc = skmetrics.accuracy_score(y_true=trues, y_pred=preds)
        f1_score = skmetrics.f1_score(y_true=trues, y_pred=preds, average="macro")
        cm = skmetrics.confusion_matrix(y_true=trues, y_pred=preds, labels=[0,1,2,3,4])
        logger.info("Eval Confusion Matrix:\n{}".format(cm))
        stop = timeit.default_timer()
        duration = stop - start
        outputs = {
            "test/trues": trues,
            "test/preds": preds,
            "test/loss": loss,
            "test/accuracy": acc,
            "test/f1_score": f1_score,
            "test/cm": cm,
            "test/duration": duration,
        }
        return outputs


    def get_current_epoch(self):
        return self.run(self.global_epoch)

    def pass_one_epoch(self):
        self.run(tf.assign(self.global_epoch, self.global_epoch+1))

    def run(self, *args, **kwargs):
        return self.sess.run(*args, **kwargs)

    def save_checkpoint(self, name):
        path = self.saver.save(self.sess, os.path.join(self.checkpoint_path, "{}.ckpt".format(name)), global_step=self.global_step)
        logger.info("Saved checkpoint to {}".format(path))

    def save_best_checkpoint(self, name):
        path = self.best_saver.save(self.sess, os.path.join(self.best_ckpt_path, "{}.ckpt".format(name)), global_step=self.global_step)
        logger.info("Saved best checkpoint to {}".format(path))

    def save_weights(self, scope, name, key_variables=tf.GraphKeys.TRAINABLE_VARIABLES):
        path = os.path.join(self.weights_path, "{}.npz".format(name))
        logger.info("Saving weights in scope: {} to {}".format(scope, path))
        save_dict = {}
        cnn_vars = tf.get_collection(key_variables, scope=scope)
        for v in cnn_vars:
            save_dict[v.name] = self.sess.run(v)
            logger.info("  variable: {}".format(v.name))
        if not os.path.exists(self.weights_path):
            os.makedirs(self.weights_path)
        np.savez(path, **save_dict)

    def load_weights(self, scope, weight_file, key_variables=tf.GraphKeys.TRAINABLE_VARIABLES):
        logger.info("Loading weights in scope: {} from {}".format(scope, weight_file))
        cnn_vars = tf.get_collection(key_variables, scope=scope)
        with np.load(weight_file) as f:
            for v in cnn_vars:
                tensor = tf.get_default_graph().get_tensor_by_name(v.name)
                self.run(tf.assign(tensor, f[v.name]))
                logger.info("  variable: {}".format(v.name))
