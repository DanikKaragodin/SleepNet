# import os
# import numpy as np
# import sklearn.metrics as skmetrics
# import tensorflow as tf
# import timeit
# import tensorflow.contrib.metrics as contrib_metrics
# import tensorflow.contrib.slim as contrib_slim
# import nn
# import logging

# logger = logging.getLogger("default_log")

# class TinySleepNetRNN(object):
#     def __init__(self, config, output_dir="./output", testing=False, use_best=False):
#         self.config = config
#         self.output_dir = output_dir
#         self.checkpoint_path = os.path.join(self.output_dir, "checkpoint")
#         self.best_ckpt_path = os.path.join(self.output_dir, "best_ckpt")
#         self.weights_path = os.path.join(self.output_dir, "weights")
#         self.log_dir = os.path.join(self.output_dir, "log")

#         # Placeholders
#         with tf.variable_scope("placeholders") as scope:
#             self.signals = tf.placeholder(dtype=tf.float32, shape=(None, self.config["input_size"], 1, 1), name='signals')
#             self.labels = tf.placeholder(dtype=tf.int32, shape=(None,), name='labels')
#             self.is_training = tf.placeholder(dtype=tf.bool, shape=(), name='is_training')
#             self.loss_weights = tf.placeholder(dtype=tf.float32, shape=(None,), name='loss_weights')
#             self.seq_lengths = tf.placeholder(dtype=tf.int32, shape=(None,), name='seq_lengths')

#         # Global step and epoch
#         self.global_step = tf.Variable(0, trainable=False, name='global_step')
#         self.global_epoch = tf.Variable(0, trainable=False, name='global_epoch')

#         # Build RNN
#         net = self.append_rnn(self.signals)

#         # Softmax linear
#         net = nn.fc("softmax_linear", net, self.config["n_classes"], bias=0.0)

#         # Outputs
#         self.logits = net
#         self.preds = tf.argmax(self.logits, axis=1)

#         # Loss



#         # В __init__
#         self.loss_per_sample = self.focal_loss(self.logits, self.labels, gamma=2.0, alpha=0.25)
#         loss_w_seq = tf.multiply(self.loss_weights, self.loss_per_sample)
#         sample_weights = tf.reduce_sum(
#             tf.multiply(
#                 tf.one_hot(indices=self.labels, depth=self.config["n_classes"]),
#                 np.asarray(self.config["class_weights"], dtype=np.float32)
#             ), 1
#         )
#         loss_w_class = tf.multiply(loss_w_seq, sample_weights)
#         self.loss_ce = tf.reduce_sum(loss_w_class) / tf.reduce_sum(self.loss_weights)

#         # self.loss_per_sample = tf.nn.sparse_softmax_cross_entropy_with_logits(
#         #     labels=self.labels, logits=self.logits, name="loss_ce_per_sample")
#         # loss_w_seq = tf.multiply(self.loss_weights, self.loss_per_sample)
#         # sample_weights = tf.reduce_sum(
#         #     tf.multiply(
#         #         tf.one_hot(indices=self.labels, depth=self.config["n_classes"]),
#         #         np.asarray(self.config["class_weights"], dtype=np.float32)
#         #     ), 1
#         # )
#         # loss_w_class = tf.multiply(loss_w_seq, sample_weights)
#         # self.loss_ce = tf.reduce_sum(loss_w_class) / tf.reduce_sum(self.loss_weights)

#         self.reg_losses = self.regularization_loss()
#         self.loss = self.loss_ce + self.reg_losses

#         # Metrics
#         with tf.variable_scope("stream_metrics") as scope:
#             self.metric_value_op, self.metric_update_op = contrib_metrics.aggregate_metric_map({
#                 "loss": tf.metrics.mean(values=self.loss),
#                 "accuracy": tf.metrics.accuracy(labels=self.labels, predictions=self.preds),
#                 "precision": tf.metrics.precision(labels=self.labels, predictions=self.preds),
#                 "recall": tf.metrics.recall(labels=self.labels, predictions=self.preds),
#             })
#             metric_vars = contrib_slim.get_local_variables(scope=scope.name)
#             self.metric_init_op = tf.variables_initializer(metric_vars)

#         # Training and test outputs
#         self.train_outputs = {
#             "global_step": self.global_step,
#             "train/loss": self.loss,
#             "train/preds": self.preds,
#             "train/stream_metrics": self.metric_update_op,
#             "train/init_state": self.init_state,
#             "train/final_state": self.final_state,
#         }
#         self.test_outputs = {
#             "global_step": self.global_step,
#             "test/loss": self.loss,
#             "test/preds": self.preds,
#             "test/init_state": self.init_state,
#             "test/final_state": self.final_state,
#         }

#         # Session
#         config_sess = tf.ConfigProto()
#         config_sess.gpu_options.allow_growth = True
#         self.sess = tf.Session(config=config_sess)
#         if not testing:
#             self.train_writer = tf.summary.FileWriter(os.path.join(self.log_dir, "train"))
#             self.train_writer.add_graph(self.sess.graph)
#             logger.info("Saved tensorboard graph to {}".format(self.train_writer.get_logdir()))

#         # Optimizer
#         if not testing:
#             # self.lr = tf.constant(self.config["learning_rate"], dtype=tf.float32)
#             self.lr = tf.train.exponential_decay(
#                 learning_rate=self.config["learning_rate"],
#                 global_step=self.global_step,
#                 decay_steps=1000,
#                 decay_rate=0.96,
#                 staircase=True,
#                 name="learning_rate"
#             )
#             with tf.variable_scope("optimizer") as scope:
#                 update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
#                 with tf.control_dependencies(update_ops):
#                     self.train_step_op, self.grad_op = nn.adam_optimizer_clip(
#                         loss=self.loss,
#                         training_variables=tf.trainable_variables(),
#                         global_step=self.global_step,
#                         learning_rate=self.lr,
#                         beta1=self.config["adam_beta_1"],
#                         beta2=self.config["adam_beta_2"],
#                         epsilon=self.config["adam_epsilon"],
#                         clip_value=self.config["clip_grad_value"],
#                     )

#         # Initializer
#         with tf.variable_scope("initializer") as scope:
#             self.init_global_op = tf.variables_initializer(tf.global_variables())
#             self.init_local_op = tf.variables_initializer(tf.local_variables())

#         # Saver
#         self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)
#         self.best_saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)

#         # Initialize and restore
#         self.run([self.init_global_op, self.init_local_op])
#         is_restore = False
#         if use_best and os.path.exists(self.best_ckpt_path) and os.path.isfile(os.path.join(self.best_ckpt_path, "checkpoint")):
#             latest_checkpoint = tf.train.latest_checkpoint(self.best_ckpt_path)
#             self.saver.restore(self.sess, latest_checkpoint)
#             logger.info("Best model restored from {}".format(latest_checkpoint))
#             is_restore = True
#         elif os.path.exists(self.checkpoint_path) and os.path.isfile(os.path.join(self.checkpoint_path, "checkpoint")):
#             latest_checkpoint = tf.train.latest_checkpoint(self.checkpoint_path)
#             self.saver.restore(self.sess, latest_checkpoint)
#             logger.info("Model restored from {}".format(latest_checkpoint))
#             is_restore = True
#         if not is_restore:
#             logger.info("Model started from random weights")

#     def focal_loss(self, logits, labels, gamma=2.0, alpha=0.25):
#         ce_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
#         pt = tf.exp(-ce_loss)
#         focal_loss = alpha * (1 - pt) ** gamma * ce_loss
#         return focal_loss


#     def append_rnn(self, inputs):
#         with tf.variable_scope("rnn") as scope:
#             # Убираем дополнительные размерности (1, 1)
#             inputs = tf.squeeze(inputs, axis=[2, 3])  # Преобразуем (None, 3000, 1, 1) в (None, 3000)
            
#             # Определяем input_dim (длина сигнала = 3000)
#             input_dim = inputs.shape[-1].value  # 3000
            
#             # Вычисляем batch_size динамически
#             batch_size = tf.shape(inputs)[0] // self.config["seq_length"]
            
#             # Преобразуем вход в форму (batch_size, seq_length, input_dim)
#             seq_inputs = tf.reshape(
#                 inputs,
#                 shape=[batch_size, self.config["seq_length"], input_dim],
#                 name="reshape_seq_inputs"
#             )

#             def _create_rnn_cell(n_units):
#                 cell = tf.contrib.rnn.LSTMCell(
#                     num_units=n_units,
#                     use_peepholes=True,
#                     forget_bias=1.0,
#                     state_is_tuple=True,
#                 )
#                 keep_prob = tf.cond(self.is_training, lambda: tf.constant(0.7), lambda: tf.constant(1.0))
#                 cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)
#                 return cell

#             # Увеличенная RNN
#             cells = []
#             for l in range(4):  # Увеличено до 4 слоев
#                 cells.append(_create_rnn_cell(256))  # Увеличено до 256 юнитов

#             multi_cell = tf.nn.rnn_cell.MultiRNNCell(cells=cells, state_is_tuple=True)
#             self.init_state = multi_cell.zero_state(batch_size, tf.float32)  # Используем динамический batch_size

#             outputs, states = tf.nn.dynamic_rnn(
#                 cell=multi_cell,
#                 inputs=seq_inputs,
#                 initial_state=self.init_state,
#                 sequence_length=self.seq_lengths,
#             )
#             self.final_state = states

#             net = tf.reshape(outputs, shape=[-1, 256], name="reshape_nonseq_input")
#             net = nn.fc("fc_1", net, 512)
#             net = tf.nn.relu(net, name="relu_fc_1")
#             net = tf.layers.dropout(net, rate=0.3, training=self.is_training, name="drop_fc_1")

#         return net

#     def regularization_loss(self):
#         reg_losses = []
#         list_vars = [
#             # "rnn/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel:0",
#             # "rnn/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel:0",
#             # "rnn/rnn/multi_rnn_cell/cell_2/lstm_cell/kernel:0",
#             # "rnn/rnn/multi_rnn_cell/cell_3/lstm_cell/kernel:0",
#             "rnn/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel:0",
#             "rnn/rnn/multi_rnn_cell/cell_0/lstm_cell/bias:0",
#             "rnn/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel:0",
#             "rnn/rnn/multi_rnn_cell/cell_1/lstm_cell/bias:0",
#             "rnn/rnn/multi_rnn_cell/cell_2/lstm_cell/kernel:0",
#             "rnn/rnn/multi_rnn_cell/cell_2/lstm_cell/bias:0",
#             "rnn/rnn/multi_rnn_cell/cell_3/lstm_cell/kernel:0",
#             "rnn/rnn/multi_rnn_cell/cell_3/lstm_cell/bias:0",
#             "fc_1/dense/kernel:0",
#             "fc_1/dense/bias:0",
#             "softmax_linear/dense/kernel:0",
#             "softmax_linear/dense/bias:0",
#         ]
#         for v in tf.trainable_variables():
#             if any(v.name in s for s in list_vars):
#                 reg_losses.append(tf.nn.l2_loss(v))
#         if len(reg_losses):
#             reg_losses = tf.multiply(tf.add_n(reg_losses, name="l2_loss"), self.config["l2_weight_decay"])
#         else:
#             reg_losses = 0
#         return reg_losses

#     def train(self, minibatches):
#         self.run(self.metric_init_op)
#         start = timeit.default_timer()
#         preds = []
#         trues = []
#         for x, y, w, sl, re in minibatches:
#             feed_dict = {
#                 self.signals: x,
#                 self.labels: y,
#                 self.is_training: True,
#                 self.loss_weights: w,
#                 self.seq_lengths: sl,
#             }
#             if re:
#                 # Используем фиктивный вход для инициализации состояния
#                 dummy_signals = np.zeros(
#                     (self.config["batch_size"] * self.config["seq_length"], self.config["input_size"], 1, 1),
#                     dtype=np.float32
#                 )
#                 state = self.run(self.init_state, feed_dict={self.signals: dummy_signals})

#             # Передаем состояния между батчами
#             for i, (c, h) in enumerate(self.init_state):
#                 feed_dict[c] = state[i].c
#                 feed_dict[h] = state[i].h

#             _, outputs = self.run([self.train_step_op, self.train_outputs], feed_dict=feed_dict)

#             # Сохраняем финальные состояния
#             state = outputs["train/final_state"]

#             tmp_preds = np.reshape(outputs["train/preds"], (self.config["batch_size"], self.config["seq_length"]))
#             tmp_trues = np.reshape(y, (self.config["batch_size"], self.config["seq_length"]))

#             for i in range(self.config["batch_size"]):
#                 preds.extend(tmp_preds[i, :sl[i]])
#                 trues.extend(tmp_trues[i, :sl[i]])

#         acc = skmetrics.accuracy_score(y_true=trues, y_pred=preds)
#         f1_score = skmetrics.f1_score(y_true=trues, y_pred=preds, average="macro")
#         cm = skmetrics.confusion_matrix(y_true=trues, y_pred=preds, labels=[0,1,2,3,4])
#         logger.info("Train Confusion Matrix:\n{}".format(cm))
#         stop = timeit.default_timer()
#         duration = stop - start
#         outputs.update({
#             "train/trues": trues,
#             "train/preds": preds,
#             "train/accuracy": acc,
#             "train/f1_score": f1_score,
#             "train/cm": cm,
#             "train/duration": duration,
#         })
#         return outputs


#     def evaluate(self, minibatches):
#         start = timeit.default_timer()
#         losses = []
#         preds = []
#         trues = []
#         for x, y, w, sl, re in minibatches:
#             # logger.info(f"Batch x shape: {x.shape}, Seq lengths: {sl}")
#             feed_dict = {
#                 self.signals: x,
#                 self.labels: y,
#                 self.is_training: False,
#                 self.loss_weights: w,
#                 self.seq_lengths: sl,
#             }
#             if re:
#                 # Используем фиктивный вход для инициализации состояния
#                 dummy_signals = np.zeros(
#                     (self.config["batch_size"] * self.config["seq_length"], self.config["input_size"], 1, 1),
#                     dtype=np.float32
#                 )
#                 # logger.info(f"Dummy signals shape: {dummy_signals.shape}")
#                 state = self.run(self.init_state, feed_dict={self.signals: dummy_signals})
#             for i, (c, h) in enumerate(self.init_state):
#                 feed_dict[c] = state[i].c
#                 feed_dict[h] = state[i].h
#             outputs = self.run(self.test_outputs, feed_dict=feed_dict)
#             state = outputs["test/final_state"]
#             losses.append(outputs["test/loss"])
#             tmp_preds = np.reshape(outputs["test/preds"], (self.config["batch_size"], self.config["seq_length"]))
#             tmp_trues = np.reshape(y, (self.config["batch_size"], self.config["seq_length"]))
#             for i in range(self.config["batch_size"]):
#                 preds.extend(tmp_preds[i, :sl[i]])
#                 trues.extend(tmp_trues[i, :sl[i]])
#         loss = np.mean(losses)
#         acc = skmetrics.accuracy_score(y_true=trues, y_pred=preds)
#         f1_score = skmetrics.f1_score(y_true=trues, y_pred=preds, average="macro")
#         cm = skmetrics.confusion_matrix(y_true=trues, y_pred=preds, labels=[0,1,2,3,4])
#         logger.info("Validate or Test Confusion Matrix:\n{}".format(cm))
#         stop = timeit.default_timer()
#         duration = stop - start
#         outputs = {
#             "test/trues": trues,
#             "test/preds": preds,
#             "test/loss": loss,
#             "test/accuracy": acc,
#             "test/f1_score": f1_score,
#             "test/cm": cm,
#             "test/duration": duration,
#         }
#         return outputs

#     def get_current_epoch(self):
#         return self.run(self.global_epoch)

#     def pass_one_epoch(self):
#         self.run(tf.assign(self.global_epoch, self.global_epoch+1))

#     def run(self, *args, **kwargs):
#         return self.sess.run(*args, **kwargs)

#     def save_checkpoint(self, name):
#         path = self.saver.save(self.sess, os.path.join(self.checkpoint_path, "{}.ckpt".format(name)), global_step=self.global_step)
#         logger.info("Saved checkpoint to {}".format(path))

#     def save_best_checkpoint(self, name):
#         path = self.best_saver.save(self.sess, os.path.join(self.best_ckpt_path, "{}.ckpt".format(name)), global_step=self.global_step)
#         logger.info("Saved best checkpoint to {}".format(path))

#     def save_weights(self, scope, name, key_variables=tf.GraphKeys.TRAINABLE_VARIABLES):
#         path = os.path.join(self.weights_path, "{}.npz".format(name))
#         logger.info("Saving weights in scope: {} to {}".format(scope, path))
#         save_dict = {}
#         cnn_vars = tf.get_collection(key_variables, scope=scope)
#         for v in cnn_vars:
#             save_dict[v.name] = self.sess.run(v)
#             logger.info("  variable: {}".format(v.name))
#         if not os.path.exists(self.weights_path):
#             os.makedirs(self.weights_path)
#         np.savez(path, **save_dict)

#     def load_weights(self, scope, weight_file, key_variables=tf.GraphKeys.TRAINABLE_VARIABLES):
#         logger.info("Loading weights in scope: {} from {}".format(scope, weight_file))
#         cnn_vars = tf.get_collection(key_variables, scope=scope)
#         with np.load(weight_file) as f:
#             for v in cnn_vars:
#                 tensor = tf.get_default_graph().get_tensor_by_name(v.name)
#                 self.run(tf.assign(tensor, f[v.name]))
#                 logger.info("  variable: {}".format(v.name))


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

def sparse_categorical_focal_loss(y_true, y_pred, gamma=2.0, alpha=None):
    y_true = tf.cast(y_true, tf.int64)
    y_pred = tf.cast(y_pred, tf.float32)
    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
    p_t = tf.exp(-ce)
    if alpha is not None:
        alpha = tf.gather(alpha, y_true)
        focal_loss = alpha * tf.pow(1 - p_t, gamma) * ce
    else:
        focal_loss = tf.pow(1 - p_t, gamma) * ce
    return tf.reduce_mean(focal_loss)

class TinySleepNetRNN(object):
    def __init__(self, config, output_dir="./output", testing=False, use_best=False):
        self.config = config
        self.output_dir = output_dir
        self.checkpoint_path = os.path.join(self.output_dir, "checkpoint")
        self.best_ckpt_path = os.path.join(self.output_dir, "best_ckpt")
        self.weights_path = os.path.join(self.output_dir, "weights")
        self.log_dir = os.path.join(self.output_dir, "log")

        # Placeholders
        with tf.variable_scope("placeholders") as scope:
            self.signals = tf.placeholder(dtype=tf.float32, shape=(None, self.config["input_size"], 1, 1), name='signals')
            self.labels = tf.placeholder(dtype=tf.int32, shape=(None,), name='labels')
            self.is_training = tf.placeholder(dtype=tf.bool, shape=(), name='is_training')
            self.loss_weights = tf.placeholder(dtype=tf.float32, shape=(None,), name='loss_weights')
            self.seq_lengths = tf.placeholder(dtype=tf.int32, shape=(None,), name='seq_lengths')

        # Global step and epoch
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.global_epoch = tf.Variable(0, trainable=False, name='global_epoch')

        # Build RNN
        net = self.append_rnn(self.signals)

        # Softmax linear
        net = nn.fc("softmax_linear", net, self.config["n_classes"], bias=0.0)

        # Outputs
        self.logits = net
        self.preds = tf.argmax(self.logits, axis=1)

        # Loss
        self.loss_per_sample = sparse_categorical_focal_loss(
            self.labels, self.logits, gamma=3.0, alpha=self.config["class_weights"]
        )
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
            self.lr = tf.train.exponential_decay(
                learning_rate=self.config["learning_rate"],
                global_step=self.global_step,
                decay_steps=1000,
                decay_rate=0.96,
                staircase=True,
                name="learning_rate"
            )
            with tf.variable_scope("optimizer") as scope:
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(update_ops):
                    self.train_step_op, self.grad_op = nn.adam_optimizer_clip(
                        loss=self.loss,
                        training_variables=tf.trainable_variables(),
                        global_step=self.global_step,
                        learning_rate=self.lr,
                        beta1=self.config["adam_beta_1"],
                        beta2=self.config["adam_beta_2"],
                        epsilon=self.config["adam_epsilon"],
                        clip_value=self.config["clip_grad_value"],
                    )
            # Замените блок с Adam на это:
            # with tf.variable_scope("optimizer") as scope:
            #     update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            #     with tf.control_dependencies(update_ops):
            #         # Nadam: Adam + Nesterov momentum
            #         self.train_step_op = tf.train.AdamOptimizer(
            #             learning_rate=self.lr,
            #             beta1=0.9,  # Рекомендуется для Nadam
            #             beta2=0.999,
            #             epsilon=self.config["adam_epsilon"]
            #         ).minimize(
            #             self.loss,
            #             global_step=self.global_step
            #         )
            #         # Если нужно добавить клиппинг градиентов:
            #         # gradients, variables = zip(*optimizer.compute_gradients(self.loss))
            #         # gradients, _ = tf.clip_by_global_norm(gradients, self.config["clip_grad_value"])
            #         # self.train_step_op = optimizer.apply_gradients(zip(gradients, variables))
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

    # def append_rnn(self, inputs):
    #     with tf.variable_scope("rnn") as scope:
    #         # Убираем дополнительные размерности (1, 1)
    #         inputs = tf.squeeze(inputs, axis=[2, 3])  # Преобразуем (None, 3000, 1, 1) в (None, 3000)
            
    #         # Определяем input_dim (длина сигнала = 3000)
    #         input_dim = inputs.shape[-1].value  # 3000
            
    #         # Вычисляем batch_size динамически
    #         batch_size = tf.shape(inputs)[0] // self.config["seq_length"]
            
    #         # Преобразуем вход в форму (batch_size, seq_length, input_dim)
    #         seq_inputs = tf.reshape(
    #             inputs,
    #             shape=[batch_size, self.config["seq_length"], input_dim],
    #             name="reshape_seq_inputs"
    #         )

    #         # Создаём двунаправленные LSTM
    #         fw_cell = tf.nn.rnn_cell.LSTMCell(num_units=256, state_is_tuple=True)
    #         bw_cell = tf.nn.rnn_cell.LSTMCell(num_units=256, state_is_tuple=True)
            
    #         # Вызываем bidirectional_dynamic_rnn
    #         outputs, output_states = tf.nn.bidirectional_dynamic_rnn(
    #             fw_cell,
    #             bw_cell,
    #             seq_inputs,
    #             sequence_length=self.seq_lengths,
    #             dtype=tf.float32
    #         )
            
    #         # Объединяем выходы прямого и обратного проходов
    #         outputs = tf.concat(outputs, 2)  # shape (batch_size, seq_length, 512)
            
    #         # Преобразуем для FC-слоя
    #         net = tf.reshape(outputs, [-1, 512], name="reshape_nonseq_input")
    #         net = nn.fc("fc_1", net, 512)
    #         net = tf.nn.relu(net, name="relu_fc_1")
    #         net = tf.layers.dropout(net, rate=0.6, training=self.is_training, name="drop_fc_1")

    #     return net

    def append_rnn(self, inputs):
        with tf.variable_scope("rnn") as scope:
            inputs = tf.squeeze(inputs, axis=[2, 3])
            input_dim = inputs.shape[-1].value
            batch_size = tf.shape(inputs)[0] // self.config["seq_length"]
            seq_inputs = tf.reshape(inputs, shape=[batch_size, self.config["seq_length"], input_dim])

            def create_bilstm_cell():
                fw_cell = tf.nn.rnn_cell.LSTMCell(num_units=256, state_is_tuple=True)
                bw_cell = tf.nn.rnn_cell.LSTMCell(num_units=256, state_is_tuple=True)
                keep_prob = tf.cond(self.is_training, lambda: 0.75, lambda: 1.0)
                fw_cell = tf.nn.rnn_cell.DropoutWrapper(fw_cell, input_keep_prob=keep_prob, output_keep_prob=keep_prob)
                bw_cell = tf.nn.rnn_cell.DropoutWrapper(bw_cell, input_keep_prob=keep_prob, output_keep_prob=keep_prob)
                return fw_cell, bw_cell

            # Создаем 2 слоя
            cells = [create_bilstm_cell() for _ in range(3)]
            outputs = seq_inputs
            for i, (fw_cell, bw_cell) in enumerate(cells):
                outputs, _ = tf.nn.bidirectional_dynamic_rnn(
                    fw_cell, bw_cell, outputs, sequence_length=self.seq_lengths, dtype=tf.float32, scope=f"bilstm_{i}"
                )
                outputs = tf.concat(outputs, 2)

            net = tf.reshape(outputs, [-1, 512], name="reshape_nonseq_input")
            net = nn.fc("fc_1", net, 512)
            net = tf.nn.relu(net, name="relu_fc_1")
            net = tf.layers.dropout(net, rate=0.5, training=self.is_training, name="drop_fc_1")
            return net

    # def append_rnn(self, inputs):
    #     with tf.variable_scope("rnn") as scope:
    #         inputs = tf.squeeze(inputs, axis=[2, 3])
    #         input_dim = inputs.shape[-1].value
    #         inputs = tf.layers.batch_normalization(inputs, training=self.is_training)
    #         batch_size = tf.shape(inputs)[0] // self.config["seq_length"]
    #         seq_inputs = tf.reshape(inputs, shape=[batch_size, self.config["seq_length"], input_dim])

    #         # 2 слоя двунаправленных GRU
    #         for i in range(3):
    #             with tf.variable_scope(f"bilstm_{i}"):
    #                 # Прямой и обратный GRU
    #                 fw_cell = tf.nn.rnn_cell.GRUCell(256)
    #                 bw_cell = tf.nn.rnn_cell.GRUCell(256)
                    
    #                 # Дропаут
    #                 fw_cell = tf.nn.rnn_cell.DropoutWrapper(fw_cell, output_keep_prob=0.8)
    #                 bw_cell = tf.nn.rnn_cell.DropoutWrapper(bw_cell, output_keep_prob=0.8)
                    
    #                 outputs, _ = tf.nn.bidirectional_dynamic_rnn(
    #                     fw_cell, 
    #                     bw_cell,
    #                     seq_inputs,
    #                     dtype=tf.float32
    #                 )
    #                 # Конкатенация прямого и обратного выходов [batch, seq, 512]
    #                 seq_inputs = tf.concat(outputs, axis=-1)

    #         # Self-Attention
    #         with tf.variable_scope("attention"):
    #             attention_logits = tf.matmul(seq_inputs, seq_inputs, transpose_b=True)
    #             attention_weights = tf.nn.softmax(attention_logits, axis=-1)
    #             attention_output = tf.matmul(attention_weights, seq_inputs)
                
    #         outputs = tf.concat([seq_inputs, attention_output], axis=-1)  # [batch, seq, 1024]

    #         net = tf.reshape(outputs, [-1, 1024])
    #         net = nn.fc("fc_1", net, 512)
    #         net = tf.nn.relu(net)
    #         net = tf.layers.dropout(net, rate=0.3, training=self.is_training, name="drop_2")
    #         net = nn.fc("fc_2", net, 256)
    #         net = tf.nn.relu(net)
    #         return net

    def regularization_loss(self):
        reg_losses = []
        list_vars = [
            "rnn/bilstm_0/forward/lstm_cell/kernel:0",
            "rnn/bilstm_0/forward/lstm_cell/bias:0",
            "rnn/bilstm_0/backward/lstm_cell/kernel:0",
            "rnn/bilstm_0/backward/lstm_cell/bias:0",
            "rnn/bilstm_1/forward/lstm_cell/kernel:0",
            "rnn/bilstm_1/forward/lstm_cell/bias:0",
            "rnn/bilstm_1/backward/lstm_cell/kernel:0",
            "rnn/bilstm_1/backward/lstm_cell/bias:0",
            "fc_1/dense/kernel:0",
            "fc_1/dense/bias:0",
            "softmax_linear/dense/kernel:0",
            "softmax_linear/dense/bias:0",

            # # Первый слой BiGRU
            # "rnn/bilstm_0/bidirectional_rnn/fw/gru_cell/gates/kernel:0",
            # "rnn/bilstm_0/bidirectional_rnn/fw/gru_cell/gates/bias:0",
            # "rnn/bilstm_0/bidirectional_rnn/fw/gru_cell/candidate/kernel:0",
            # "rnn/bilstm_0/bidirectional_rnn/fw/gru_cell/candidate/bias:0",
            # "rnn/bilstm_0/bidirectional_rnn/bw/gru_cell/gates/kernel:0",
            # "rnn/bilstm_0/bidirectional_rnn/bw/gru_cell/gates/bias:0",
            # "rnn/bilstm_0/bidirectional_rnn/bw/gru_cell/candidate/kernel:0",
            # "rnn/bilstm_0/bidirectional_rnn/bw/gru_cell/candidate/bias:0",
            
            # # Второй слой BiGRU
            # "rnn/bilstm_1/bidirectional_rnn/fw/gru_cell/gates/kernel:0",
            # "rnn/bilstm_1/bidirectional_rnn/fw/gru_cell/gates/bias:0",
            # "rnn/bilstm_1/bidirectional_rnn/fw/gru_cell/candidate/kernel:0",
            # "rnn/bilstm_1/bidirectional_rnn/fw/gru_cell/candidate/bias:0",
            # "rnn/bilstm_1/bidirectional_rnn/bw/gru_cell/gates/kernel:0",
            # "rnn/bilstm_1/bidirectional_rnn/bw/gru_cell/gates/bias:0",
            # "rnn/bilstm_1/bidirectional_rnn/bw/gru_cell/candidate/kernel:0",
            # "rnn/bilstm_1/bidirectional_rnn/bw/gru_cell/candidate/bias:0",
            
            #  # Третий слой BiGRU
            # "rnn/bilstm_2/bidirectional_rnn/fw/gru_cell/gates/kernel:0",
            # "rnn/bilstm_2/bidirectional_rnn/fw/gru_cell/gates/bias:0",
            # "rnn/bilstm_2/bidirectional_rnn/fw/gru_cell/candidate/kernel:0",
            # "rnn/bilstm_2/bidirectional_rnn/fw/gru_cell/candidate/bias:0",
            # "rnn/bilstm_2/bidirectional_rnn/bw/gru_cell/gates/kernel:0",
            # "rnn/bilstm_2/bidirectional_rnn/bw/gru_cell/gates/bias:0",
            # "rnn/bilstm_2/bidirectional_rnn/bw/gru_cell/candidate/kernel:0",
            # "rnn/bilstm_2/bidirectional_rnn/bw/gru_cell/candidate/bias:0",
            

            # # FC-слои
            # "fc_1/dense/kernel:0",
            # "fc_1/dense/bias:0",
            # "fc_2/dense/kernel:0",
            # "fc_2/dense/bias:0",
            # "softmax_linear/dense/kernel:0",
            # "softmax_linear/dense/bias:0"
        ]
        for v in tf.trainable_variables():
            if any(v.name in s for s in list_vars):
                reg_losses.append(tf.nn.l2_loss(v))
        if len(reg_losses):
            reg_losses = tf.multiply(tf.add_n(reg_losses, name="l2_loss"), self.config["l2_weight_decay"])
        else:
            reg_losses = 0
        return reg_losses

    def train(self, minibatches):
        self.run(self.metric_init_op)
        start = timeit.default_timer()
        preds = []
        trues = []
        for x, y, w, sl, re in minibatches:
            feed_dict = {
                self.signals: x,
                self.labels: y,
                self.is_training: True,
                self.loss_weights: w,
                self.seq_lengths: sl,
            }
            _, outputs = self.run([self.train_step_op, self.train_outputs], feed_dict=feed_dict)

            tmp_preds = np.reshape(outputs["train/preds"], (self.config["batch_size"], self.config["seq_length"]))
            tmp_trues = np.reshape(y, (self.config["batch_size"], self.config["seq_length"]))

            for i in range(self.config["batch_size"]):
                preds.extend(tmp_preds[i, :sl[i]])
                trues.extend(tmp_trues[i, :sl[i]])

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

    def evaluate(self, minibatches):
        start = timeit.default_timer()
        losses = []
        preds = []
        trues = []
        for x, y, w, sl, re in minibatches:
            feed_dict = {
                self.signals: x,
                self.labels: y,
                self.is_training: False,
                self.loss_weights: w,
                self.seq_lengths: sl,
            }
            outputs = self.run(self.test_outputs, feed_dict=feed_dict)
            losses.append(outputs["test/loss"])
            tmp_preds = np.reshape(outputs["test/preds"], (self.config["batch_size"], self.config["seq_length"]))
            tmp_trues = np.reshape(y, (self.config["batch_size"], self.config["seq_length"]))
            for i in range(self.config["batch_size"]):
                preds.extend(tmp_preds[i, :sl[i]])
                trues.extend(tmp_trues[i, :sl[i]])
        loss = np.mean(losses)
        acc = skmetrics.accuracy_score(y_true=trues, y_pred=preds)
        f1_score = skmetrics.f1_score(y_true=trues, y_pred=preds, average="macro")
        cm = skmetrics.confusion_matrix(y_true=trues, y_pred=preds, labels=[0,1,2,3,4])
        logger.info("Validate or Test Confusion Matrix:\n{}".format(cm))
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