import tensorflow as tf

class AutoEncoder(object):
    def __init__(self, config):
        self.struc_size = config['struc_size']
        self.cont_size = config['cont_size']
        self.encoder1 = config['encoder1']
        self.encoder2 = config['encoder2']
        self.decoder1 = config['decoder1']
        self.decoder2 = config['decoder2']
        self.learning_rate = config['learning_rate']

    def _add_placeholders(self):
        self.input_x = tf.placeholder(tf.float32, [None, self.struc_size], name = "struc_input")
        self.input_y = tf.placeholder(tf.float32, [None, self.cont_size], name = "cont_input")

        # Homophily neighbors for structure and content
        self.input_x_neigh1 = tf.placeholder(tf.float32, [None, self.struc_size], name = "struc_input_neigh1")
        self.input_x_neigh2 = tf.placeholder(tf.float32, [None, self.struc_size], name = "struc_input_neigh2")
        self.input_y_neigh1 = tf.placeholder(tf.float32, [None, self.cont_size], name = "cont_input_neigh1")
        self.input_y_neigh2 = tf.placeholder(tf.float32, [None, self.cont_size], name = "cont_input_neigh2")

        self.oval1 = tf.placeholder(tf.float32, [None], name = "o1_coeff")
        self.oval2 = tf.placeholder(tf.float32, [None], name = "o2_coeff")
        self.oval3 = tf.placeholder(tf.float32, [None], name = "o3_coeff")

    def _add_encoder_struc(self, batch_x, reuse = False):
        xvec =  batch_x
        with tf.variable_scope("struc_encoder", reuse = reuse):
            for ii in range(len(self.encoder1)):
                layer_name = 'layer_' + str(ii)
                xvec = tf.layers.dense(xvec, self.encoder1[ii],
                                       activation = tf.nn.leaky_relu, use_bias = True, name = layer_name)

        # struc_embeddings
        return xvec

    def _add_encoder_cont(self, batch_x, reuse = False):
        xvec =  batch_x
        with tf.variable_scope("cont_encoder", reuse = reuse):
            for ii in range(len(self.encoder2)):
                layer_name = 'layer_' + str(ii)
                xvec = tf.layers.dense(xvec, self.encoder2[ii],
                                       activation = tf.nn.leaky_relu, use_bias = True, name = layer_name)
        # cont_embeddings
        return xvec

    def _add_decoder_struc(self, hidden_x):
        xvec =  hidden_x
        with tf.variable_scope("struc_decoder"):
            for ii in range(len(self.decoder1)):
                layer_name = 'layer_' + str(ii)
                xvec = tf.layers.dense(xvec, self.decoder1[ii],
                                       activation = tf.nn.leaky_relu, use_bias = True, name = layer_name)
            input_rec = tf.layers.dense(xvec, self.struc_size, activation=tf.nn.relu, use_bias=False, name = "struc_final_layer")

        return input_rec

    def _add_decoder_cont(self, hidden_x):
        xvec =  hidden_x
        with tf.variable_scope("cont_decoder"):
            for ii in range(len(self.decoder2)):
                layer_name = 'layer_' + str(ii)
                xvec = tf.layers.dense(xvec, self.decoder2[ii],
                                       activation = tf.nn.leaky_relu, use_bias = True, name = layer_name)
            input_rec = tf.layers.dense(xvec, self.cont_size, activation=tf.nn.relu, use_bias=False, name = "cont_final_layer")

        return input_rec

    def _add_loss(self, batch_x, batch_y,
                        decoded_x, decoded_y,
                        struc_hid, cont_hid,
                        struct_neigh1, struct_neigh2,
                        cont_neigh1, cont_neigh2,
                        lo1, lo2, lo3):
        with tf.variable_scope('loss'):
            #  Loss 1 struct
            self.loss1 = tf.reduce_sum(tf.square((5.0 * batch_x + 1e-2) - decoded_x), axis=1)

            #  Loss 2 cont
            self.loss2 = tf.reduce_sum(tf.square((5.0 * batch_y + 1e-2) - decoded_y), axis=1) # 5 for best

            # Loss 3 align
            self.loss3 = tf.reduce_sum(tf.square(struc_hid - cont_hid), axis=1)

            # Homophily regularizer for Structure
            self.loss4 = tf.reduce_sum(tf.square(struc_hid - struct_neigh1), axis = 1) +\
                            tf.reduce_sum(tf.square(struc_hid - struct_neigh2), axis=1)

            # Homophile regularizer for Content
            self.loss5 = tf.reduce_sum(tf.square(cont_hid - cont_neigh1), axis = 1) +\
                            tf.reduce_sum(tf.square(cont_hid - cont_neigh2), axis=1)

            # Multiply with coefficients
            loss1 = tf.multiply(lo1, self.loss1)
            loss2 = tf.multiply(lo2, self.loss2)
            loss3 = tf.multiply(lo3, self.loss3)

            loss4 = tf.multiply(lo1, self.loss4)
            loss5 = tf.multiply(lo2, self.loss5)

            loss = loss1 + loss2 + loss3 + loss4 + loss5
            self.loss = tf.reduce_mean(loss)

            tf.summary.scalar("Total_Loss", self.loss)
            tf.summary.scalar("Structure_Loss", tf.reduce_mean(loss1))
            tf.summary.scalar("Content_Loss", tf.reduce_mean(loss2))
            tf.summary.scalar("Align_Loss", tf.reduce_mean(loss3))
            tf.summary.scalar("Struct_Homophily_Loss", tf.reduce_mean(self.loss4))
            tf.summary.scalar("Content_Homophily_Loss", tf.reduce_mean(self.loss5))

    def create_network(self):
        self._add_placeholders()

        self.struc_hid = self._add_encoder_struc(self.input_x)
        self.cont_hid = self._add_encoder_cont(self.input_y)

        self.struct_neigh1 = self._add_encoder_struc(self.input_x_neigh1, reuse = True)
        self.struct_neigh2 = self._add_encoder_struc(self.input_x_neigh2, reuse = True)
        self.cont_neigh1 = self._add_encoder_cont(self.input_y_neigh1, reuse = True)
        self.cont_neigh2 = self._add_encoder_cont(self.input_y_neigh2, reuse = True)

        self.decoded_x = self._add_decoder_struc(self.struc_hid)
        self.decoded_y = self._add_decoder_cont(self.cont_hid)

        # Calculate coefficients
        lo1 = -1.0 * tf.log(self.oval1)
        lo2 = -1.0 * tf.log(self.oval2)
        lo3 = -1.0 * tf.log(self.oval3)

        self._add_loss(self.input_x, self.input_y, self.decoded_x, self.decoded_y,
                        self.struc_hid, self.cont_hid, self.struct_neigh1, self.struct_neigh2,
                        self.cont_neigh1, self.cont_neigh2, lo1, lo2, lo3)

    def initialize_summary_writer(self, sess, fname):
        self.all_summary = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(fname, sess.graph)

    def initialize_optimizer(self, config):
        self.global_step = tf.Variable(0, name = "global_step", trainable = False)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            self.grads_and_vars = optimizer.compute_gradients(self.loss)
            self.train_op = optimizer.apply_gradients(self.grads_and_vars, global_step = self.global_step)

    def train_step(self, sess, feed_dict, start_align = False, print_this = True):
        feed = {}
        feed[self.input_x] = feed_dict["struc_input"]
        feed[self.input_y] = feed_dict["cont_input"]
        feed[self.input_x_neigh1] = feed_dict["struc_input_neigh1"]
        feed[self.input_x_neigh2] = feed_dict["struc_input_neigh2"]
        feed[self.input_y_neigh1] = feed_dict["cont_input_neigh1"]
        feed[self.input_y_neigh2] = feed_dict["cont_input_neigh2"]
        feed[self.oval1] = feed_dict["o1_coeff"]
        feed[self.oval2] = feed_dict["o2_coeff"]
        feed[self.oval3] = feed_dict["o3_coeff"]

        run_vars = [self.train_op, self.global_step, self.loss, self.all_summary]
        _, idx, rloss, summ = sess.run(run_vars, feed_dict = feed)

        self.writer.add_summary(summ, idx)
        if print_this:
            print(idx, 'LOSS =', rloss)

    def get_hidden(self, sess, x_batch, y_batch):
        feed = {}
        feed[self.input_x] = x_batch
        feed[self.input_y] = y_batch

        struc_emb, cont_emb = sess.run([self.struc_hid, self.cont_hid], feed_dict = feed)

        return struc_emb, cont_emb

    def get_all_losses(self, sess, feed_dict):
        feed = {}
        feed[self.input_x] = feed_dict["struc_input"]
        feed[self.input_y] = feed_dict["cont_input"]
        feed[self.input_x_neigh1] = feed_dict["struc_input_neigh1"]
        feed[self.input_x_neigh2] = feed_dict["struc_input_neigh2"]
        feed[self.input_y_neigh1] = feed_dict["cont_input_neigh1"]
        feed[self.input_y_neigh2] = feed_dict["cont_input_neigh2"]

        run_vars = [self.loss1, self.loss2, self.loss3, self.loss4, self.loss5]
        l1, l2, l3, l4, l5 = sess.run(run_vars, feed_dict = feed)

        return l1, l2, l3, l4, l5
