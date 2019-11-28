import numpy as np
import tensorflow as tf
import os
import json
import csv
from tqdm import tqdm
from datetime import datetime
from preprocessRW import computeRep
import argparse
from collections import defaultdict

from aemodel_done import AutoEncoder

def parse_args():
    parser = argparse.ArgumentParser(description="DONE embeddings")
    parser.add_argument('--config', nargs='?', default='config_done', help='Input config path')
    return parser.parse_args()

def load_config(fname='./config_done'):
    with open(fname, 'r') as fp:
        config = json.load(fp)
    return config

def read_csv_file_as_numpy(fname):
	with open(fname, 'r') as fp:
		rd = csv.reader(fp)
		ret = []
		for row in tqdm(rd):
			ret.append([float(r) for r in row])
	return np.array(ret)

def batch_iter(in_x, in_y, in_id, o1, o2, o3, batch_size, num_epochs, shuffle = True):
    data_x = np.array(in_x)
    data_y = np.array(in_y)
    data_i = np.array(in_id, dtype=int)

    data_size = data_x.shape[0]
    sample_idx = np.arange(data_size)
    order = np.arange(data_size)
    num_batches = int((data_size - 1) / batch_size) + 1

    iter_obj = tqdm(range(num_epochs)) if num_epochs > 1 else range(num_epochs)
    for epoch in iter_obj:
        if shuffle:
            np.random.shuffle(order)
            data_x = data_x[order]
            data_y = data_y[order]
            data_i = data_i[order]

        samples = np.zeros((data_size, 2))

        idx1, idx2 = [], []
        # For feeding tne Homophily inputs
        for idx in range(data_size):
            try:
                p=data_x[idx] / np.sum(data_x[idx])
                samples = np.random.choice(sample_idx, size=2,p = p/np.sum(p))
                idx1.append(samples[0])
                idx2.append(samples[1])
            except ZeroDivisionError:
                print 'Exception encountered'
                idx1.append(order[idx])
                idx2.append(order[idx])
                pass

        for batch_num in range(num_batches):
            feed_dict = {}
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)

            feed_dict["struc_input"] = data_x[start_index:end_index]
            feed_dict["cont_input"] = data_y[start_index:end_index]
            feed_dict["struc_input_neigh1"] = in_x[idx1[start_index : end_index]]
            feed_dict["struc_input_neigh2"] = in_x[idx2[start_index : end_index]]
            feed_dict["cont_input_neigh1"] = in_y[idx1[start_index : end_index]]
            feed_dict["cont_input_neigh2"] = in_y[idx2[start_index : end_index]]

            yield feed_dict, data_i[start_index:end_index], batch_num == num_batches - 1

def get_total_loss(model, sess, x_train, y_train, indexes, o1, o2, o3, batch_size):
    batcher = batch_iter(x_train, y_train, indexes, o1, o2, o3, batch_size, num_epochs = 1, shuffle = False)
    L1, L2, L3 = [], [], []
    L4, L5 = [], []

    for feed_dict, order, _ in batcher:
        feed_dict["o1_coeff"] = o1[order]
        feed_dict["o2_coeff"] = o2[order]
        feed_dict["o3_coeff"] = o3[order]
        l1, l2, l3, l4, l5 = model.get_all_losses(sess, feed_dict)
        L1.append(l1)
        L2.append(l2)
        L3.append(l3)
        L4.append(l4)
        L5.append(l5)

    L1, L2, L3 = np.concatenate(L1), np.concatenate(L2), np.concatenate(L3)
    L4, L5 = np.concatenate(L4), np.concatenate(L5)

    L1 = L1 + L4 / 2.0
    L2 = L2 + L5 / 2.0
    return L1, L2, L3

def update_ovals(l1, l2, l3):
    o1 = l1 / np.sum(l1)
    o2 = l2 / np.sum(l2)
    o3 = l3 / np.sum(l3)
    return o1, o2, o3

def trainer(sess, model, x_train, y_train, num_epochs):
    with sess.as_default():
        saver = tf.train.Saver(tf.global_variables(), max_to_keep = 1)

        o1 = (1.0 / Adj.shape[0]) * np.ones(x_train.shape[0])
        o2 = (1.0 / Adj.shape[0]) * np.ones(x_train.shape[0])
        o3 = (1.0 / Adj.shape[0]) * np.ones(x_train.shape[0])

        indexes = np.arange(x_train.shape[0])
        batcher = batch_iter(x_train, y_train, indexes, o1, o2, o3, batch_size, num_epochs)

        epoch = 0
        print 'Training...'
        for feed_dict, order, epoch_end in batcher:
            feed_dict["o1_coeff"] = o1[order]
            feed_dict["o2_coeff"] = o2[order]
            feed_dict["o3_coeff"] = o3[order]

            model.train_step(sess, feed_dict, start_align = (epoch > config['pretrain_threshold']), print_this = False)
            if epoch_end:
                if epoch > config['pretrain_threshold']:
                    l1, l2, l3 = get_total_loss(model, sess, x_train, y_train, indexes, o1, o2, o3, batch_size)
                    o1, o2, o3 = update_ovals(l1, l2, l3)
                epoch += 1

        struc_emb, cont_emb = model.get_hidden(sess, x_train, y_train)
        path = saver.save(sess, os.path.join(summ_file, 'model.ckpt'))
        print 'Final model saved at', path

        print 'Saving embeddings...'
        final = np.hstack((struc_emb, cont_emb))
        np.savetxt(os.path.join('emb', config['experiment_name'] + '.emb'), final)

        print 'Saving outlier values...'
        fname = 'ovals/' + config['experiment_name'] + '-oval1'
        np.savetxt(fname, o1)
        fname = 'ovals/' + config['experiment_name'] + '-oval2'
        np.savetxt(fname, o2)
        fname = 'ovals/' + config['experiment_name'] + '-oval3'
        np.savetxt(fname, o3)

args = parse_args()

# Load Config
print 'Using config path', args.config
config = load_config(args.config)

print 'Running experiment', config['experiment_name']

# Read Data
print 'Reading structure from', config['struc_file']
Adj = read_csv_file_as_numpy(config['struc_file'])
Adj = computeRep(Adj, 3, 0.3)
print 'Reading content from', config['cont_file']
Con = read_csv_file_as_numpy(config['cont_file'])
print 'Data load complete. Structure input size', Adj.shape, 'Content input size', Con.shape

config['struc_size'] = Adj.shape[1]
config['cont_size'] = Con.shape[1]

start_time = datetime.now()

# Build Network
model = AutoEncoder(config)
model.create_network()
model.initialize_optimizer(config)

# Create Session
session_conf = tf.ConfigProto(allow_soft_placement = True,
                              log_device_placement = False)
session_conf.gpu_options.allow_growth = True
sess = tf.Session(config = session_conf)
summ_file = './log/' + config['experiment_name']
model.initialize_summary_writer(sess, summ_file)
sess.run(tf.global_variables_initializer())

batch_size = config['batch_size'] if config['batch_size'] > 0 else Adj.shape[0]
num_epochs = config['num_epochs']

trainer(sess, model, Adj, Con, num_epochs)
sess.close()
end_time = datetime.now()
print 'Run completed in', (end_time - start_time).seconds, 'seconds.'
