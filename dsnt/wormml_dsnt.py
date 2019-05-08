# Imports
import tensorflow as tf
import cv2
import numpy as np
import sonnet as snt

# Import for us of the transform layer and loss function
import dsnt

# For the Sonnet Module
# from dsnt_snt import DSNT

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as lines
from matplotlib.patches import Polygon

import logging
import os
from training import train_sess, evaluate_sess, store_image_preds
import argparse

from tqdm import trange
import os
from utils import Params
import sys
sys.path.extend(['./'])

from input_fn import build_real_data_v2
from model_fn import *
import tensorflow.contrib.slim as slim

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/base_model',
                    help="Directory containing params.json")

args = parser.parse_args()

#model_dir = './experiments/base_model/'
model_dir = args.model_dir
json_path = os.path.join(model_dir, 'params.json')
params = Params(json_path)

#tf.enable_eager_execution()
tf.reset_default_graph()
tf.set_random_seed(params.random_seed)
np.random.seed(params.random_seed)


itr_tr_batch, itr_tr_op, itr_tst_batch, itr_tst_op = build_real_data_v2(params.img_h, params.img_w,
                                                                            params.batch_size, params)

train_inputs = {}
train_inputs['images'], train_inputs['labels'] = itr_tr_batch
train_inputs['iterator_init_op'] = itr_tr_op

tst_inputs = {}
tst_inputs['images'], tst_inputs['labels'] = itr_tst_batch
tst_inputs['iterator_init_op'] = itr_tst_op

# Define the model
train_model_spec = model_fn('train', train_inputs, params)
eval_model_spec = model_fn('eval', tst_inputs, params, reuse=True)


i = 0

def model_summary():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)

model_summary()

with tf.Session() as sess:
    sess.run(train_model_spec['variable_init_op'])
    # For tensorboard (takes care of writing summaries to files)
    train_writer = tf.summary.FileWriter(os.path.join(model_dir, 'train_summaries'), sess.graph)
    eval_writer = tf.summary.FileWriter(os.path.join(model_dir, 'eval_summaries'), sess.graph)


    for epoch in range(params.num_epochs):
        sess.run(itr_tr_op)
        train_sess(sess, train_model_spec, train_writer, params)
        metrics = evaluate_sess(sess, eval_model_spec, eval_writer)

    #Store predictions
    store_image_preds(sess, train_model_spec, train_writer, params, 'train', model_dir)
    store_image_preds(sess, eval_model_spec, eval_writer, params, 'test', model_dir)

