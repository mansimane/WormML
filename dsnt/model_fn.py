import tensorflow as tf
import cv2
import numpy as np
import sonnet as snt
import dsnt
import sys


def inference_batch_norm_vgg16_head_tail_scratch(inputs):


    inputs = snt.Conv2D(output_channels=64,
                        kernel_shape=3,
                        rate=1,
                        padding='SAME',
                        name='conv1_1')(inputs)
    inputs = tf.nn.relu(inputs)


    inputs = snt.Conv2D(output_channels=64,
                        kernel_shape=3,
                        rate=1,
                        padding='SAME',
                        name='conv1_2')(inputs)
    inputs = tf.nn.relu(inputs)

    inputs = tf.nn.max_pool(inputs, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

    inputs = snt.Conv2D(output_channels=128,
                        kernel_shape=3,
                        rate=1,
                        padding='SAME',
                        name='conv2_1')(inputs)
    inputs = tf.nn.relu(inputs)

    inputs = snt.Conv2D(output_channels=128,
                        kernel_shape=3,
                        rate=1,
                        padding='SAME',
                        name='conv2_2')(inputs)
    inputs = tf.nn.relu(inputs)
    inputs = tf.nn.max_pool(inputs, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')


    inputs = snt.Conv2D(output_channels=256,
                        kernel_shape=3,
                        rate=1,
                        padding='SAME',
                        name='conv3_1')(inputs)
    inputs = tf.nn.relu(inputs)

    inputs = snt.Conv2D(output_channels=256,
                        kernel_shape=3,
                        rate=1,
                        padding='SAME',
                        name='conv3_2')(inputs)
    inputs = tf.nn.relu(inputs)

    inputs = snt.Conv2D(output_channels=256,
                        kernel_shape=3,
                        rate=1,
                        padding='SAME',
                        name='conv3_3')(inputs)
    inputs = tf.nn.relu(inputs)
    inputs = tf.nn.max_pool(inputs, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

    inputs = snt.Conv2D(output_channels=512,
                        kernel_shape=3,
                        rate=1,
                        padding='SAME',
                        name='conv4_1')(inputs)
    inputs = tf.nn.relu(inputs)

    inputs = snt.Conv2D(output_channels=512,
                        kernel_shape=3,
                        rate=1,
                        padding='SAME',
                        name='conv4_2')(inputs)
    inputs = tf.nn.relu(inputs)

    inputs = snt.Conv2D(output_channels=512,
                        kernel_shape=3,
                        rate=1,
                        padding='SAME',
                        name='conv4_3')(inputs)
    inputs = tf.nn.relu(inputs)
    inputs = tf.nn.max_pool(inputs, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

    inputs = snt.Conv2D(output_channels=512,
                        kernel_shape=3,
                        rate=1,
                        padding='SAME',
                        name='conv5_1')(inputs)
    inputs = tf.nn.relu(inputs)

    inputs = snt.Conv2D(output_channels=512,
                        kernel_shape=3,
                        rate=1,
                        padding='SAME',
                        name='conv5_2')(inputs)
    inputs = tf.nn.relu(inputs)

    inputs = snt.Conv2D(output_channels=512,
                        kernel_shape=3,
                        rate=1,
                        padding='SAME',
                        name='conv5_3')(inputs)
    inputs = tf.nn.relu(inputs)
    inputs = tf.nn.max_pool(inputs, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

    inputs_h = snt.Conv2D(output_channels=1,
                          kernel_shape=1,
                          padding='SAME',
                          name='conv6_1_h')(inputs)
    norm_heatmap_h, coords_h = dsnt.dsnt(inputs_h)

    inputs_t = snt.Conv2D(output_channels=1,
                          kernel_shape=1,
                          padding='SAME',
                          name='conv6_1_t')(inputs)
    norm_heatmap_t, coords_t = dsnt.dsnt(inputs_t)

    # The Sonnet option
    # coords, norm_heatmap = DSNT()(inputs)
    return norm_heatmap_h, coords_h, norm_heatmap_t, coords_t



def model_fn(mode, inputs, params, reuse=False):
    '''

    :param mode:  (string) can be 'train' or 'eval'
    :param inputs: (dict) contains the inputs of the graph (features, labels...)
                this can be `tf.placeholder` or outputs of `tf.data`
    :param params: (Params) contains hyperparameters of the model (ex: `params.learning_rate`)
    :param reuse:(bool) whether to reuse the weights
    :return:  model_spec: (dict) contains the graph operations or nodes needed for training / evaluation
    '''


    with tf.variable_scope('model', reuse=reuse):

        # MODEL: define the layers of the model
        with tf.variable_scope('model', reuse=reuse):
            if params.task == 'headtail':
                if params.network == 'vgg16':
                    heatmap_h, preds_h, heatmap_t, preds_t = inference_batch_norm_vgg16_head_tail_scratch(inputs['images'])

                # The predictions are in the range [-1, 1] but I prefer to work with [0, 1]
                preds_h = (preds_h + 1) / 2
                preds_t = (preds_t + 1) / 2
                predictions = tf.stack([preds_h, preds_t], axis=1)#(predictions + 1) / 2
                predictions = tf.reshape(predictions, [-1, 4])

    # Coordinate regression loss
    loss_1 = tf.losses.mean_squared_error(inputs['labels'], predictions)
    # Regularization loss
    if params.task == 'headtail':

        loss_2 = dsnt.js_reg_loss(heatmap_h, inputs['labels'][:,0:2])
        loss_3 = dsnt.js_reg_loss(heatmap_t, inputs['labels'][:,2:4])
        loss = loss_1 + loss_2 + loss_3


    # Calculate accuracy

    if params.task == 'headtail':
        headinputs = tf.slice(inputs['labels'], [0, 0], [-1, 2])
        headpreds = tf.slice(predictions, [0, 0], [-1, 2])
        tailinputs = tf.slice(inputs['labels'], [0, 2], [-1, 2])
        tailpreds = tf.slice(predictions, [0, 2], [-1, 2])

        th0 = params.alpha0 * 1.0
        th1 = params.alpha1 * 1.0
        th2 = params.alpha2 * 1.0
        th3 = params.alpha3 * 1.0
        th4 = params.alpha4 *1.0



        head_predictions0 = tf.math.less(tf.norm(headinputs - headpreds, axis=1), th0)
        tail_predictions0 = tf.math.less(tf.norm(tailinputs - tailpreds, axis=1), th0)
        head_predictions1 = tf.math.less(tf.norm(headinputs - headpreds, axis=1), th1)
        tail_predictions1 = tf.math.less(tf.norm(tailinputs - tailpreds, axis=1), th1)
        head_predictions2 = tf.math.less(tf.norm(headinputs - headpreds, axis=1), th2)
        tail_predictions2 = tf.math.less(tf.norm(tailinputs - tailpreds, axis=1), th2)
        head_predictions3 = tf.math.less(tf.norm(headinputs - headpreds, axis=1), th3)
        tail_predictions3 = tf.math.less(tf.norm(tailinputs - tailpreds, axis=1), th3)
        head_predictions4 = tf.math.less(tf.norm(headinputs - headpreds, axis=1), th4)
        tail_predictions4 = tf.math.less(tf.norm(tailinputs - tailpreds, axis=1), th4)

        head_predictions0 = tf.cast(head_predictions0, tf.float16)
        head_predictions1 = tf.cast(head_predictions1, tf.float16)
        head_predictions2 = tf.cast(head_predictions2, tf.float16)
        head_predictions3 = tf.cast(head_predictions3, tf.float16)
        head_predictions4 = tf.cast(head_predictions4, tf.float16)

        tail_predictions0 = tf.cast(tail_predictions0, tf.float16)
        tail_predictions1 = tf.cast(tail_predictions1, tf.float16)
        tail_predictions2 = tf.cast(tail_predictions2, tf.float16)
        tail_predictions3 = tf.cast(tail_predictions3, tf.float16)
        tail_predictions4 = tf.cast(tail_predictions4, tf.float16)

        ref = tf.ones_like(head_predictions0)

    # Define training step that minimizes the loss
    if mode == 'train':
        optimizer = tf.train.AdamOptimizer(learning_rate=params.learning_rate)
        global_step = tf.train.get_or_create_global_step()
        train_op = optimizer.minimize(loss, global_step=global_step)


    # Metrics for evaluation using tf.metrics (average over whole dataset)
    if params.task == 'headtail':

        with tf.variable_scope("metrics"):
            metrics = {
                'loss': tf.metrics.mean(loss),

                'head_acc'+str(params.alpha0):tf.metrics.accuracy(ref, head_predictions0),
                'head_acc'+str(params.alpha1):tf.metrics.accuracy(ref, head_predictions1),
                'head_acc'+str(params.alpha2):tf.metrics.accuracy(ref, head_predictions2),
                'head_acc'+str(params.alpha3):tf.metrics.accuracy(ref, head_predictions3),
                'head_acc'+str(params.alpha4):tf.metrics.accuracy(ref, head_predictions4),

                'tail_acc'+str(params.alpha0): tf.metrics.accuracy(ref, tail_predictions0),
                'tail_acc'+str(params.alpha1): tf.metrics.accuracy(ref, tail_predictions1),
                'tail_acc'+str(params.alpha2): tf.metrics.accuracy(ref, tail_predictions2),
                'tail_acc'+str(params.alpha3): tf.metrics.accuracy(ref, tail_predictions3),
                'tail_acc'+str(params.alpha4): tf.metrics.accuracy(ref, tail_predictions4)
                }

    # Group the update ops for the tf.metrics
    update_metrics_op = tf.group(*[op for _, op in metrics.values()])
    # Get the op to reset the local variables used in tf.metrics
    metric_variables = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="metrics")
    metrics_init_op = tf.variables_initializer(metric_variables)

    # Summaries for training
    tf.summary.scalar('loss', loss)

    model_spec = inputs
    model_spec['variable_init_op'] = tf.global_variables_initializer()
    model_spec['loss'] = loss
    model_spec['metrics_init_op'] = metrics_init_op
    model_spec['metrics'] = metrics
    model_spec['update_metrics'] = update_metrics_op
    model_spec['summary_op'] = tf.summary.merge_all()
    model_spec['predictions'] = predictions
    model_spec['head_predictions0'] = head_predictions0
    model_spec['headinputs'] = headinputs
    model_spec['headpreds'] = headpreds
    model_spec['th0'] = th0

    if mode == 'train':
        model_spec['train_op'] = train_op

    return model_spec