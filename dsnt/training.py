import logging
import os

from tqdm import trange
import tensorflow as tf
import numpy as np
import cv2


def train_sess(sess, model_spec, writer, params):
    """Train the model on `num_steps` batches
    Args:
        sess: (tf.Session) current session
        model_spec: (dict) contains the graph operations or nodes needed for training
        num_steps: (int) train for this number of batches
        writer: (tf.summary.FileWriter) writer for summaries
        params: (Params) hyperparameters
    """
    # Get relevant graph operations or nodes needed for training
    loss = model_spec['loss']
    train_op = model_spec['train_op']
    update_metrics = model_spec['update_metrics']
    metrics = model_spec['metrics']
    summary_op = model_spec['summary_op']
    global_step = tf.train.get_global_step()

    # Load the training dataset into the pipeline and initialize the metrics local variables
    sess.run(model_spec['iterator_init_op'])
    sess.run(model_spec['metrics_init_op'])

    i = 0
    while True:
        try:
            # Evaluate summaries for tensorboard only once in a while
            if i % params.save_summary_steps == 0:
                # Perform a mini-batch update

                _, _, loss_val, summ, global_step_val = sess.run([train_op, update_metrics, loss,
                                                                  summary_op, global_step])

                # Write summaries for tensorboard
                writer.add_summary(summ, global_step_val)

            else:
                _, _, loss_val = sess.run([train_op, update_metrics, loss])

            print("step: {} Training MSE: {:.5f}".format(global_step_val, loss_val))
            i += 1
        except tf.errors.OutOfRangeError:
            break


def evaluate_sess(sess, model_spec, writer=None, params=None):
    """Train the model on `num_steps` batches.
    Args:
        sess: (tf.Session) current session
        model_spec: (dict) contains the graph operations or nodes needed for training
        num_steps: (int) train for this number of batches
        writer: (tf.summary.FileWriter) writer for summaries. Is None if we don't log anything
        params: (Params) hyperparameters
    """
    update_metrics = model_spec['update_metrics']
    eval_metrics = model_spec['metrics']
    global_step = tf.train.get_global_step()

    # Load the evaluation dataset into the pipeline and initialize the metrics init op
    sess.run(model_spec['iterator_init_op'])
    sess.run(model_spec['metrics_init_op'])

    # compute metrics over the dataset
    while True:
        try:
            sess.run(update_metrics)
        except tf.errors.OutOfRangeError:
            break
    # Get the values of the metrics
    metrics_values = {k: v[0] for k, v in eval_metrics.items()}
    metrics_val = sess.run(metrics_values)
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_val.items())
    logging.info("- Eval metrics: " + metrics_string)
    print("Test MSE: {:.5f}".format(metrics_val['loss']))

    # Add summaries manually to writer at global_step_val
    if writer is not None:
        global_step_val = sess.run(global_step)
        for tag, val in metrics_val.items():
            summ = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=val)])
            writer.add_summary(summ, global_step_val)

    return metrics_val

def store_image_preds(sess, model_spec, writer, params, mode, model_dir):
    """Store predictions on train images
      Args:
          sess: (tf.Session) current session
          model_spec: (dict) contains the graph operations or nodes needed for training
          writer: (tf.summary.FileWriter) writer for summaries
          params: (Params) hyperparameters
      """
    # Get relevant graph operations or nodes needed for training
    loss = model_spec['loss']
    update_metrics = model_spec['update_metrics']
    metrics = model_spec['metrics']
    summary_op = model_spec['summary_op']
    global_step = tf.train.get_global_step()

    images = model_spec['images']
    labels = model_spec['labels']
    predictions = model_spec['predictions']
    # Load the training dataset into the pipeline and initialize the metrics local variables
    sess.run(model_spec['iterator_init_op'])
    sess.run(model_spec['metrics_init_op'])


    if mode == 'train':
        path = os.path.join(model_dir, 'train_images')
    else:
        path = os.path.join(model_dir, 'eval_images')

    if not os.path.exists(path):
        os.mkdir(path)

    j = 0
    while True:
        try:
            # Perform a mini-batch update
            imgs, y_true, y_pred, _ = sess.run([images, labels, predictions, update_metrics])
            targets_pred_int = y_pred
            # if j==0:
            #     print('predictions: ', y_pred)
            #     print('ground truth: ', y_true)
            targets_pred_int[:, 0] = targets_pred_int[:, 0] * params.img_h
            targets_pred_int[:, 1] = targets_pred_int[:, 1] * params.img_w

            if params.task == 'headtail':
                targets_pred_int[:, 2] = targets_pred_int[:, 2] * params.img_h
                targets_pred_int[:, 3] = targets_pred_int[:, 3] * params.img_w

            targets_pred_int = np.floor(targets_pred_int).astype('uint8')

            for i in range(imgs.shape[0]):
                imgs[i] = imgs[i]*255
                x_rgb = cv2.cvtColor(imgs[i], cv2.COLOR_GRAY2RGB)

                cv2.circle(x_rgb, (targets_pred_int[i,0], targets_pred_int[i, 1]), params.radius, (256, 0, 0), -1)
                cv2.circle(x_rgb, (int(y_true[i, 0] * params.img_h), int(y_true[i, 1] * params.img_w)),
                               params.radius, (0, 256, 0), -1)

                if params.task == 'headtail':
                    cv2.circle(x_rgb, (targets_pred_int[i, 2], targets_pred_int[i, 3]), params.radius, (256, 0, 256), -1)
                    cv2.circle(x_rgb, (int(y_true[i, 2] * params.img_h), int(y_true[i, 3] * params.img_w)),
                               params.radius, (0, 0, 256), -1)

                cv2.imwrite(os.path.join(path, str(j) + '.png'), x_rgb)
                j += 1

        except tf.errors.OutOfRangeError:
            break

    if mode == 'train':
        print("Final Training metrics")
    if mode == 'test':
        print("Final Eval metrics")
    # Get the values of the metrics
    metrics_values = {k: v[0] for k, v in metrics.items()}
    metrics_val = sess.run(metrics_values)
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_val.items())
    logging.info("- Eval metrics: " + metrics_string)
    print(" MSE: {:.5f}".format(metrics_val['loss']))
    if params.task=='headtail':
        print(" Head Accuracy alpha {:.5f} = : {:.5f}".format(params.alpha0, metrics_val['head_acc'+str(params.alpha0)]))
        print(" Head Accuracy alpha {:.5f} = : {:.5f}".format(params.alpha1, metrics_val['head_acc'+str(params.alpha1)]))
        print(" Head Accuracy alpha {:.5f} = : {:.5f}".format(params.alpha2, metrics_val['head_acc'+str(params.alpha2)]))
        print(" Head Accuracy alpha {:.5f} = : {:.5f}".format(params.alpha3, metrics_val['head_acc'+str(params.alpha3)]))
        print(" Head Accuracy alpha {:.5f} = : {:.5f}".format(params.alpha4, metrics_val['head_acc'+str(params.alpha4)]))
        print('\n')
        print(" Tail Accuracy alpha {:.5f} = : {:.5f}".format(params.alpha0, metrics_val['tail_acc'+str(params.alpha0)]))
        print(" Tail Accuracy alpha {:.5f} = : {:.5f}".format(params.alpha1, metrics_val['tail_acc'+str(params.alpha1)]))
        print(" Tail Accuracy alpha {:.5f} = : {:.5f}".format(params.alpha2, metrics_val['tail_acc'+str(params.alpha2)]))
        print(" Tail Accuracy alpha {:.5f} = : {:.5f}".format(params.alpha3, metrics_val['tail_acc'+str(params.alpha3)]))
        print(" Tail Accuracy alpha {:.5f} = : {:.5f}".format(params.alpha4, metrics_val['tail_acc'+str(params.alpha4)]))



