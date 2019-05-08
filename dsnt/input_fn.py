import tensorflow as tf
import numpy as np
import cv2
import os
import pandas as pd
import sys




def _parse_function(filename, label):
    """Obtain the image from the filename (for both training and validation).
    The following operations are applied:
        - Decode the image from jpeg format
        - Convert to float and to range [0, 1]
    """
    # size = 300
    image_string = tf.read_file(filename)

    # Don't use tf.image.decode_image, or the output shape will be undefined
    image_decoded = tf.image.decode_png(image_string, channels=1)

    # This will convert to float values in [0, 1]
    image = tf.image.convert_image_dtype(image_decoded, tf.float32)
    # tf.image.per_image_standardization
    #resized_image = tf.image.resize_images(image, [size, size])
    label = tf.cast(label, tf.float32)
    return image, label

def train_preprocess(image, label, use_random_flip=True):
    """Image preprocessing for training.
    Apply the following operations:
        - Horizontally flip the image with probability 1/2
        - Apply random brightness and saturation
    """

    if use_random_flip:

        mask = tf.zeros_like(image)
        H, W = tf.cast(tf.shape(image)[0], tf.float32), tf.cast(tf.shape(image)[1], tf.float32)
        x, y = tf.cast(H * label[1],tf.int32), tf.cast(W * label[0], tf.int32)
        H_int = tf.cast(H,tf.int64)
        W_int = tf.cast(W,tf.int64)
        shape = [H_int, W_int, 1]
        indices = [[x, y,0]]
        values = [1.0]
        delta = tf.SparseTensor(indices, values, shape)
        mask = tf.sparse_tensor_to_dense(delta) + mask # this just does mask[x,y,0] = 1.0


        concat = tf.concat([image, mask], axis=2)
        concat = tf.image.random_flip_left_right(concat)
        concat = tf.image.random_flip_up_down(concat)
        concat = tf.image.rot90(concat, tf.random_uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))

        mask = concat[:,:,1]
        label = tf.where(tf.equal(mask, 1.0))

        label = tf.cast(label, tf.float32) / H  #this division needs to be done separately if H != W
        image = concat[:,:,0]
        image = tf.expand_dims(image, 2)
        label = tf.reshape(label, [2, ])
        label = tf.reverse(label, axis=[0])
    image = tf.image.random_brightness(image, max_delta=32.0 / 255.0)

    # # Make sure the image is still in [0, 1]
    image = tf.clip_by_value(image, 0.0, 1.0)

    return image, label


def train_preprocess_head_tail(image, label, use_random_flip=True):
    """Image preprocessing for training.
    Apply the following operations:
        - Horizontally flip the image with probability 1/2
        - Apply random brightness and saturation
    """
    if use_random_flip:

        mask = tf.zeros_like(image)
        H, W = tf.cast(tf.shape(image)[0], tf.float32), tf.cast(tf.shape(image)[1], tf.float32)
        xh, yh = tf.cast(H * label[1],tf.int32), tf.cast(W * label[0], tf.int32)
        xt, yt = tf.cast(H * label[3],tf.int32), tf.cast(W * label[2], tf.int32)
        H_int = tf.cast(H,tf.int64)
        W_int = tf.cast(W,tf.int64)
        shape = [H_int, W_int, 1]

        indices1 = [[xh, yh, 0]]
        values1 = [1.0]
        delta1 = tf.SparseTensor(indices1, values1, shape)

        indices2 = [[xt, yt, 0]]
        values2 = [2.0]
        delta2 = tf.SparseTensor(indices2, values2, shape)

        mask = tf.sparse_tensor_to_dense(delta1) + mask  + tf.sparse_tensor_to_dense(delta2) # this just does mask[x,y,0] = 1.0


        concat = tf.concat([image, mask], axis=2)
        concat = tf.image.random_flip_left_right(concat)
        concat = tf.image.random_flip_up_down(concat)
        concat = tf.image.rot90(concat, tf.random_uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))

        mask = concat[:,:,1]
        image = concat[:,:,0]
        image = tf.expand_dims(image, 2)

        labelh = tf.where(tf.equal(mask, 1.0))
        labelt = tf.where(tf.equal(mask, 2.0))

        labelh = tf.cast(labelh, tf.float32) / H  #this division needs to be done separately if H != W
        labelt = tf.cast(labelt, tf.float32) / H  #this division needs to be done separately if H != W

        labelh = tf.reverse(labelh, axis=[1])
        labelt = tf.reverse(labelt, axis=[1])

        label = tf.stack([labelh, labelt], axis=1)
        label = tf.reshape(label, [4, ])
        #tf.print(label,output_stream=sys.stderr)
        #tf.print(indices, output_stream=sys.stderr)
    #chx = tf.zeros(image.)
    image = tf.image.random_brightness(image, max_delta=32.0 / 255.0)

    #image = tf.image.random_saturation(image, lower=0.5, upper=1.5)


    # # Make sure the image is still in [0, 1]
    image = tf.clip_by_value(image, 0.0, 1.0)

    return image, label


def build_real_data_v2 (img_h, img_w, batch_size, params):
    '''

    :param img_h:
    :param img_w:
    :param batch_size:
    :return: Train batch, Train op, Test batch, Test op
    '''

    if params.task == "headtail":
        label_idx_low = 0
        label_idx_hi = 4

    elif params.TAIL == True:
        label_idx_low = 2
        label_idx_hi = 4
    else:
        label_idx_low = 0
        label_idx_hi = 2

    tf.set_random_seed(params.random_seed)
    no_of_tr_files = False
    labels_df = pd.read_csv('../MaskRCNN/datasets/'+ params.dataset + '/normalized_labels_df.csv')
    labels_df = labels_df.set_index('Unnamed: 0')

    total_no_of_tr_files = len(os.listdir('../MaskRCNN/datasets/' + params.dataset + '/train/'))

    if no_of_tr_files:
        tr_folder_names = sorted(os.listdir('../MaskRCNN/datasets/' + params.dataset + '/train/'))[0:no_of_tr_files]
        labels_tr = np.array(labels_df.loc[tr_folder_names])[:,label_idx_low:label_idx_hi]

    else:
        tr_folder_names = sorted(os.listdir('../MaskRCNN/datasets/' + params.dataset +  '/train/'))
        no_of_tr_files = len(tr_folder_names)
        labels_tr = np.array(labels_df.loc[tr_folder_names])[:,label_idx_low:label_idx_hi]

    prev_path = '../MaskRCNN/datasets/' + params.dataset + '/train/'
    intermediate_folder = '/crops/'

    tr_filenames = [prev_path + folder_name + intermediate_folder + folder_name + '.png' for folder_name in tr_folder_names]

    dataset_tr = tf.data.Dataset.from_tensor_slices((tf.constant(tr_filenames), tf.constant(labels_tr)))
    dataset_tr = dataset_tr.shuffle(buffer_size=54)
    dataset_tr = dataset_tr.map(_parse_function, num_parallel_calls=params.num_parallel_calls)
    if params.task == 'headtail':
        dataset_tr = dataset_tr.map(train_preprocess_head_tail, num_parallel_calls=params.num_parallel_calls)
    else:
        dataset_tr = dataset_tr.map(train_preprocess, num_parallel_calls=params.num_parallel_calls)

    dataset_tr = dataset_tr.batch(batch_size).prefetch(1)

    '''
        Test Data
    '''
    # Create reinitializable iterator from dataset
    itr_tr = dataset_tr.make_initializable_iterator()
    itr_tr_batch = itr_tr.get_next()
    itr_tr_op = itr_tr.initializer

    tst_folder_names = sorted(os.listdir('../MaskRCNN/datasets/' + params.dataset + '/val/'))
    prev_path = '../MaskRCNN/datasets/' + params.dataset + '/val/'
    tst_filenames = [prev_path + folder_name + intermediate_folder + folder_name + '.png' for folder_name in tst_folder_names]
    labels_tst = np.array(labels_df.loc[tst_folder_names])[:,label_idx_low:label_idx_hi]

    dataset_tst = tf.data.Dataset.from_tensor_slices((tf.constant(tst_filenames), tf.constant(labels_tst)))
    dataset_tst = dataset_tst.map(_parse_function, num_parallel_calls=params.num_parallel_calls)
    dataset_tst = dataset_tst.batch(batch_size).prefetch(1)

    itr_tst = dataset_tst.make_initializable_iterator()
    itr_tst_batch = itr_tst.get_next()
    itr_tst_op = itr_tst.initializer
    return itr_tr_batch, itr_tr_op , itr_tst_batch, itr_tst_op

