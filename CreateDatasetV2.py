import numpy as np
import cv2 as cv2
from matplotlib import pyplot as plt
import glob
import imutils
from random import shuffle
import os
import pandas as pd
import DataProcessing as DP
import random
import argparse
import sys


PYTHON = sys.executable
parser = argparse.ArgumentParser()
parser.add_argument('--random_seed', default='0',
                    help="Seed")
parser.add_argument('--split', default='0.7',
                    help="Directory containing the dataset")
parser.add_argument('--data_dir', default='./datasets/wormml_v2/',
                    help="Directory containing the dataset")
args = parser.parse_args()
seed = args.random_seed

random.seed(seed)
print("seed: ", seed)
CROP = True
dataset_path = os.path.join('../datasets/',args.data_dir)
if os.path.exists(dataset_path):
    os.rmdir(dataset_path)

input_files_dir = '../datasets/wormml_v2/600HeadTailImagesXY/'
files = sorted([file for file in os.listdir(input_files_dir) if file.endswith(".jpg")])
labels_file = [file for file in os.listdir(input_files_dir) if file.endswith(".csv")]

#shuffle(files)
split_ratio = 0.7
no_of_files = len(files)
no_of_train_files = np.round(no_of_files * split_ratio)
data_type_path = 'train/'

if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)
if not os.path.exists(os.path.join(dataset_path, data_type_path)):
    os.makedirs(os.path.join(dataset_path, data_type_path))
if not os.path.exists(os.path.join(dataset_path, 'val')):
    os.makedirs(os.path.join(dataset_path, 'val'))

img_h = 480
img_w = 640
resize_h = 150
resize_w = 150
i = 0

normalized_labels = np.zeros((len(files), 4))
label_df = DP.collectAllData_v2(os.path.join(input_files_dir, labels_file[0]))
normalized_labels_df = pd.DataFrame(columns=['headX', 'headY', 'tailX', 'tailY'])



random.shuffle(files)
for idx, file_name in enumerate(files):
    file_name_w_path = os.path.join(input_files_dir,file_name)
    img = cv2.imread(file_name_w_path, 0)

    folder_name = file_name[:-4] + '/'

    blur = cv2.GaussianBlur(img, (5,5), 1)
    th3 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                          cv2.THRESH_BINARY, 15, 6)
    #Invert the image as 0 is treated as background (for our case white is background)
    th3 = 255 - th3

    connectivity = 8
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(th3, connectivity, cv2.CV_32S)
    # initially, just grab the biggest connected component, later can use sizethresh = 9000 based on pixel area
    ix_of_tracked_component = np.argmax(stats[1:, 4]) + 1

    worm_img = (labels == ix_of_tracked_component).astype(np.uint8) * 255

    cnts = cv2.findContours(worm_img.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    x, y, w, h = cv2.boundingRect(cnts[0])
    #print(h, w)

    margin = 20     #include 10 pixels around bounding box
    y_low = max(y - margin,0)
    y_high = min(y + h + margin, img_h)
    x_low = max(x - margin, 0)
    x_high = min(x + w + margin, img_w)

    crop_img = img[y_low:y_high, x_low:x_high]
    crop_img_binary = worm_img[y_low:y_high, x_low:x_high]
    crop_img = cv2.resize(crop_img, (resize_w, resize_h))
    crop_img_binary = cv2.resize(crop_img_binary, (resize_w, resize_h))

    HeadX, HeadY, TailX, TailY = label_df.loc[folder_name[:-1]]

    normalized_labels[idx, 0] = (HeadX - x_low)/(x_high-x_low) #x
    normalized_labels[idx, 0] = normalized_labels[idx, 0]
    normalized_labels[idx, 1] = (HeadY - y_low)/(y_high-y_low)
    normalized_labels[idx, 1] = normalized_labels[idx, 1]

    normalized_labels[idx, 2] = (TailX - x_low)/(x_high-x_low)
    normalized_labels[idx, 2] = normalized_labels[idx, 2]
    normalized_labels[idx, 3] = (TailY - y_low)/(y_high-y_low)
    normalized_labels[idx, 3] = normalized_labels[idx, 3]
    normalized_labels_df.loc[folder_name[:-1]] = normalized_labels[idx, 0], normalized_labels[idx, 1], normalized_labels[idx, 2], normalized_labels[idx, 3]

    # Uncomment if want to visualize labels on crops
    #crop_img = cv2.cvtColor(crop_img, cv2.COLOR_GRAY2RGB)
    #crop_img = cv2.circle(crop_img, (int(normalized_labels[idx, 0]*resize_h), int(normalized_labels[idx, 1]*resize_w)), 10,  (255, 0, 0), -1)
    #crop_img = cv2.circle(crop_img, (int(normalized_labels[idx,2]*resize_h), int(normalized_labels[idx, 3]*resize_w)), 10, (0, 255, 0), -1)

    if idx == no_of_train_files:
        data_type_path = 'val/'
    path = dataset_path + data_type_path + folder_name
    if not os.path.exists(path):
        os.mkdir(path)
    if not os.path.exists(path + 'images/'):
        os.mkdir(path + 'images/')
    if not os.path.exists(path + 'masks/'):
        os.mkdir(path + 'masks/')
    if not os.path.exists(path + 'crops_binary/'):
        os.mkdir(path + 'crops_binary/')
    if not os.path.exists(path + 'crops/'):
        os.mkdir(path + 'crops/')

    cv2.imwrite(dataset_path + data_type_path + folder_name + 'images/' + folder_name[:-1] + '.png', img)
    cv2.imwrite(dataset_path + data_type_path + folder_name + 'masks/' + folder_name[:-1] + '.png', worm_img)
    cv2.imwrite(dataset_path + data_type_path + folder_name + 'crops/' + folder_name[:-1] + '.png', crop_img)
    cv2.imwrite(dataset_path + data_type_path + folder_name + 'crops_binary/' + folder_name[:-1] + '.png', crop_img_binary)


np.save(dataset_path + 'bbox_head_tale_coord.csv', normalized_labels)
normalized_labels_df.to_csv(dataset_path+'normalized_labels_df.csv')
