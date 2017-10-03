import cv2
import numpy as np
import os
from random import shuffle
from tqdm import tqdm

TRAIN_DIR = '/home/mohammad/Desktop/ML/imageClassifier/train'
TEST_DIR = '/home/mohammad/Desktop/ML/imageClassifier/test1'
IMG_SIZE = 50
LR = 1e-3


MODEL_NAME = 'dogsVScats-{}-{}.model'.format(LR, '2conv-basic')

def label_img(img):
    word_label = img.split('.')[-3]
    if word_label == 'cat': return [1,0]
    elif word_label == 'dog': return[0,1]


def create_train_data():
    train_data = []
    for img in tqdm(os.listdir(TRAIN_DIR)):
        label = label_img(img)
        path = os.path.join(TRAIN_DIR, img)
        img = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (IMG_SIZE,IMG_SIZE))
        train_data.append([np.array(img), np.array(label)])
    shuffle(train_data)
    np.save('train_data.npy', train_data)
    return train_data


create_train_data()

def process_test_dat():
    testing_data = []
    for img in tqdm(os.listdir(TEST_DIR)): 
        path = os.path.join(TEST_DIR, img)
        img_num = img.split('.')[0]
        img = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (IMG_SIZE,IMG_SIZE))
        train_data.append([np.array(img), img_num])

    np.save('test_data.npy', testing_data)
    return testing_data

train_data = creat_train_data()
# if you have it
#train_data = np.load('train_data.npy')
