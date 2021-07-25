# -*- coding: utf-8 -*-
import sys

import os
import shutil
from PIL import Image
from dataset.transform_pixel import *
#from dataset.util import *
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from typing import List
import numpy as np
from PIL import Image

sys.path.append('../')
import config

# 13 15 18
'''
All classes in dataset VOC2012 (including 21 classes if background is counted)
'''
ALL_CLASSES = [
    "bg",           #  0
    "aeroplane",    #  1
    "bicycle",      #  2
    "bird",         #  3
    "boat",         #  4
    "bottle",       #  5
    "bus",          #  6
    "car",          #  7
    "cat",          #  8
    "chair",        #  9
    "cow",          # 10
    "diningtable",  # 11
    "dog",          # 12
    "horse",        # 13
    "motorbike",    # 14
    "person",       # 15
    "pottedplant",  # 16
    "sheep",        # 17
    "sofa",         # 18
    "train",        # 19
    "tvmonitor"     # 20
]

split = [
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], # split 0
    [               6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], # split 1
    [1, 2, 3, 4, 5,                 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], # split 2
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,                     16, 17, 18, 19, 20], # split 3
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15                    ]  # split 4
]



import xml.etree.ElementTree as ET
def get_class_names_of_file(pure_file_name, xml_dir = config.DATA_VOC + 'Annotations/'):
    """
    Get object names of a file, @param pure_file_name must be preprocessed!
    """
    xml_file = xml_dir + pure_file_name + '.xml'
    assert os.path.isfile(xml_file)
    tree = ET.parse(xml_file)
    root = tree.getroot()
    objs = root.findall('object')
    ans = set()
    for obj in objs:
        currentObj = obj.find('name').text
        ans.add(currentObj)
    return list(ans)


def _gen_split(
    classes,
    voc_path = config.DATA_VOC,
    train_or_val = 'train',
    file_name = 'split.txt',
    save_or_not = True
    ):
    xml_dir = voc_path + 'Annotations/'
    jpg_dir = voc_path + 'JPEGImages/'
    seg_dir = voc_path + 'ImageSets/Segmentation/'
    ans = {'classes': list(map(lambda cls: ALL_CLASSES.index(cls), classes)), 'files': []}
    lines = []
    session = 'train.txt' if train_or_val == 'train' else 'val.txt'
    with open(os.path.join(seg_dir + session), "r") as f:
        lines = f.read().splitlines()
        print("Total: {}".format(len(lines)))

    for line in lines:
        classes_of_file = get_class_names_of_file(line)
        for cls in classes_of_file:
            if cls in classes:
                ans['files'].append(line)
                break
    if save_or_not == True:
        fp = open(config.DATA_PATH + file_name,'w+')
        fp.write(', '.join(map(lambda n: str(n), ans['classes'])) + '\n')
        fp.write('\n'.join(ans['files']))
        fp.close()

    print("\
        Generated classes: {}, \n\
        Generated images: {}"
            .format(len(ans['classes']), len(ans['files']))
    )
    return ans


def gen_split0(
    voc_path = config.DATA_VOC,
    train_or_val = 'train',
    file_name = 'split0.txt',
    save_or_not = True
    ):
    return _gen_split(
        list(map(lambda i: ALL_CLASSES[i], split[0])), 
        file_name = file_name
    )

def gen_split1(
    voc_path = config.DATA_VOC,
    train_or_val = 'train',
    file_name = 'split1.txt',
    save_or_not = True
    ):
    return _gen_split(
        list(map(lambda i: ALL_CLASSES[i], split[1])), 
        file_name = file_name
    )

def gen_split2(
    voc_path = config.DATA_VOC,
    train_or_val = 'train',
    file_name = 'split2.txt',
    save_or_not = True
    ):
    return _gen_split(
        list(map(lambda i: ALL_CLASSES[i], split[2])), 
        file_name = file_name
    )

def gen_split3(
    voc_path = config.DATA_VOC,
    train_or_val = 'train',
    file_name = 'split3.txt',
    save_or_not = True
    ):
    return _gen_split(
        list(map(lambda i: ALL_CLASSES[i], split[3])), 
        file_name = file_name
    )

def gen_split4(
    voc_path = config.DATA_VOC,
    train_or_val = 'train',
    file_name = 'split4.txt',
    save_or_not = True
    ):
    return _gen_split(
        list(map(lambda i: ALL_CLASSES[i], split[4])), 
        file_name = file_name
    )



'''
The public `gen_split` interface
'''
def gen_split(
    i: int,
    voc_path = config.DATA_VOC,
    train_or_val = 'train',
    file_name = 'split.txt',
    save_or_not = True
    ):
    assert (i >= 0 and i < 5), "Error while generating splits: invalid split number: {}".format(i)
    file_name = "split{}.txt".format(i)
    return _gen_split(
        list(map(lambda i: ALL_CLASSES[i], split[i])), 
        file_name = file_name
    )
