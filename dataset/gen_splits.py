# -*- coding: utf-8 -*-
import sys

import os
import shutil
from PIL import Image
from transform_pixel import *
#from dataset.util import *
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from typing import List
import numpy as np
from PIL import Image

sys.path.append('../')
import config


'''
All classes in dataset VOC2012 (including 21 classes if background is counted)
'''
ALL_CLASSES = [
    "bg",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor"
]

split = [
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], # split 0
    [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],                # split 1
    [1, 2, 3, 4, 5, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],                 # split 2
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 16, 17, 18, 19, 20],                     # split 3
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]                      # split 4
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


def _gen_split(classes,
              voc_path=config.DATA_VOC,
              train_or_val='train',
              file_name='split.txt',
              save_or_not=True
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

    print("Generated classes: {}, \nGenerated images: {}".format(len(ans['classes']), len(ans['files'])))
    return ans


def gen_split0(voc_path=config.DATA_VOC,
               train_or_val='train',
               file_name='split.txt',
               save_or_not=True):
    return _gen_split(list(map(lambda i: ALL_CLASSES[i], split[0])))

def gen_split1(voc_path=config.DATA_VOC,
               train_or_val='train',
               file_name='split.txt',
               save_or_not=True):
    return _gen_split(list(map(lambda i: ALL_CLASSES[i], split[1])))

def gen_split2(voc_path=config.DATA_VOC,
               train_or_val='train',
               file_name='split.txt',
               save_or_not=True):
    return _gen_split(list(map(lambda i: ALL_CLASSES[i], split[2])))

def gen_split3(voc_path=config.DATA_VOC,
               train_or_val='train',
               file_name='split.txt',
               save_or_not=True):
    return _gen_split(list(map(lambda i: ALL_CLASSES[i], split[3])))

def gen_split4(voc_path=config.DATA_VOC,
               train_or_val='train',
               file_name='split.txt',
               save_or_not=True):
    return _gen_split(list(map(lambda i: ALL_CLASSES[i], split[4])))



'''
The public `gen_split` interface
'''
def gen_split(i: int,
              voc_path=config.DATA_VOC,
              train_or_val='train',
              file_name='split.txt',
              save_or_not=True):
    assert (i >= 0 and i < 5), "Error while generating splits: invalid split number: {}".format(i)
    return _gen_split(list(map(lambda i: ALL_CLASSES[i], split[i])))
