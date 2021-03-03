import os
import os.path as ospx
import torch

split = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
RESTORE_FROM_WHERE = "pretrained"
EMBEDDING = "all"
lambdaa = 0.2
#USE_CPU = True
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 9
NUM_WORKERS = 3
ITER_SIZE = 1
IGNORE_LABEL = 255 # the background
INPUT_SIZE = "512,512"
LEARNING_RATE = 1e-4
MOMENTUM = 0.9
NUM_CLASSES = 21
NUM_EPOCHS = 50
POWER = 0.9
RANDOM_SEED = 1234
SAVE_NUM_IMAGES = 2
SAVE_PRED_EVERY = 500
WEIGHT_DECAY = 0.0005
LOG_DIR = "./log"
weak_size = BATCH_SIZE
weak_proportion = 0.2

DATA_PATH = "dataset/"
PRETRAINED_OUR_PATH = "model/segmentation/pretrained/our_qfsl_confidence"
SNAPSHOT_PATH = "model/segmentation/snapshots/vgg/lambda_split_single_1"
PATH = "output/"


DATA_VOC = DATA_PATH + "voc2012/"
DATA_SEM = DATA_PATH # Semantic embeddings path
SNAPSHOT_DIR = PATH + SNAPSHOT_PATH + "/" + EMBEDDING
RESULT_DIR = PATH + SNAPSHOT_PATH + "/" + "result.txt"
