import json, os
import copy
import numpy as np
import cv2
import math

from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask
from itertools import groupby
from skimage import measure
from operator import itemgetter
from tqdm import tqdm

class predictionProcessor:
    _DEBUGGING = False
    _VERBOSE = False

    def __init__(self, dataset_name):
        pass
