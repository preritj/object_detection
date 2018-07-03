from collections import defaultdict
import numpy as np
import os
import xml.etree.ElementTree as ET
from glob import glob
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from dataset.object_data import ObjectData


class AVAretail(ObjectData):
    def __init__(self, cfg, data_dir, train_files):
        self.cfg = cfg
        super().__init__(cfg, data_dir, train_files)

    def _build_dataset(self, dataset):
        data_dir = os.path.join(self.data_dir, "")
        for i, data in enumerate(dataset.items()):
            img_id = i
            img_file, ann_file = data
            img_file = img_file.split(data_dir)[1]
            h, w = self.cfg.image_shape
            self.imgs[img_id] = {'filename': img_file,
                                 'shape': [h, w]}
            tree = ET.parse(ann_file)
            root = tree.getroot()
            for obj in root:
                obj_name = obj.find('name').text
                label = self.product_labels[obj_name]
                bbox = obj.find('bndbox')
                x1, x2, y1, y2 = [int(bbox[i].text) for i in range(4)]
                bbox = [y1 / h, x1 / w, y2 / h, x2 / w]
                self.anns[img_id].append({'bbox': bbox,
                                          'label': label})

    def create_index(self):
        # create index
        print('creating index...')
        self.imgs, self.anns = {}, defaultdict(list)

        for dataset in self.datasets:
            self._build_dataset(dataset)

        print('index created!')
        self.ids = list(self.anns.keys())
