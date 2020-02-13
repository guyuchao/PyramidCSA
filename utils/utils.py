from tensorboardX import SummaryWriter
import logging
import os
import torch
from PIL import Image
from math import ceil

def get_Logger_and_SummaryWriter():
    for i in range(1000):
        tag_dir = 'checkpoints/tensorboard/try_{}'.format(i)
        if not os.path.exists(tag_dir):
            os.makedirs(tag_dir, exist_ok=True)
            logger = logging.getLogger("PGGAN")
            file_handler = logging.FileHandler(os.path.join(tag_dir, 'log.txt'), "w")
            stdout_handler = logging.StreamHandler()
            logger.addHandler(file_handler)
            logger.addHandler(stdout_handler)
            stdout_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
            file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
            logger.setLevel(logging.INFO)
            return logger, SummaryWriter(tag_dir),tag_dir

def safe_save(img,save_path):
    os.makedirs(save_path.replace(save_path.split('/')[-1],""),exist_ok=True)
    img.save(save_path)
