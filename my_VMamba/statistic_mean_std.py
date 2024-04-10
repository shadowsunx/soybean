import os
import glob
import random
import shutil
import numpy as np
from PIL import Image
import cv2


""" 统计数据库中所有图片的每个通道的均值和标准差 """
if __name__ == '__main__':

    dataset_train_dir = "/data/ldz/images/images"
    train_files = glob.glob(os.path.join(dataset_train_dir, '*.jpg'))

    print(f'Totally {len(train_files)} files for training')
    result = []
    for file in train_files:
        img = Image.open(file).convert('RGB')
        img = np.array(img).astype(np.uint8)
        img = img/255.
        result.append(img)
    
    print(np.shape(result))
    mean = np.mean(result, axis=(0, 1, 2))
    std = np.std(result, axis=(0, 1, 2))
    print(mean)
    print(std)

