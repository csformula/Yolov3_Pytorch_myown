from __future__ import division

import cv2
import os
import os.path as osp

images_names = os.listdir('original')
images_dirs = [osp.join(osp.realpath('original'), image) for image in images_names]

images_bgr = [cv2.imread(image) for image in images_dirs]

def resizing(image):
    h, w = image.shape[0], image.shape[1]
    factor = min(608/h, 608/w)
    newh, neww = int(h*factor), int(w*factor)
    return cv2.resize(image, (neww, newh))
images_resized = list(map(resizing, images_bgr))

det_names = [f'{image}' for image in images_names]
list(map(cv2.imwrite, det_names, images_resized))
    