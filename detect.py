from __future__ import division
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2
from util import *
import argparse
import os
import os.path as osp
from darknet import Darknet
import pickle as pkl
#import pandas as pd
import random
#import matplotlib.pyplot as plt

def arg_parse():
    '''  
    Parse arguements to the detect module '''
    
    parser = argparse.ArgumentParser(description = 'Yolo v3 Detection Module')
    parser.add_argument('--images', dest = 'images', help =
                        'Image / Directory containing images to perform detection upon',
                        default = 'images', type = str)
    parser.add_argument('--det_dir', dest = 'det_dir', help = 
                        'Image / Directory to store detections to',
                        default = 'det', type = str)
    parser.add_argument('--bs', dest = 'bs', help = 'Batch Size', default = 1, type = int)
    parser.add_argument('--confidence', dest = 'confidence', help = 'Object confidence to filter predictions', default = 0.5, type = float)
    parser.add_argument('--nms_threshold', dest = 'nms_threshold', help = 'NMS Threshold', default = 0.4, type = float)
    parser.add_argument('--cfg', dest = 'cfg', help = 
                        'Config file',
                        default = 'cfg/yolov3.cfg', type = str)
    parser.add_argument('--weights', dest = 'weightsfile', help =
                        'Weights file',
                        default = 'yolov3.weights', type = str)
    parser.add_argument('--reso', dest = 'reso', help =
                        'Input resolution of the network. Increase to increase accuracy. Decrease to increase speed',
                        default = 416, type = int)
    
    return parser.parse_args()

args = arg_parse()
images = args.images
batch_size = args.bs
confidence = args.confidence
nms_threshold = args.nms_threshold
start = 0
CUDA = torch.cuda.is_available()

classes = load_classes('data/coco.names')
num_classes = len(classes)

#Set up the neural network
print('Loading network.....')
model = Darknet(args.cfg)
model.load_weights(args.weightsfile)
print('Network successfully loaded!')

model.net_info['height'] = args.reso
inp_dim = model.net_info['height']
assert inp_dim % 32== 0
assert inp_dim > 32

#If there's a GPU availible, put the model on GPU
if CUDA:
    model.cuda()
    
#Set the model in evaluation mode
model.eval()

# checkpoint for read images
read_dir_ckp = time.time()
# Detection phase
# imlist saves image file names, str
try:
    imlist = [osp.join(os.path.realpath('.'), images, img) for img in os.listdir(images) if img[-3:]=='jpg']
except NotADirectoryError:
    imlist = []
    imlist.append(osp.join(os.path.realpath('.'), images))
except FileNotFoundError:
    print (f'No file or directory with the name {images}')
    exit()
if not osp.exists(args.det_dir):
    os.makedirs(args.det_dir)
# checkpoint for load images with cv2
load_batch_ckp = time.time()
# load images with cv2, and saved in a list
loaded_imgs_bgr = [cv2.imread(img) for img in imlist]
loaded_imgs = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in loaded_imgs_bgr]
#plt.imshow(resize_image(loaded_imgs[0], inp_dim))

# PyTorch Tensors for images
im_tensors = list(map(prep_image, loaded_imgs, [inp_dim for i in range(len(loaded_imgs))]))
# List containing dimensions of original images
im_dims_list = [(img.shape[0], img.shape[1]) for img in loaded_imgs]
im_dims_list = torch.FloatTensor(im_dims_list)
# split image-tensors into batches
im_batches = prep_batches(im_tensors, batch_size)

# loop to detect
first_detection = True
start_loop_ckp = time.time()
for i, batch in enumerate(im_batches):
    start = time.time()
    if CUDA:
        batch = batch.cuda()
    # disable gradient calculation
    with torch.no_grad():
        prediction = model(batch, CUDA)
        pred_bbox = write_results(prediction, confidence, num_classes, nms_threshold)
    end = time.time()
    # output detection info
    for batch_i, img_name in enumerate(imlist[i*batch_size: i*batch_size+batch.size(0)]):
        img_name = img_name.split('/')[-1]
        img_i = batch_i + i*batch_size
        each_time = (end-start)/batch.size(0)
        print('{:<20s} predicted in {:^7.3f} seconds'.format(img_name.split("\\")[-1], each_time))
        if isinstance(pred_bbox, int):  # no bbox was found
            print(f'{"Objects Detected:":<20s} None')
        else:
            objects = [classes[int(bbox[-1])] for bbox in pred_bbox if bbox[0] == batch_i]
            print(f'{"Objects Detected:":<20s} {" ".join(objects)}')
        print("----------------------------------------------------------")
    # record detections
    if not isinstance(pred_bbox, int):
        pred_bbox[:,0] += i*batch_size  #transform the attribute from index in batch to index in imlist
        if first_detection:
            all_bbox = pred_bbox
            first_detection = False
        else:
            all_bbox = torch.cat((all_bbox, pred_bbox))
    if CUDA:
        torch.cuda.synchronize()

# Drawing bounding boxes on images
# try to see if there were detections
try:
    all_bbox
except NameError:
    print('No Detections were made!')
    exit()
# transform coordinates of bbox to match original image size
im_dims_list = torch.index_select(im_dims_list, 0, all_bbox[:,0].long())
scale_factor = torch.min(inp_dim/im_dims_list, 1, keepdim = True)[0]
all_bbox[:,[1,3]] -= (inp_dim - im_dims_list[:,1].unsqueeze(1)*scale_factor)/2
all_bbox[:,[2,4]] -= (inp_dim - im_dims_list[:,0].view(-1,1)*scale_factor)/2
all_bbox[:,1:5] /= scale_factor
# clip bounding boxes that have boundaries outside the image
for i in range(all_bbox.size(0)):
    all_bbox[i, [1,3]] = torch.clamp(all_bbox[i, [1,3]], 0.0, im_dims_list[i,1])
    all_bbox[i, [2,4]] = torch.clamp(all_bbox[i, [2,4]], 0.0, im_dims_list[i,0])
# draw bbox, each bbox each color
class_load_ckp = time.time()
with open('pallete', 'rb') as f:
    colors = pkl.load(f)
draw_ckp = time.time()
def draw_bbox(bbox, results):
    c1 = tuple(bbox[1:3].int())
    c2 = tuple(bbox[3:5].int())
    img = results[int(bbox[0])]
    cls = int(bbox[-1])
    color = random.choice(colors)
    label = f'{classes[cls]}'
    cv2.rectangle(img, c1, c2, color, 2)
    size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)
    c3 = (c1[0], c1[1]-size[1])
    c4 = (c1[0]+size[0]+3, c1[1])
    cv2.rectangle(img, c3, c4, color, -1)
    cv2.putText(img, label, c1, cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255))
    return img
# modifies the images inside loaded_ims inplace
list(map(lambda x: draw_bbox(x, loaded_imgs_bgr), all_bbox))  
# list of directories of our detection images
det_dirlist = [osp.join(args.det_dir, 'det_{}'.format(name.split("\\")[-1])) for name in imlist]
# write the images
list(map(cv2.imwrite, det_dirlist, loaded_imgs_bgr))
end_ckp = time.time()

# Print summary
print('SUMMARY')
print("----------------------------------------------------------")
print(f'{"Task":<25s}: Time Taken (in seconds)')
print('')
print('{:<25s}: {:>6.3f}'.format("Reading addresses", load_batch_ckp-read_dir_ckp))
print('{:<25s}: {:>6.3f}'.format("Loading batch", start_loop_ckp-load_batch_ckp))
print('{:<25s}: {:>6.3f}'.format("Detection ("+str(len(imlist))+" images)", class_load_ckp-start_loop_ckp)) 
print('{:<25s}: {:>6.3f}'.format("Colors loading", draw_ckp-class_load_ckp))   
print('{:<25s}: {:>6.3f}'.format("Drawing Boxes", end_ckp-draw_ckp))
print('{:<25s}: {:>6.3f}'.format("Average time_per_img", (end_ckp-load_batch_ckp)/len(imlist))) 
print("----------------------------------------------------------")

torch.cuda.empty_cache()

    
    
    
    
    
    
    
    
    
    
