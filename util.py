from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import cv2

def predict_transform(prediction, inp_dim, anchors, num_classes, CUDA = True):
    batch_size = prediction.size(0)
    stride = inp_dim // prediction.size(2)
    grid_size = inp_dim // stride
    num_anchors = len(anchors)
    bbox_attrs = 5 + num_classes
    
    prediction = prediction.view(batch_size, num_anchors*bbox_attrs, grid_size*grid_size)
    prediction = prediction.transpose(1,2).contiguous()   
    prediction = prediction.view(batch_size, grid_size*grid_size*num_anchors, bbox_attrs)
#    print(prediction[0,0,:5])
    anchors = [(anchor[0]/stride, anchor[1]/stride) for anchor in anchors]
    
    # Sigmoid the  centre_X, centre_Y. and object confidencce
    prediction[:,:,:2] = torch.sigmoid(prediction[:,:,:2])
    prediction[:,:,4] = torch.sigmoid(prediction[:,:,4])
    
    # Add the center offsets
    grid = np.arange(grid_size)
    x, y = np.meshgrid(grid, grid)
    x = torch.FloatTensor(x).view(-1,1)
    y = torch.FloatTensor(y).view(-1,1)
    if CUDA:
        x = x.cuda()
        y = y.cuda()
    offset = torch.cat((x,y), 1).repeat(1, num_anchors).view(-1,2)
    offset = offset.unsqueeze(0)
    prediction[:,:,:2] += offset
    
    # log space transform height and the width
    anchors = torch.FloatTensor(anchors)
    if CUDA:
        anchors = anchors.cuda()
    anchors = anchors.repeat(grid_size*grid_size, 1)
    anchors = anchors.unsqueeze(0)
    prediction[:,:,2:4] = torch.exp(prediction[:,:,2:4]) * anchors
    
    # Apply sigmoid activation to the the class scores
    prediction[:,:,5:] = torch.sigmoid(prediction[:,:,5:])
    
    # resize the detections map to the size of the input image
    prediction[:,:,:4] *= stride
    
    return prediction
    
def write_results(prediction, confidence, num_classes, nms_conf=0.4):
    # set zero if bbox has a objectness score below confidence 
    mask_conf = (prediction[:,:,4]>confidence).float().unsqueeze(2)
    prediction = prediction * mask_conf
    
    # convert bbox parameters to corners' coordinate
    bbox_corners = prediction[:,:,:4]
    bbox_corners[:,:,0] = prediction[:,:,0] - prediction[:,:,2]/2       # top-left x
    bbox_corners[:,:,1] = prediction[:,:,1] - prediction[:,:,3]/2       # top-left y
    bbox_corners[:,:,2] = prediction[:,:,0] + prediction[:,:,2]       # bottom-right x
    bbox_corners[:,:,3] = prediction[:,:,1] + prediction[:,:,3]       # bottom-right y
    prediction[:,:,:4] = bbox_corners
    
    # loop over each image
    batch_size = prediction.size(0)
    first_output = True
    for i in range(batch_size):
        # for each image, shrink predictions to [bbox, confidence, max_class_score, max_class]
        # dims from (10647, 85) to (10647, 7)
        each_pred = prediction[i]
        max_classconf, max_classconf_index = torch.max(each_pred[:,5:], 1, keepdim=True)
        max_classconf_index = max_classconf_index.float()
        each_pred = torch.cat((each_pred[:,:5], max_classconf, max_classconf_index), 1)
        # confidence threshholding
        nonzero_index = torch.nonzero(each_pred[:,4])
        if nonzero_index.size(0) == 0:
            continue
        each_pred_ = each_pred[nonzero_index.squeeze(1),:]
        
        # NMS
        img_classes = unique(each_pred_[:,-1])
        for cls in img_classes:
            # pick cls
            cls_mask = each_pred_*((each_pred_[:, -1] == cls).view(-1,1)).float()
            tmp_nonzero = torch.nonzero(cls_mask[:,-2]).squeeze(1)
            class_pred = cls_mask[tmp_nonzero, :]
            
            sort_index = torch.sort(class_pred[:,-2], descending=True)[1]
            sort_class_pred = class_pred[sort_index, :]
            num_pred = sort_class_pred.size(0)

            # loop to get rid of bboxes which iou larger than nms_conf
            for j in range(num_pred):
                if j >= sort_class_pred.size(0)-1:
                    break               
                iou = bbox_iou(sort_class_pred[j].unsqueeze(0), sort_class_pred[j+1:])
                iou_mask = (iou < nms_conf).float()
                sort_class_pred[j+1:,:] = sort_class_pred[j+1:,:]*iou_mask
                nonzero_bbox = torch.nonzero(sort_class_pred[:,-2]).squeeze(1)
                sort_class_pred = sort_class_pred[nonzero_bbox, :]
            
            # write outputs
            image_index = sort_class_pred.new_empty(sort_class_pred.size(0), 1).fill_(i)
            class_output = torch.cat((image_index, sort_class_pred), 1)
            if first_output:
                outputs = class_output
                first_output = False
            else:
                outputs = torch.cat((outputs, class_output), 0)
    try:            
        return outputs
    except:
        return 0

# return unique detection-classes on each image
def unique(tensor):
    np_tensor = tensor.cpu().numpy()
    np_unique = np.unique(np_tensor)
    unique_tensor = torch.from_numpy(np_unique)
    # make sure returned tensor has same dtype and device with input tensor
    tensor_res = tensor.new_empty(unique_tensor.size())
    tensor_res.copy_(unique_tensor)
    return unique_tensor

# return iou of box1 and box2
# box1 has one bbox, box2 has one or more bbox which have smaller class scores than box1
def bbox_iou(box1, box2):      
    ''' return a pytorch.tensor including all ious between bbox in box1(one bbox) and 
        bbox in box2(one or more bbox)'''
    inter_topx = torch.max(box1[:,0], box2[:,0])
    inter_topy = torch.max(box1[:,1], box2[:,1])
    inter_bottomx = torch.min(box1[:,2], box2[:,2])
    inter_bottomy = torch.min(box1[:,3], box2[:,3])
    
    inter_area = torch.clamp(inter_bottomx-inter_topx, min=0) * torch.clamp(inter_bottomy-inter_topy, min=0)
    bbox_area_sum = (box1[:,2]-box1[:,0])*(box1[:,3]-box1[:,1]) + (box2[:,2]-box2[:,0])*(box2[:,3]-box2[:,1])
    union_area = bbox_area_sum - inter_area
    
    iou = (inter_area/union_area).view(-1,1)
    return iou

# return a list of coco's names of objects
def load_classes(filename):
    with open(filename, 'r') as f:
        names = f.read().split('\n')
    return names    
    
# resize image, keeping the aspect ratio consistent, and padding the left-out areas
def resize_image(image, inp_size):
    imgh, imgw = image.shape[:2]
    h, w = inp_size, inp_size 
    newh, neww = int(imgh*min(h/imgh, w/imgw)), int(imgw*min(h/imgh, w/imgw))
    new_img = cv2.resize(image, (neww, newh), interpolation = cv2.INTER_CUBIC)
    resized_img = np.zeros((h, w, 3),dtype=int) + 128 
    resized_img[(h-newh)//2:(h-newh)//2+newh, (w-neww)//2:(w-neww)//2+neww] = new_img
    return resized_img

# Prepare image for inputting to the neural network. return a Tensor
def prep_image(image, inp_dim):
    resized_img = resize_image(image, inp_dim)
    img_tensor = (torch.from_numpy(resized_img).float()/255.0).permute(2,0,1).unsqueeze(0)
    return img_tensor
    
# split image-tensors into batches
def prep_batches(img_tensors, batch_size):
    num_batches = len(img_tensors) // batch_size
    if len(img_tensors) % batch_size:
        num_batches += 1     
    img_batches = list(torch.chunk(torch.cat(img_tensors), num_batches))
    return img_batches
    
    
    
    
    
    
    
    
    
    
    
    
    