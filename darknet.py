from __future__ import division
from util import *

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

def get_test_input(dim):
    img = cv2.cv2.imread('dog-cycle-car.png')
    img = cv2.resize(img, (dim, dim))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255
    img_tensor = torch.from_numpy(img).float().permute(2,0,1).contiguous().unsqueeze(0)
#     img_tensor_ = Variable(img_tensor)
    return img_tensor

def parse_cfg(cfgfile):
    """
    Takes a configuration file
    Returns a list of blocks. Each blocks describes a block in the neural
    network to be built. Block is represented as a dictionary in the list
    """
    file = open(cfgfile, 'r')
    lines = file.readlines()
    lines = [x for x in lines if x!='\n']
    lines = [x for x in lines if x[0]!='#']
    lines = [x.strip() for x in lines]
    
    blocks = []
    block = {}
    for line in lines:
        if line[0] == '[':
            if block:                   # if block is not empty
                blocks.append(block)
                block = {}
            block['type'] = line[1:-1]
        else:
            key, value = line.split('=')
            block[key.strip()] = value.strip()
    blocks.append(block)
    
    return blocks,lines

def create_modules(blocks):
    net_info = blocks[0]                # Captures the information about the input and pre-processing 
    module_list = nn.ModuleList()
    prev_filters = 3                    # keep track of number of filters in the layer on which the convolutional layer is being applied
    output_filters = []                 # keep a track of the number of filters in each one of the preceding layers
    
    for index, block in enumerate(blocks[1:]):
        module = nn.Sequential()
        
        #check the type of block
        #create a new module for the block
        #append to module_list
         
        # if it's a conv layer
        if block['type'] == 'convolutional':
            activation = block['activation']
            filters = int(block['filters'])
            stride = int(block['stride'])
            size = int(block['size'])
            pad = int(block['pad'])            # it's 1 or 0, decide whether we pad the feature maps
            if pad:
                padding = (size-1) // 2
            else:
                padding = 0           
            try:
                batch_normalize = int(block['batch_normalize'])
                bias = False
            except:
                batch_normalize = 0
                bias = True
       
            # add the conv layer
            conv = nn.Conv2d(prev_filters, filters, size, stride, padding, bias = bias)
            module.add_module(f'conv_{index}', conv)            
            # add batch-norm layer
            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                module.add_module(f'batch_norm_{index}', bn)                
            # add activation layer
            if activation == 'leaky':
                an = nn.LeakyReLU(0.1, True)
                module.add_module(f'leaky_{index}', an)
        
        # if it's a upsample layer
        elif block['type'] == 'upsample':
            stride = int(block['stride'])
            upsample = nn.Upsample(scale_factor = stride, mode = 'nearest')
            module.add_module(f'upsample_{index}', upsample)
            
        # if it's a route layer
        elif block['type'] == 'route':
            layers = block['layers'].split(',')
            if len(layers)==1:
                filters = output_filters[index+int(layers[0])]
            else:
                filters = output_filters[index+int(layers[0])] + output_filters[int(layers[1])]
            route = EmptyLayer()
            module.add_module(f'route_{index}', route)
        
        # shortcut layer
        elif block['type'] == 'shortcut':
            shortcut = EmptyLayer()
            module.add_module(f'shortcut_{index}', shortcut)            
            
        # Yolo layer, it is the detection layer
        elif block['type'] == 'yolo':
            mask = [int(x) for x in block['mask'].split(',')]
            anchors = [int(x) for x in block['anchors'].split(',')]
            anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in mask]
            detection = DetectionLayer(anchors)
            module.add_module(f'detection_{index}', detection)
        
        # update prev_filters and output_filters
        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)
        
    return net_info, module_list
        
class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()
    
class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors
    
class Darknet(nn.Module):
    def __init__(self, cfgfile):
        super(Darknet, self).__init__()
        self.blocks, _ = parse_cfg(cfgfile)
        self.net_info, self.modulelist = create_modules(self.blocks)
    
    def forward(self, x, CUDA):        # x stands for inputs(images)
        modules = self.blocks[1:]
        outputs = {}                   # We cache the output of each layer
        fisrt_detection = True                      

        for index, module in enumerate(modules):
            module_type = module['type']
            if module_type == 'convolutional' or module_type == 'upsample':
                x = self.modulelist[index](x)
            
            elif module_type == 'route':
                layers = [int(x) for x in module['layers'].split(',')]
                if len(layers) == 1:
                    x = outputs[index + layers[0]]
                else:
                    output1 = outputs[index + layers[0]]
                    output2 = outputs[layers[1]]
                    x = torch.cat((output1, output2), 1)
                    
            elif module_type == 'shortcut':
                if module['activation'] == 'linear':
                    x = x + outputs[index + int(module['from'])]
            
            elif module_type == 'yolo':
                anchors = self.modulelist[index][0].anchors
                num_classes = int(module['classes'])
                inp_size = int(self.net_info['height'])
                x = x.data
                x = predict_transform(x, inp_size, anchors, num_classes, CUDA)
                if fisrt_detection:
                    detections = x
                    fisrt_detection = False
                else:
                    detections = torch.cat((detections, x), 1)
            # update outputs
            outputs[index] = x

        return detections
    
    def load_weights(self, weightsfile):
        #Open the weights file
        with open(weightsfile, 'rb') as wf:        
            #The first 5 values are header information 
            # 1. Major version number
            # 2. Minor Version Number
            # 3. Subversion number 
            # 4,5. Images seen by the network (during training)
            weight_header = np.fromfile(wf, dtype=np.int32, count=5)
            self.weight_header = torch.from_numpy(weight_header)
            self.seen = self.weight_header[3]
            weights = np.fromfile(wf, dtype=np.float32)
        
        # load weights to network
        ptr = 0
        for i in range(len(self.modulelist)):
            moduletype = self.blocks[i+1]['type']
            #If module_type is convolutional load weights
            #Otherwise ignore.
            if moduletype == 'convolutional':
                module = self.modulelist[i]
              
                try:       # there is a bn layer
                    num_bn_bias = module[1].state_dict()['bias'].numel()
                    bn_bias = torch.from_numpy(weights[ptr:ptr+num_bn_bias])
                    ptr += num_bn_bias
                    bn_weight = torch.from_numpy(weights[ptr:ptr+num_bn_bias])
                    ptr += num_bn_bias
                    bn_running_mean = torch.from_numpy(weights[ptr:ptr+num_bn_bias])
                    ptr += num_bn_bias
                    bn_running_var = torch.from_numpy(weights[ptr:ptr+num_bn_bias])
                    ptr += num_bn_bias
                    
                    bn_bias = bn_bias.view_as(module[1].state_dict()['bias'])
                    bn_weight = bn_weight.view_as(module[1].state_dict()['weight'])
                    bn_running_mean = bn_running_mean.view_as(module[1].state_dict()['running_mean'])
                    bn_running_var = bn_running_var.view_as(module[1].state_dict()['running_var'])
                    
                    module[1].state_dict()['bias'].copy_(bn_bias)
                    module[1].state_dict()['weight'].copy_(bn_weight)
                    module[1].state_dict()['running_mean'].copy_(bn_running_mean)
                    module[1].state_dict()['running_var'].copy_(bn_running_var)
                
                except:    # there is NOT a bn layer, but conv bias
                    num_conv_bias = module[0].state_dict()['bias'].numel()
                    conv_bias = torch.from_numpy(weights[ptr:ptr+num_conv_bias])
                    ptr += num_conv_bias
                    
                    conv_bias = conv_bias.view_as(module[0].state_dict()['bias'])
                    
                    module[0].state_dict()['bias'].copy_(conv_bias)                 
                    
                # conv layer's weight
                num_conv_weight = module[0].state_dict()['weight'].numel()
                conv_weight = torch.from_numpy(weights[ptr:ptr+num_conv_weight])
                ptr += num_conv_weight

                conv_weight = conv_weight.view_as(module[0].state_dict()['weight'])

                module[0].state_dict()['weight'].copy_(conv_weight)

        return
        
    
    
    
    
    
    
    
    