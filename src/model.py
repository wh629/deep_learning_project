"""
Module to define our model architecture
"""

import torch
import torch.nn as nn
import torchvision
import os

class Model(nn.Module):
    def __init__(self):
        
        """
        TO DO
        """
        super(Model, self).__init__()
        self.image_w = image_w
        self.road_w = road_w        
        
        self.image_detect = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, progress=True, num_classes=9, pretrained_backbone=False)
        self.transform = self.image_detect.transform
        self.backbone = self.image_detect.backbone
        
        #TODO get into right output shape
        self.road = nn.Sequential(
            nn.Conv2d(...)
            nn.ReLU()
            ...
        )

    def forward(self,
                images = None,
                box_targets = None,
                ):
        """
        Forward
        
        Returns output as specified below

        """
        # do image detection
        out_box = self.image_detect(images, box_targets)
        
        # do road map
        images, targets = self.transform(images, None)
        features = self.backbone(images.tensors)
        out_road = self.road(features)
    
        return out_box, out_road
