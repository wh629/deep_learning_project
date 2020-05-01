"""
Module to define our model architecture
"""

import torch
import torch.nn as nn
import torchvision
import os

class Model(nn.Module):
    def __init__(self,
                 road_lambda = 1,          # relative weight of road map loss
                 box_lambda  = 1,          # relative weight of bounding box loss
                 preload_backbone = False, # whether to load pretrained weights
                 backbone_weights = None,  # pretrained backbone weights if needed
                 ):
        
        """
        Model initializer
        """
        super(Model, self).__init__()
        # TODO: double check how num_classes is used
        self.image_detect = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False,
                                                                                 progress=True,
                                                                                 num_classes=9,
                                                                                 pretrained_backbone=False)
        # comment: not sure if transform needed because already handeled in data_helper
        #self.transform = self.image_detect.transform
        self.backbone = self.image_detect.backbone
        
        # to prepare for pretraining
        if preload_backbone:
            assert os.path.exists(backbone_weights), "No backbone weights to load. {}".format(backbone_weights)
            self.backbone.load_state_dict(torch.load(backbone_weights))

        #TODO get into right output shape
        # comment: if you make it [N, 800, 800, 2] and use CrossEntropyLoss() it will make it easier to make a decision whether it's true or false
        #           for each cell we can just take the max of the 2 logits
        self.road = nn.Sequential(
            nn.Conv2d(...)
            nn.ReLU()
            ...
        )

        #TODO doublecheck road loss
        #self.road_loss = nn.BCEWithLogitsLoss()
        self.road_loss = nn.CrossEntropyLoss()

    def forward(self,
                images = None,
                box_targets = None,
                road_targets = None,
                ):
        """
        Forward
        ----------------------
        Returns:
        List with 
            0) First element as loss
            1) Second element as predicted bounding boxes
            2) Third element as road map

        """
        # TODO: group images and resize to 800x800
        # samples is a cuda tensor with size [batch_size, 6, 3, 256, 306]
        rescaled_images = ...

        # TODO: repackage labels from [N, 2, 4] to [N, 4]
        repackaged_box_labels = ...

        # do image detection

        # TODO: get box loss
        # output includes losses
        # https://github.com/pytorch/vision/blob/master/torchvision/models/detection/generalized_rcnn.py
        out = self.image_detect(rescaled_images, repackaged_box_targets)
        box_loss = ...

        # TODO: get predicted box losses from output and repackage for output
        pred_boxes = ...

        # do road map
        # TODO: double check
        # comment: not sure if transform needed since images already transformed in data_helper
        #images, targets = self.transform(images, None)
        features = self.backbone(images.tensors)
        roads = self.road(features)

        # should be [N, 800, 800, 2]
        batch_size, width, height, logits = roads.shape

        # TODO: cast road_targets from boolean TRUE/FALSE to Integers 1/0
        road_labels = ...

        road_loss = 0
        for i in range(batch_size):
            # can apply CrossEntropyLoss to unpacked road_map
            # https://pytorch.org/docs/stable/nn.html#crossentropyloss
            single_road = roads[i,:,:,:].squeeze().view(width*height, logits)
            single_road_labels = road_labels[i,:,:].squeeze().view(width*height)
            road_loss += self.road_loss(single_road, sing_road_labels)
        road_loss = road_loss.mean()

        #TODO: Calculate losses if applicable
        loss = self.road_lambda*road_loss + self.box_lambda*box_loss

        #TODO: repackage bounding boxes
        boxes = ...

        #TODO: make road map binary values, then to boolean TRUE/FALSE
        # not sure if this works but check documentation here
        # https://pytorch.org/docs/stable/torch.html#comparison-ops
        values, b_roads = torch.max(roads, dim = 3)
        # not sure if this works but check documentation here
        # https://pytorch.org/docs/stable/tensors.html#torch.Tensor.to
        b_roads = b_roads.bool()

        out = []
        out.append(loss)
        out.append(boxes)
        out.append(b_roads)
        out.append(road_loss)
        out.append(box_loss)

        return out
