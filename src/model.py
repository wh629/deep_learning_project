"""
Module to define our model architecture
"""

import torch
import torch.nn as nn
import torchvision
import os
import logging as log

def concat_6_images(sample: torch.Tensor):
    """
    Combine a stack of 6 images into a big image of 2 rows and 3 columns.

    The sample.shape = (6, 3, H, W), while output.shape = (3, 2H, 3W).

    Args:
        sample (Tensor): a sample of size (6, 3, H, W) that contains 6 images.
    """
    num_images_per_row = 3
    return torchvision.utils.make_grid(sample, nrow=num_images_per_row)


def repackage_image_batch(samples: torch.Tensor):
    """
    Repackage a batch of images from [batch_size, 6, 3, 256, 306] to [batch_size, 3, 800, 800]
    """
    samples = [concat_6_images(sample) for sample in samples]
    samples = torch.stack(samples, 0)
    samples = nn.functional.interpolate(samples, size=(800, 800), mode='nearest')  #resize
    return samples


def get_boxes_from_4_points_to_2_points(boxes: torch.Tensor):
    """
    N boxes in an image. The boxes.shape = [N, 2, 4]
    """

    # select two points
    points = boxes[:, :, [0, 3]]

    # convert meters to pixels
    points[:, 0] = points[:, 0] * 10 + 400
    points[:, 1] = - points[:, 1] * 10 + 400

    # from [N, 2, 2] to [N, 4]
    points = points.permute(0, 2, 1).reshape(-1, 4)

    return points


def get_boxes_from_2_points_to_4_points(boxes: torch.Tensor):
    """
    N boxes in an image. The boxes.shape = [N, 4]
    """
    t = boxes.reshape(-1, 2, 2).permute(0, 2, 1)
    t[:, 0] = (t[:, 0] - 400) / 10
    t[:, 1] = - (t[:, 1] - 400) / 10
    x_vals = torch.stack([t[:, 0, 0], t[:, 0, 0], t[:, 0, 1], t[:, 0, 1]], 1)
    y_vals = torch.stack([t[:, 1, 0], t[:, 1, 1], t[:, 1, 0], t[:, 1, 1]], 1)
    return torch.stack([x_vals, y_vals], 1)


def repackage_targets(targets):
    """
    Targets are a batch of individual targets.

    Repackage boxes from [batch_size, N, 2, 4] to [batch_size, N, 4]
    """
    repackaged = []
    for target in targets:
        target_new = {'boxes': get_boxes_from_4_points_to_2_points(target['bounding_box']).float(),
                      'labels': target['category'].to(torch.int64)}
        repackaged.append(target_new)

    return repackaged


class Model(nn.Module):
    def __init__(self,
                 road_lambda = 1,           # relative weight of road map loss
                 box_lambda  = 1,           # relative weight of bounding box loss
                 preload_backbone = False,  # whether to load pretrained weights
                 backbone_weights = None,   # pretrained backbone weights if needed
                 ):
        
        """
        Model initializer
        """
        super(Model, self).__init__()
        # weights for loss
        self.road_lambda = road_lambda
        self.box_lambda = box_lambda

        # for bounding box detection
        self.image_detect = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False,
                                                                                 progress=True,
                                                                                 num_classes=9,
                                                                                 pretrained_backbone=False)

        self.backbone = self.image_detect.backbone
        
        # to prepare for pretraining
        if preload_backbone:
            assert os.path.exists(backbone_weights), "No backbone weights to load. {}".format(backbone_weights)
            self.backbone.load_state_dict(torch.load(backbone_weights))
            log.info(f"Preloaded weights from {backbone_weights}")

        # gradient tracking for early layers
        for param in self.backbone.parameters():
            param.requires_grad = True

        # for road map
        self.double_dim_minus1 = nn.ConvTranspose2d(in_channels = 256, out_channels = 256, kernel_size = 3, stride = 2, padding = 1)
        self.relu = nn.ReLU()
        self.target_size = 800
        self.conv_256_1 = nn.Conv2d(in_channels = 256, out_channels = 1, kernel_size = 3, stride = 1, padding = 1)
        self.conv_5_1 = nn.Conv2d(in_channels = 5, out_channels = 1, kernel_size = 3, stride = 1, padding = 1)
        self.fc = nn.Conv2d(in_channels = 1, out_channels = 1, kernel_size = 1, stride = 1, padding = 0)
        self.sig = nn.Sigmoid()
        self.threshold = 0.5

        self.road_loss = nn.BCEWithLogitsLoss()

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
            out[0] First element as loss
            out[1] Second element as predicted bounding boxes (for evaluation. o.w. empty list)
            out[2] Third element as road map
            out[3] road_loss
            out[4] box_loss (for train. o.w. 0)

        """
        # repackage images from [batch_size, 6, 3, 256, 306] to [batch_size, 3, 800, 800]
        rescaled_images = repackage_image_batch(images)
        # repackage labels from [batch_size, N, 2, 4] to [batch_size, N, 4]
        repackaged_box_targets = repackage_targets(box_targets)

        # do image detection

        # https://github.com/pytorch/vision/blob/master/torchvision/models/detection/generalized_rcnn.py
        if self.training:
            rcnn_output = self.image_detect(rescaled_images, repackaged_box_targets)
            box_loss = rcnn_output['loss_box_reg']
            rcnn_boxes = []
        else:
            # rcnn_boxes is a list of dicts with keys = ['boxes', 'labels', 'scores']
            # each dict is for a sample in batch.
            rcnn_boxes = self.image_detect(rescaled_images)
            box_loss = 0


        # do road map

        features = self.backbone(rescaled_images)
        # The features is a OrderDict as follows:
        #     key: 0    -> value: a tensor of shape [2, 256, 200, 200]
        #     key: 1    -> value: a tensor of shape [2, 256, 100, 100]
        #     key: 2    -> value: a tensor of shape [2, 256, 50, 50]
        #     key: 3    -> value: a tensor of shape [2, 256, 25, 25]
        #     key: pool -> value: a tensor of shape [2, 256, 13, 13]

        features_new = []
        mapping_to = {13: 25, 25: 50, 50: 100, 100: 200, 200: 400, 400: 800}
        for key, feature in features.items():
            temp = feature
            bs, c, h, w = temp.shape
            while h < self.target_size:
                temp = self.relu(self.double_dim_minus1(temp, output_size = [mapping_to[h], mapping_to[h]]))
                bs, c, h, w = temp.shape
            features_new.append(self.relu(self.conv_256_1(temp)))

        #thin_feature = torch.stack(features_new, dim=1)
        thin_feature = torch.cat(features_new, dim=1)
        single_feature = self.relu(self.conv_5_1(thin_feature))
        roads = self.fc(single_feature)
        bs, c, h, w = roads.shape
        roads = roads.view((bs, h, w))

        if road_targets is not None:
            road_loss = self.road_loss(roads, torch.stack(road_targets, dim=0).float())
        else:
            road_loss = 0

        # Calculate losses if applicable
        loss = self.road_lambda*road_loss + self.box_lambda*box_loss

        # repackage bounding boxes
        boxes = [get_boxes_from_2_points_to_4_points(boxes_i['boxes']) for boxes_i in rcnn_boxes]
        boxes_tensor = torch.stack(boxes, dim=0)

        # repackage roads
        b_roads = self.sig(roads) > self.threshold


        out = []
        out.append(loss)
        out.append(boxes)
        out.append(b_roads)
        out.append(road_loss)
        out.append(box_loss)
        out.append(boxes_tensor)

        return out
