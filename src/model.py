"""
Module to define our model architecture
"""

import torch
import torch.nn as nn
import torchvision
import os


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
    samples = nn.functional.interpolate(samples, size=(800, 800))  #resize
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
    Repackage boxes from [batch_size, N, 2, 4] to [batch_size, N, 4]
    """
    repackaged = []
    for target in targets:
        target_new = {'boxes': get_boxes_from_4_points_to_2_points(target['bounding_box']),
                      'labels': target['category']}
        repackaged.append(target_new)

    return repackaged


def map_level_loss(map_pred: torch.Tensor, map_true: torch.Tensor):
    """
    Intersection over Union for two maps.
    """
    intersection_part = map_pred * map_true
    union_part = map_pred + map_true - intersection_part
    return - intersection_part.sum() / union_part.sum()


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
        # repackage images from [batch_size, 6, 3, 256, 306] to [batch_size, 3, 800, 800]
        rescaled_images = repackage_image_batch(images)

        # repackage labels from [batch_size, N, 2, 4] to [batch_size, N, 4]
        repackaged_box_targets = repackage_targets(box_targets)

        # do image detection

        # TODO: get box loss
        # output includes losses
        # https://github.com/pytorch/vision/blob/master/torchvision/models/detection/generalized_rcnn.py
        boxes_out = self.image_detect(rescaled_images, repackaged_box_targets)
        box_loss = ...

        # TODO: get predicted box losses from output and repackage for output
        pred_boxes = ...

        # do road map
        # TODO: double check
        # comment: not sure if transform needed since images already transformed in data_helper
        #images, targets = self.transform(images, None)
        features = self.backbone(images.tensors)
        # The features is a OrderDict as follows:
        #     key: 0    -> value: a tensor of shape [2, 256, 200, 200]
        #     key: 1    -> value: a tensor of shape [2, 256, 100, 100]
        #     key: 2    -> value: a tensor of shape [2, 256, 50, 50]
        #     key: 3    -> value: a tensor of shape [2, 256, 25, 25]
        #     key: pool -> value: a tensor of shape [2, 256, 13, 13]

        roads = self.road(features)

        # should be [batch_size, 800, 800, 2]
        batch_size, width, height, logits = roads.shape

        # cast road_targets from boolean TRUE/FALSE to Integers 1/0
        road_labels = [road_target.int() for road_target in road_targets]

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
        boxes = [get_boxes_from_2_points_to_4_points(batch_boxes) for batch_boxes in boxes_out]

        #TODO: make road map binary values, then to boolean TRUE/FALSE
        # not sure if this works but check documentation here
        # https://pytorch.org/docs/stable/torch.html#comparison-ops
        values, b_roads = torch.max(roads, dim = 3)
        # not sure if this works but check documentation here
        # https://pytorch.org/docs/stable/tensors.html#torch.Tensor.to
        b_roads = b_roads.bool()

        boxes_out = []
        boxes_out.append(loss)
        boxes_out.append(boxes)
        boxes_out.append(b_roads)
        boxes_out.append(road_loss)
        boxes_out.append(box_loss)

        return boxes_out
