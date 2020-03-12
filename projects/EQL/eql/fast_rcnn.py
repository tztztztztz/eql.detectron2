import logging
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers, FastRCNNOutputs

logger = logging.getLogger(__name__)


class EQLFastRCNNOutputLayers(FastRCNNOutputLayers):
    """
    Two linear layers for predicting Fast R-CNN outputs:
      (1) proposal-to-detection box regression deltas
      (2) classification scores
    """

    def __init__(self, input_size, num_classes, cls_agnostic_bbox_reg, box_dim=4, prior_prob=0.001):
        """
        Args:
            input_size (int): channels, or (channels, height, width)
            num_classes (int): number of foreground classes
            cls_agnostic_bbox_reg (bool): whether to use class agnostic for bbox regression
            box_dim (int): the dimension of bounding boxes.
                Example box dimensions: 4 for regular XYXY boxes and 5 for rotated XYWHA boxes
        """
        super(FastRCNNOutputLayers, self).__init__()

        if not isinstance(input_size, int):
            input_size = np.prod(input_size)

        # The prediction layer for num_classes foreground classes and one background class
        # (hence + 1)
        self.cls_score = nn.Linear(input_size, num_classes)
        num_bbox_reg_classes = 1 if cls_agnostic_bbox_reg else num_classes
        self.bbox_pred = nn.Linear(input_size, num_bbox_reg_classes * box_dim)

        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        nn.init.constant_(self.bbox_pred.bias, 0)

        bias_value = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.cls_score.bias, bias_value)
        

class EQLFastRCNNOutputs(FastRCNNOutputs):
    """
    A class that stores information about outputs of a Fast R-CNN head.
    It provides methods that are used to decode the outputs of a Fast R-CNN head.
    """

    def __init__(
        self, box2box_transform, pred_class_logits, pred_proposal_deltas, proposals, smooth_l1_beta,
        lambda_, freq_info
    ):
        """
        Args:
            box2box_transform (Box2BoxTransform/Box2BoxTransformRotated):
                box2box transform instance for proposal-to-detection transformations.
            pred_class_logits (Tensor): A tensor of shape (R, K + 1) storing the predicted class
                logits for all R predicted object instances.
                Each row corresponds to a predicted object instance.
            pred_proposal_deltas (Tensor): A tensor of shape (R, K * B) or (R, B) for
                class-specific or class-agnostic regression. It stores the predicted deltas that
                transform proposals into final box detections.
                B is the box dimension (4 or 5).
                When B is 4, each row is [dx, dy, dw, dh (, ....)].
                When B is 5, each row is [dx, dy, dw, dh, da (, ....)].
            proposals (list[Instances]): A list of N Instances, where Instances i stores the
                proposals for image i, in the field "proposal_boxes".
                When training, each Instances must have ground-truth labels
                stored in the field "gt_classes" and "gt_boxes".
            smooth_l1_beta (float): The transition point between L1 and L2 loss in
                the smooth L1 loss function. When set to 0, the loss becomes L1. When
                set to +inf, the loss becomes constant 0.
        """
        super(EQLFastRCNNOutputs, self).__init__(box2box_transform=box2box_transform,
                                                 pred_class_logits=pred_class_logits,
                                                 pred_proposal_deltas=pred_proposal_deltas,
                                                 proposals=proposals,
                                                 smooth_l1_beta=smooth_l1_beta)

        self.lambda_ = lambda_
        self.freq_info = freq_info

    def exclude_func(self):
        # instance-level weight
        bg_ind = self.n_c
        weight = (self.gt_classes != bg_ind).float()
        weight = weight.view(self.n_i, 1).expand(self.n_i, self.n_c)
        return weight

    def threshold_func(self):
        # class-level weight
        weight = self.pred_class_logits.new_zeros(self.n_c)
        weight[self.freq_info < self.lambda_] = 1
        weight = weight.view(1, self.n_c).expand(self.n_i, self.n_c)
        return weight

    def eql_loss(self):

        self.n_i, self.n_c = self.pred_class_logits.size()

        def expand_label(pred, gt_classes):
            target = pred.new_zeros(self.n_i, self.n_c + 1)
            target[torch.arange(self.n_i), gt_classes] = 1
            return target[:, :self.n_c]

        target = expand_label(self.pred_class_logits, self.gt_classes)

        eql_w = 1 - self.exclude_func() * self.threshold_func() * (1 - target)

        cls_loss = F.binary_cross_entropy_with_logits(self.pred_class_logits, target,
                                                      reduction='none')

        return torch.sum(cls_loss * eql_w) / self.n_i

    """
    A subclass is expected to have the following methods because
    they are used to query information about the head predictions.0
    """

    def losses(self):
        """
        Compute the default losses for box head in Fast(er) R-CNN,
        with softmax cross entropy loss and smooth L1 loss.

        Returns:
            A dict of losses (scalar tensors) containing keys "loss_cls" and "loss_box_reg".
        """
        return {
            "loss_cls": self.eql_loss(),
            "loss_box_reg": self.smooth_l1_loss(),
        }

    def predict_probs(self):
        """
        Returns:
            list[Tensor]: A list of Tensors of predicted class probabilities for each image.
                Element i has shape (Ri, K + 1), where Ri is the number of predicted objects
                for image i.
        """
        probs = F.sigmoid(self.pred_class_logits)
        # add a dummy probs, representing a fake score for background class
        n = probs.size(0)
        dummy_probs = probs.new_zeros(n, 1)
        probs = torch.cat([probs, dummy_probs], dim=1)

        return probs.split(self.num_preds_per_image, dim=0)
