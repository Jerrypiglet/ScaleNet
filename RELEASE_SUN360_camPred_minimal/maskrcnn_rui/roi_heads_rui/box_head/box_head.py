# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn

from .roi_box_feature_extractors import make_roi_box_feature_extractor, make_classifier_feature_extractor
from .roi_box_predictors import make_roi_box_predictor_rui, make_roi_box_predictor
from .inference import make_roi_box_post_processor
from .loss import make_roi_box_loss_evaluator

class ROIBoxHeadRui(torch.nn.Module):
    """
    Generic Box Head class.
    """

    def __init__(self, cfg, in_channels, predictor_fn, output_cls_num=None):
        super(ROIBoxHeadRui, self).__init__()
        self.feature_extractor = make_roi_box_feature_extractor(cfg, in_channels)
        # print(in_channels)

        self.predictor = predictor_fn(
            cfg, self.feature_extractor.out_channels, output_cls_num=output_cls_num)
        # self.post_processor = make_roi_box_post_processor(cfg)
        # self.loss_evaluator = make_roi_box_loss_evaluator(cfg)

    def forward(self, features, proposals, targets=None):
        """
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels
            proposals (list[BoxList]): proposal boxes
            targets (list[BoxList], optional): the ground-truth targets.

        Returns:
            x (Tensor): the result of the feature extractor
            proposals (list[BoxList]): during training, the subsampled proposals
                are returned. During testing, the predicted boxlists are returned
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """
        x = self.feature_extractor(features, proposals)
        # print('xxxxx bboxRui', x.shape)
        # class_logits, x_avgPool = self.predictor(x)
# <<<<<<< HEAD
#         class_logits, feats = self.predictor(x)
#         # print('xxxxx bboxRui-class_logits', class_logits.shape)
#         return {'class_logits': class_logits, 'feats': feats}
# =======
        class_logits = self.predictor(x)
        # print('xxxxx bboxRui-class_logits', class_logits.shape)
        return {'class_logits': class_logits, 'feats': None}
# >>>>>>> pose-input2

# class ClassifiersHeadRui(torch.nn.Module):
#     """
#     Generic Box Head class.
#     """
#
#     def __init__(self, cfg, in_channels, predictor_fn):
#         super(ClassifiersHeadRui, self).__init__()
#         self.feature_extractor = make_classifier_feature_extractor(cfg, in_channels)
#
#         self.classifier_horizon = predictor_fn(cfg, self.feature_extractor.out_channels)
#         self.classifier_pitch = predictor_fn(cfg, self.feature_extractor.out_channels)
#         self.classifier_roll = predictor_fn(cfg, self.feature_extractor.out_channels)
#         self.classifier_vfov = predictor_fn(cfg, self.feature_extractor.out_channels)
#         self.classifier_camH = predictor_fn(cfg, self.feature_extractor.out_channels)
#
#     def forward(self, features, proposals, targets=None):
#         """
#         Arguments:
#             features (list[Tensor]): feature-maps from possibly several levels
#             proposals (list[BoxList]): proposal boxes
#             targets (list[BoxList], optional): the ground-truth targets.
#
#         Returns:
#             x (Tensor): the result of the feature extractor
#             proposals (list[BoxList]): during training, the subsampled proposals
#                 are returned. During testing, the predicted boxlists are returned
#             losses (dict[Tensor]): During training, returns the losses for the
#                 head. During testing, returns an empty dict.
#         """
#         x = self.feature_extractor(features, proposals)
#         # for feat in features:
#         #     print(feat.shape, '---')
#         # print(x.shape, '----------')
#
#         return_dict = {}
#         return_dict['output_horizon'] = self.classifier_horizon(x)
#         return_dict['output_pitch'] = self.classifier_pitch(x)
#         return_dict['output_roll'] = self.classifier_roll(x)
#         return_dict['output_vfov'] = self.classifier_vfov(x)
#         return_dict['output_camH'] = self.classifier_camH(x)
#
#         return return_dict


def build_roi_box_head_rui(cfg, in_channels, predictor_fn, output_cls_num=None):
    """
    Constructs a new box head.
    By default, uses ROIBoxHead, but if it turns out not to be enough, just register a new class
    and make it a parameter in the config
    """
    # return ROIBoxHead(cfg, in_channels)
    return ROIBoxHeadRui(cfg, in_channels, predictor_fn, output_cls_num=output_cls_num)


# ======= original ==========
class ROIBoxHead(torch.nn.Module):
    """
    Generic Box Head class.
    """

    def __init__(self, cfg, in_channels):
        super(ROIBoxHead, self).__init__()
        self.feature_extractor = make_roi_box_feature_extractor(cfg, in_channels)
        self.predictor = make_roi_box_predictor(
            cfg, self.feature_extractor.out_channels)
        self.post_processor = make_roi_box_post_processor(cfg)
        self.loss_evaluator = make_roi_box_loss_evaluator(cfg)

    def forward(self, features, proposals, targets=None, if_debug=False):
        """
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels
            proposals (list[BoxList]): proposal boxes
            targets (list[BoxList], optional): the ground-truth targets.

        Returns:
            x (Tensor): the result of the feature extractor
            proposals (list[BoxList]): during training, the subsampled proposals
                are returned. During testing, the predicted boxlists are returned
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """

        if targets is not None:
            # Faster R-CNN subsamples during training the proposals with a fixed
            # positive / negative ratio
            with torch.no_grad():
                proposals = self.loss_evaluator.subsample(proposals, targets)

        # extract features that will be fed to the final classifier. The
        # feature_extractor generally corresponds to the pooler + heads
        x = self.feature_extractor(features, proposals)
        if if_debug:
            print('xxxxx bbox', x.shape)
        # final classifier that converts the features into predictions
        class_logits, box_regression = self.predictor(x)
        if if_debug:
            print('xxxxx bbox-class_logits', class_logits.shape)

        # if not self.training:
        #     result = self.post_processor((class_logits, box_regression), proposals)
        #     return x, result, {}

        loss_classifier, loss_box_reg = self.loss_evaluator(
            [class_logits], [box_regression]
        )
        
        proposals_nms = self.post_processor((class_logits, box_regression), proposals)
        
        return (
            x,
            proposals, 
            proposals_nms, 
            dict(loss_classifier=loss_classifier, loss_box_reg=loss_box_reg),
        )


def build_roi_box_head(cfg, in_channels):
    """
    Constructs a new box head.
    By default, uses ROIBoxHead, but if it turns out not to be enough, just register a new class
    and make it a parameter in the config
    """
    return ROIBoxHead(cfg, in_channels)
