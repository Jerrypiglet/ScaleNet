# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from maskrcnn_rui.modeling import registry
from torch import nn

from maskrcnn_rui.modeling.make_layers import make_fc
from torch.nn import functional as F


@registry.ROI_BOX_PREDICTOR.register("FastRCNNPredictor")
class FastRCNNPredictor(nn.Module):
    def __init__(self, config, in_channels):
        super(FastRCNNPredictor, self).__init__()
        assert in_channels is not None

        num_inputs = in_channels

        num_classes = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES_bbox
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.cls_score = nn.Linear(num_inputs, num_classes)
        num_bbox_reg_classes = 2 if config.MODEL.CLS_AGNOSTIC_BBOX_REG else num_classes
        self.bbox_pred = nn.Linear(num_inputs, num_bbox_reg_classes * 4)

        nn.init.normal_(self.cls_score.weight, mean=0, std=0.01)
        nn.init.constant_(self.cls_score.bias, 0)

        nn.init.normal_(self.bbox_pred.weight, mean=0, std=0.001)
        nn.init.constant_(self.bbox_pred.bias, 0)

    def forward(self, x):
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        cls_logit = self.cls_score(x)
        bbox_pred = self.bbox_pred(x)
        return cls_logit, bbox_pred

@registry.ROI_BOX_PREDICTOR.register("FastRCNNPredictorRuiMod")
class FastRCNNPredictorRuiMod(nn.Module):
    def __init__(self, config, in_channels, output_cls_num=None):
        super(FastRCNNPredictorRuiMod, self).__init__()
        assert in_channels is not None

        num_inputs = in_channels

        if output_cls_num is None:
            num_classes = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES_h
        else:
            num_classes = output_cls_num
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.cls_score = nn.Linear(num_inputs, num_classes)
        # num_bbox_reg_classes = 2 if config.MODEL.CLS_AGNOSTIC_BBOX_REG else num_classes
        # self.bbox_pred = nn.Linear(num_inputs, num_bbox_reg_classes * 4)

        nn.init.normal_(self.cls_score.weight, mean=0, std=0.01)
        nn.init.constant_(self.cls_score.bias, 0)

        # nn.init.normal_(self.bbox_pred.weight, mean=0, std=0.001)
        # nn.init.constant_(self.bbox_pred.bias, 0)

        use_gn = False
        self.fc6_dim_reduce = make_fc(104, 64, use_gn)
        self.fc7_dim_reduce = make_fc(64, 16, use_gn)

    def forward(self, x):
        # print('=====1', x.shape) # torch.Size([10, 104, 3, 3])
        x = self.avgpool(x)
        # print('=====2', x.shape) # torch.Size([10, 104, 1, 1])
        x = x.view(x.size(0), -1)
        # print('=====3', x.shape) # torch.Size([10, 104])

        cls_logit = self.cls_score(x)
        # bbox_pred = self.bbox_pred(x)

        x_reduce = F.relu(self.fc6_dim_reduce(x))
        x = F.relu(self.fc7_dim_reduce(x_reduce))

        # return cls_logit, x
        return cls_logit

# @registry.CLASSIFIER_HEAD_PREDICTOR.register("FCPredictorRui")
# class FCPredictorRui(nn.Module):
#     def __init__(self, config, in_channels):
#         super(FCPredictorRui, self).__init__()
#         assert in_channels is not None
#
#         num_inputs = in_channels
#
#         num_classes = config.MODEL.CLASSIFIER_HEAD.NUM_CLASSES
#         self.avgpool = nn.AdaptiveAvgPool2d(1)
#         self.cls_score = nn.Linear(num_inputs, num_classes)
#         # num_bbox_reg_classes = 2 if config.MODEL.CLS_AGNOSTIC_BBOX_REG else num_classes
#         # self.bbox_pred = nn.Linear(num_inputs, num_bbox_reg_classes * 4)
#
#         nn.init.normal_(self.cls_score.weight, mean=0, std=0.01)
#         nn.init.constant_(self.cls_score.bias, 0)
#
#         # nn.init.normal_(self.bbox_pred.weight, mean=0, std=0.001)
#         # nn.init.constant_(self.bbox_pred.bias, 0)
#
#     def forward(self, x):
#         x = self.avgpool(x)
#         x = x.view(x.size(0), -1)
#         cls_logit = self.cls_score(x)
#         # bbox_pred = self.bbox_pred(x)
#         return cls_logit
#
@registry.ROI_BOX_PREDICTOR.register("FPNPredictorRui")
class FPNPredictorRui(nn.Module):
    def __init__(self, config, in_channels, output_cls_num=None):
        super(FPNPredictorRui, self).__init__()
        # num_classes = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        if output_cls_num is None:
            num_classes = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES_h
        else:
            num_classes = output_cls_num
        representation_size = in_channels

        self.cls_score = nn.Linear(representation_size, num_classes)
        # num_bbox_reg_classes = 2 if cfg.MODEL.CLS_AGNOSTIC_BBOX_REG else num_classes
        # self.bbox_pred = nn.Linear(representation_size, num_bbox_reg_classes * 4)

        nn.init.normal_(self.cls_score.weight, std=0.01)
        # nn.init.normal_(self.bbox_pred.weight, std=0.001)
        # for l in [self.cls_score, self.bbox_pred]:
        #     nn.init.constant_(l.bias, 0)

    def forward(self, x):
        if x.ndimension() == 4:
            assert list(x.shape[2:]) == [1, 1]
            x = x.view(x.size(0), -1)
        scores = self.cls_score(x)
        # bbox_deltas = self.bbox_pred(x)

        return scores

@registry.ROI_BOX_PREDICTOR.register("FPNPredictor")
class FPNPredictor(nn.Module):
    def __init__(self, cfg, in_channels):
        super(FPNPredictor, self).__init__()
        num_classes = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES_bbox
        # print(cfg)
        representation_size = in_channels

        self.cls_score = nn.Linear(representation_size, num_classes)
        num_bbox_reg_classes = 2 if cfg.MODEL.CLS_AGNOSTIC_BBOX_REG else num_classes
        self.bbox_pred = nn.Linear(representation_size, num_bbox_reg_classes * 4)

        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        for l in [self.cls_score, self.bbox_pred]:
            nn.init.constant_(l.bias, 0)

    def forward(self, x):
        if x.ndimension() == 4:
            assert list(x.shape[2:]) == [1, 1]
            x = x.view(x.size(0), -1)
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)

        return scores, bbox_deltas


def make_roi_box_predictor_rui(cfg, in_channels, output_cls_num):
    func = registry.ROI_BOX_PREDICTOR[cfg.MODEL.ROI_BOX_HEAD.PREDICTOR_h]
    return func(cfg, in_channels, output_cls_num)

def make_classifier_head_predictor(cfg, in_channels):
    func = registry.CLASSIFIER_HEAD_PREDICTOR[cfg.MODEL.CLASSIFIER_HEAD.PREDICTOR]
    return func(cfg, in_channels)

def make_roi_box_predictor(cfg, in_channels):
    func = registry.ROI_BOX_PREDICTOR[cfg.MODEL.ROI_BOX_HEAD.PREDICTOR_bbox]
    return func(cfg, in_channels)