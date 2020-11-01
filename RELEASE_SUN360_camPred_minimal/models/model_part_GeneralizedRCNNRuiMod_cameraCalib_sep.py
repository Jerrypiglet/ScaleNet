import cv2
import logging

import torch
import torch.nn as nn
# from torchvision import models, transforms
from torchvision import transforms as T
from torchvision.transforms import functional as F

# from torchvision.models.densenet import model_urls
# model_urls['densenet161'] = model_urls['densenet161'].replace('https://', 'http://')

from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_rui.modeling.backbone import build_backbone
from maskrcnn_rui.modeling.rpn.rpn import build_rpn
from maskrcnn_rui.roi_heads_rui.roi_heads import build_roi_h_heads, build_classifier_heads, build_roi_bbox_heads
# from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer

from termcolor import colored
from utils.logger import setup_logger, printer
from maskrcnn_benchmark.utils.comm import synchronize, get_rank
from torchsummary import summary
from utils.model_utils import CATEGORIES

# from models.model_part_pointnet import CamHPointNet

class GeneralizedRCNNRuiMod_cameraCalib(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self, cfg, opt, modules_not_build=[], logger=None, rank=-1, confidence_threshold=0.7):
        super().__init__()

        self.backbone = build_backbone(cfg)
        # self.rpn = build_rpn(cfg, self.backbone.out_channels)

        self.if_roi_h_heads = 'roi_h_heads' not in modules_not_build
        if self.if_roi_h_heads:
            self.roi_h_heads = build_roi_h_heads(cfg, opt, self.backbone.out_channels)

        self.if_classifier_heads = 'classifier_heads' not in modules_not_build
        if self.if_classifier_heads:
            self.classifier_heads = build_classifier_heads(cfg, opt, self.backbone.out_channels)

        self.if_roi_bbox_heads = 'roi_bbox_heads' not in modules_not_build and opt.est_bbox
        if self.if_roi_bbox_heads:
            self.rpn = build_rpn(cfg, self.backbone.out_channels)
            self.roi_bbox_heads = build_roi_bbox_heads(cfg, self.backbone.out_channels)

        # self.if_camH_pointnet = 'camH_pointnet' not in modules_not_build and opt.pointnet_camH
        # if self.if_camH_pointnet:
        #     self.camH_pointnet = CamHPointNet(in_channels=6, out_channels=cfg.MODEL.CLASSIFIER_HEADNUM_CLASSES.NUM_CLASSES)


        self.cfg = cfg
        self.opt = opt
        self.device = self.cfg.MODEL.DEVICE
        self.rank = rank
        self.cpu_device = torch.device("cpu")
        self.confidence_threshold = confidence_threshold
        # self.transforms = self.build_transform()
        self.palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
        self.CATEGORIES = CATEGORIES

        # self.logger = logging.getLogger("GeneralizedRCNNRuiMod:in_model")
        self.logger = logger
        self.printer = printer(get_rank(), self.opt.debug)
        if self.opt.debug:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.INFO)

    def prepare_images(self, inputCOCO_Image_maskrcnnTransform_list):
        # Transform so that the min size is no smaller than cfg.INPUT.MIN_SIZE_TRAIN, and the max size is no larger than cfg.INPUT.MIN_SIZE_TRAIN
        # image_batch = [self.transforms(original_image) for original_image in original_image_batch_list]
        image_batch = inputCOCO_Image_maskrcnnTransform_list
        image_sizes_after_transform = [(image_after.shape[2], image_after.shape[1]) for image_after in image_batch]
        # if self.training:
        #     for original_image, image_after, image_after_size in zip(inputCOCO_Image_maskrcnnTransform, image_batch, image_sizes_after_transform):
        #         self.printer.print('[generalized_rcnn_rui-prepare_images] Image sizes:', original_image.shape, '-->', image_after.shape, image_after_size)

        # [Rui] PADDING
        # convert to an ImageList, ``padded`` so that it is divisible by cfg.DATALOADER.SIZE_DIVISIBILITY
        image_list = to_image_list(image_batch, self.cfg.DATALOADER.SIZE_DIVISIBILITY)
        # print(self.cfg.INPUT.MIN_SIZE_TRAIN, self.cfg.INPUT.MAX_SIZE_TRAIN, self.cfg.INPUT.MIN_SIZE_TEST, self.cfg.INPUT.MAX_SIZE_TEST)
        if self.training:
            self.printer.print('PADDED: image_list.tensors, image_list.image_sizes (before pad):', image_list.tensors.shape, image_list.image_sizes)
        image_list = image_list.to(self.device)
        return image_list, image_sizes_after_transform

    def forward(self, original_image_batch_list, list_of_bbox_list_cpu=None, list_of_oneLargeBbox_list=None, targets=None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """

        if_print = self.training
        # if self.training and (list_of_bbox_list_cpu is None or list_of_oneLargeBbox_list_cpu is None):
        #     raise ValueError("In training mode, targets should be passed")

        # images = to_image_list(images)
        images, image_sizes_after_transform = self.prepare_images(original_image_batch_list)
        features = self.backbone(images.tensors)
        ## DETACH!!!!!!!!!!!
        # features = tuple(feat.detach() for feat in list(features))
        # if if_print:
        #     self.printer.print('[generalized_rcnn_rui] Feats:')
        # for feat in features:
        #     self.printer.print(feat.shape)

        return_dict = {'image_sizes_after_transform': image_sizes_after_transform}


        if list_of_bbox_list_cpu is not None:
            list_of_bbox_list = [bbox_list_array.to(self.device) for bbox_list_array in list_of_bbox_list_cpu]
            list_of_bbox_list = [bbox_list.resize(size) for bbox_list, size in zip(list_of_bbox_list, image_sizes_after_transform)]
            if if_print:
                self.printer.print('[generalized_rcnn_rui] list_of_bbox_list:', list_of_bbox_list) # list([BoxList(num_boxes=1000, image_width=1066, image_height=800, mode=xyxy)])

            roi_heads_output = self.roi_h_heads(features, list_of_bbox_list)
            class_logits = roi_heads_output['class_logits']
            # print('==roi_feats', roi_feats.shape, roi_feats.detach().cpu().numpy())
            class_logits_softmax = nn.functional.softmax(class_logits, dim=1)
            # print(class_logits[0], torch.sum(class_logits[0]))
            bbox_lengths = [len(bbox_list) for bbox_list in list_of_bbox_list]
            class_logits_softmax_list = class_logits_softmax.split(bbox_lengths)

            return_dict.update({'class_person_H_logits_softmax_list': class_logits_softmax_list, 'class_person_H_logits_softmax': class_logits_softmax, 'class_person_H_logits': class_logits, 'bbox_lengths': bbox_lengths})

            roi_feats = roi_heads_output['feats'] # [N_all, D]
            return_dict.update({'roi_feats': roi_feats})

            # Global feat with list_of_oneLargeBbox_list_cpu
        if list_of_oneLargeBbox_list is not None:
            list_of_oneLargeBbox_list = [bbox_list.resize(size) for bbox_list, size in zip(list_of_oneLargeBbox_list, image_sizes_after_transform)]

            cls_outputs = self.classifier_heads(features, list_of_oneLargeBbox_list)
            return_dict.update({'output_horizon': cls_outputs['output_horizon']['class_logits'], 'output_pitch': cls_outputs['output_pitch']['class_logits'], \
                                'output_roll': cls_outputs['output_roll']['class_logits'], 'output_vfov': cls_outputs['output_vfov']['class_logits']})
            # if not self.opt.pointnet_camH:
            #     return_dict.update({'output_camH': cls_outputs['output_camH']['class_logits']})

        # if self.if_camH_pointnet and bboxed_padded is not None:


        if self.if_roi_bbox_heads:
            proposals, proposal_losses = self.rpn(images, features, targets=None)
            return_dict.update({'proposals': proposals})
            if self.roi_bbox_heads:
                x, predictions, detector_losses = self.roi_bbox_heads(features, proposals, targets)
                return_dict.update({'x': x, 'predictions': predictions, 'detector_losses': detector_losses})
            else:
                # RPN-only models don't have roi_heads
                x = features
                result = proposals
                detector_losses = {}

            if self.training:
                return_dict.update(detector_losses)
                return_dict.update(proposal_losses)

        return return_dict


    def post_process(self, predictions, image_sizes_after_transform):
        predictions = [o.to(self.cpu_device) for o in predictions]

        # always single image is passed at a time
        # prediction = predictions[0] # BoxList(num_boxes=73, image_width=1066, image_height=800, mode=xyxy)

        prediction_list = []
        prediction_list_ori = []

        for size, prediction in zip(image_sizes_after_transform, predictions):
            # reshape prediction (a BoxList) into the original image size
            # height, width = original_image.shape[:-1]
            prediction_list_ori.append(prediction)
            prediction = prediction.resize(size)

            if prediction.has_field("mask"):
                # if we have masks, paste the masks in the right position
                # in the image, as defined by the bounding boxes
                masks = prediction.get_field("mask")
                # always single image is passed at a time
                masks = self.masker([masks], [prediction])[0]
                prediction.add_field("mask", masks)

            prediction_list.append(prediction)

        return prediction_list, prediction_list_ori

    def select_and_vis_bbox(self, prediction_list, prediction_list_ori, image_batch_list):
        top_prediction_list = [self.select_top_predictions(prediction) for prediction in prediction_list]
        top_prediction_list_ori = [self.select_top_predictions(prediction) for prediction in prediction_list_ori]

        result_list = []

        for image, top_predictions in zip(image_batch_list, top_prediction_list):
            result = image.copy()
            # if self.show_mask_heatmaps:
            #     return self.create_mask_montage(result, top_predictions)
            result = self.overlay_boxes(result, top_predictions)
            if self.cfg.MODEL.MASK_ON:
                result = self.overlay_mask(result, top_predictions)
            if self.cfg.MODEL.KEYPOINT_ON:
                result = self.overlay_keypoints(result, top_predictions)
            result = self.overlay_class_names(result, top_predictions)

            result_list.append(result)

        return result_list, top_prediction_list

    def select_top_predictions(self, predictions):
        """
        Select only predictions which have a `score` > self.confidence_threshold,
        and returns the predictions in descending order of score

        Arguments:
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `scores`.

        Returns:
            prediction (BoxList): the detected objects. Additional information
                of the detection properties can be found in the fields of
                the BoxList via `prediction.fields()`
        """
        scores = predictions.get_field("scores")
        keep = torch.nonzero(scores > self.confidence_threshold).squeeze(1)
        predictions = predictions[keep]
        scores = predictions.get_field("scores")
        _, idx = scores.sort(0, descending=True)
        return predictions[idx]

    def compute_colors_for_labels(self, labels):
        """
        Simple function that adds fixed colors depending on the class
        """
        colors = labels[:, None] * self.palette
        colors = (colors % 255).numpy().astype("uint8")
        return colors

    def overlay_boxes(self, image, predictions):
        """
        Adds the predicted boxes on top of the image

        Arguments:
            image (np.ndarray): an image as returned by OpenCV
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `labels`.
        """
        labels = predictions.get_field("labels")
        boxes = predictions.bbox

        colors = self.compute_colors_for_labels(labels).tolist()

        for box, color in zip(boxes, colors):
            box = box.to(torch.int64)
            top_left, bottom_right = box[:2].tolist(), box[2:].tolist()
            image = cv2.rectangle(
                image, tuple(top_left), tuple(bottom_right), tuple(color), 1
            )

        return image

    def overlay_class_names(self, image, predictions):
        """
        Adds detected class names and scores in the positions defined by the
        top-left corner of the predicted bounding box

        Arguments:
            image (np.ndarray): an image as returned by OpenCV
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `scores` and `labels`.
        """
        scores = predictions.get_field("scores").tolist()
        labels = predictions.get_field("labels").tolist()
        labels = [self.CATEGORIES[int(i)] for i in labels]
        boxes = predictions.bbox

        template = "{}: {:.2f}"
        for box, score, label in zip(boxes, scores, labels):
            x, y = box[:2]
            s = template.format(label, score)
            cv2.putText(
                image, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1
            )

        return image


class Resize(object):
    def __init__(self, min_size, max_size):
        self.min_size = min_size
        self.max_size = max_size

    # modified from torchvision to add support for max size
    def get_size(self, image_size):
        w, h = image_size
        size = self.min_size
        max_size = self.max_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def __call__(self, image):
        size = self.get_size(image.size)
        image = F.resize(image, size)
        return image