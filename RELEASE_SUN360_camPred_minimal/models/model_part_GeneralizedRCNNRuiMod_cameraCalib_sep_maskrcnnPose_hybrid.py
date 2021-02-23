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

class GeneralizedRCNNRuiMod_cameraCalib_maskrcnnPose(nn.Module):
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


        self.backbone = build_backbone(cfg)
        # self.rpn = build_rpn(cfg, self.backbone.out_channels)

        print('modules_not_build', modules_not_build)
        self.if_classifier_heads = 'classifier_heads' not in modules_not_build
        if self.if_classifier_heads:
            self.classifier_heads = build_classifier_heads(cfg, opt, self.backbone.out_channels)

        # h and potentially person h heads, sharing featmap
        self.if_shared_kps_head = self.opt.est_kps
        #  and 'kps' in self.cfg.MODEL.NAME
        print("if_shared_kps_head: ", self.if_shared_kps_head, self.opt.est_kps, self.cfg.MODEL.NAME)


        self.if_roi_h_heads = 'roi_h_heads' not in modules_not_build
        if self.if_roi_h_heads and not self.if_shared_kps_head:
            # independent h head without kps
            self.roi_h_heads = build_roi_h_heads(cfg, opt, self.backbone.out_channels)

        self.if_roi_bbox_heads = 'roi_bbox_heads' not in modules_not_build
        if self.if_roi_bbox_heads:
            # independent h + kps head, and potentially with bbox est head
            if self.opt.est_bbox:
                self.rpn = build_rpn(cfg, self.backbone.out_channels)
            self.roi_bbox_heads = build_roi_bbox_heads(cfg, opt, self.backbone.out_channels, if_roi_h_heads=self.if_roi_h_heads and self.if_shared_kps_head)

        # self.if_camH_pointnet = 'camH_pointnet' not in modules_not_build and opt.pointnet_camH
        # if self.if_camH_pointnet:
        #     self.camH_pointnet = CamHPointNet(in_channels=6, out_channels=cfg.MODEL.CLASSIFIER_HEADNUM_CLASSES.NUM_CLASSES)

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

    def forward(self, original_image_batch_list, list_of_bbox_list_cpu=None, list_of_oneLargeBbox_list=None, targets=None, list_of_box_list_kps_gt=None, input_data=''):
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

        assert input_data in ['coco', 'SUN360', 'IMDB-23K']

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
    
        if self.if_roi_bbox_heads and input_data in ['coco', 'IMDB-23K']:
            if self.opt.est_bbox:
                # print('=====targets', targets)
                proposals, proposal_losses = self.rpn(images, features, targets)
                # print('=====proposals', proposals, proposals[0].fields(), proposals[0].get_field('objectness').shape)
                target_idxes_with_valid_kps_list = []
            else:
                proposals = [box_list_kps_gt.copy_with_fields(['scores', 'labels']) for box_list_kps_gt in  list_of_box_list_kps_gt]
                targets_dup = [box_list_kps_gt.copy_with_fields(['scores', 'labels', 'keypoints']) for box_list_kps_gt in  list_of_box_list_kps_gt]

                target_with_valid_kps_list = []
                target_idxes_with_valid_kps_list = []
                for target in targets_dup:
                    keypoints_gt = target.get_field('keypoints').keypoints
                    # print(keypoints_gt.shape, keypoints_gt)
                    kps_sum = torch.sum(torch.sum(keypoints_gt[:, :, :2], 1), 1)
                    kps_mask = (kps_sum != 0.).cpu().numpy()
                    # print('====kps_mask', kps_mask, np.where(kps_mask))
                    target_with_valid_kps = target[np.where(kps_mask)]
                    # print(target_with_valid_kps)
                    target_idxes_with_valid_kps_list.append(np.where(kps_mask))
                    target_with_valid_kps_list.append(target_with_valid_kps)
                targets = targets_dup

                # targets = target_with_valid_kps_list

                # extra_bboxes = extra_bboxes.resize(image_sizes_after_transform)
                # extra_bboxes.size = proposals[0].size
                # proposals = [extra_bboxes]
            return_dict.update({'proposals': proposals})
            # if self.roi_bbox_heads:
            x, proposals, predictions, detector_losses, outputs_roi_bbox_heads = self.roi_bbox_heads(self.opt, features, proposals, targets, target_idxes_with_valid_kps_list=target_idxes_with_valid_kps_list)
            return_dict.update({'x': x, 'predictions': predictions, 'detector_losses': detector_losses})

            # print('+++++++++ predictions', predictions, detector_losses)
            
            if self.training:
                return_dict.update(detector_losses)
                if self.opt.est_bbox:
                    return_dict.update(proposal_losses)


        if list_of_bbox_list_cpu is not None:
            list_of_bbox_list = [bbox_list_array.to(self.device) for bbox_list_array in list_of_bbox_list_cpu]
            list_of_bbox_list = [bbox_list.resize(size) for bbox_list, size in zip(list_of_bbox_list, image_sizes_after_transform)]
            bbox_lengths = [len(bbox_list) for bbox_list in list_of_bbox_list]
            return_dict.update({'bbox_lengths': bbox_lengths})

            if self.if_roi_h_heads:
                if if_print:
                    self.printer.print('[generalized_rcnn_rui] list_of_bbox_list:', list_of_bbox_list) # list([BoxList(num_boxes=1000, image_width=1066, image_height=800, mode=xyxy)])

                if self.if_shared_kps_head:
                    class_logits = outputs_roi_bbox_heads['person_h_logits']
                else:
                    roi_heads_output = self.roi_h_heads(features, list_of_bbox_list)
                    class_logits = roi_heads_output['class_logits']
                # print('||||||||||||||||', class_logits.shape, sum(bbox_lengths), bbox_lengths)
                # print('==roi_feats', roi_feats.shape, roi_feats.detach().cpu().numpy())
                class_logits_softmax = nn.functional.softmax(class_logits, dim=1)
                # print(class_logits[0], torch.sum(class_logits[0]))

                class_logits_softmax_list = class_logits_softmax.split(bbox_lengths)

                return_dict.update({'class_person_H_logits_softmax_list': class_logits_softmax_list, 'class_person_H_logits_softmax': class_logits_softmax, 'class_person_H_logits': class_logits, 'bbox_lengths': bbox_lengths})

                # roi_feats = roi_heads_output['feats'] # [N_all, D]
                # return_dict.update({'roi_feats': roi_feats})

        # Global feat with list_of_oneLargeBbox_list_cpu
        if list_of_oneLargeBbox_list is not None and self.if_classifier_heads:
            list_of_oneLargeBbox_list = [bbox_list.resize(size) for bbox_list, size in zip(list_of_oneLargeBbox_list, image_sizes_after_transform)]

            cls_outputs = self.classifier_heads(features, list_of_oneLargeBbox_list)
            return_dict.update({'output_horizon': cls_outputs['output_horizon']['class_logits'], 'output_pitch': cls_outputs['output_pitch']['class_logits'], \
                                'output_roll': cls_outputs['output_roll']['class_logits'], 'output_vfov': cls_outputs['output_vfov']['class_logits']})
            if not self.opt.pointnet_camH:
                # return_dict.update({'output_camH': cls_outputs['output_camH']['class_logits']})
                pass

        return return_dict


    def post_process(self, predictions, image_sizes_ori):
        predictions = [o.to(self.cpu_device) for o in predictions]

        # always single image is passed at a time
        # prediction = predictions[0] # BoxList(num_boxes=73, image_width=1066, image_height=800, mode=xyxy)

        prediction_list = []
        prediction_list_ori = []

        for size, prediction in zip(image_sizes_ori, predictions):
            # reshape prediction (a BoxList) into the original image size
            # height, width = original_image.shape[:-1]
            # prediction_list.append(prediction)
            prediction_ori = prediction.resize(size)

            if prediction.has_field("mask"):
                # if we have masks, paste the masks in the right position
                # in the image, as defined by the bounding boxes
                masks = prediction.get_field("mask")
                # always single image is passed at a time
                masks = self.masker([masks], [prediction])[0]
                prediction.add_field("mask", masks)

            prediction_list.append(prediction)
            prediction_list_ori.append(prediction_ori)

        return prediction_list, prediction_list_ori

    def select_and_vis_bbox(self, prediction_list, image_batch_list, select_top=True):
        if self.opt.est_bbox and select_top:
            top_prediction_list = [self.select_top_predictions(prediction) for prediction in prediction_list]
            # top_prediction_list_ori = [self.select_top_predictions(prediction) for prediction in prediction_list_ori]
        else:
            top_prediction_list = prediction_list

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
                # self.overlay_straighten_ratios(result, top_predictions)
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

    def overlay_keypoints(self, image, predictions):
        keypoints = predictions.get_field("keypoints")
        kps = keypoints.keypoints
        if_fake_scores = False
        if keypoints.has_field("logits"):
            scores = keypoints.get_field("logits")
        else:
            scores = self.fake_kps_score_logits(kps)
            if_fake_scores = True
        kps = torch.cat((kps[:, :, 0:2], scores[:, :, None]), dim=2).numpy()
        for region in kps:
            image = vis_keypoints(image, region.transpose((1, 0)), if_fake_scores=if_fake_scores)
        return image

    def fake_kps_score_logits(self, kps):
        scores = torch.ones(kps.shape[:2]) * kps[:, :, -1] * 1000
        return scores

    def overlay_straighten_ratios(self, image, predictions):
        straighten_ratio_list = self.get_straighten_ratio_from_pred(predictions)
        # print(straighten_ratio_list)
        labels = predictions.get_field("labels")
        boxes = predictions.bbox

        colors = self.compute_colors_for_labels(labels).tolist()

        template = "str_ratio {:.2f}"
        for box, color, straighten_ratio in zip(boxes, colors, straighten_ratio_list):
            x, y = box[:2]
            s = template.format(straighten_ratio)
            cv2.putText(
                image, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 0, 0), 2
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

    def get_straighten_ratio_from_pred(self, prediction, kp_thresh=2, gt_input=False):
        keypoints = prediction.get_field("keypoints")
        kps = keypoints.keypoints
        if gt_input:
            scores = torch.ones((kps.shape[0], kps.shape[1], 1)) + kp_thresh
            scores *= kps[:, :, 2:3].cpu()
            kps_batch = torch.cat((kps[:, :, 0:2].cpu(), scores), dim=2).numpy() # (2, 17, 3)
        else:
            scores = keypoints.get_field("logits")
            kps_batch = torch.cat((kps[:, :, 0:2], scores[:, :, None]), dim=2).cpu().numpy() # (2, 17, 3)
        ratio_batch = []
        for region_idx, kps in enumerate(kps_batch):
            kps = kps.transpose((1, 0))

            dataset_keypoints = PersonKeypoints.NAMES
            # kp_lines = PersonKeypoints.CONNECTIONS
            mid_shoulder = (
                kps[:2, dataset_keypoints.index('right_shoulder')] +
                kps[:2, dataset_keypoints.index('left_shoulder')]) / 2.0
            sc_mid_shoulder = np.minimum(
                kps[2, dataset_keypoints.index('right_shoulder')],
                kps[2, dataset_keypoints.index('left_shoulder')])
            mid_hip = (
                kps[:2, dataset_keypoints.index('right_hip')] +
                kps[:2, dataset_keypoints.index('left_hip')]) / 2.0
            sc_mid_hip = np.minimum(
                kps[2, dataset_keypoints.index('right_hip')],
                kps[2, dataset_keypoints.index('left_hip')])
            dist_pred = 0.
            dist_straighten = 0.

            if sc_mid_shoulder > kp_thresh and sc_mid_hip > kp_thresh:
                dist_pred += np.abs(mid_shoulder[1] - mid_hip[1])
                dist_straighten += pts_dist(mid_shoulder, mid_hip)

            which_side_vis_list = []
            for which_side in ['left', 'right']:
                which_side_vis_list.append(kps[2, dataset_keypoints.index(which_side+'_hip')] > kp_thresh and kps[2, dataset_keypoints.index(which_side+'_ankle')] > kp_thresh and kps[2, dataset_keypoints.index(which_side+'_knee')] > kp_thresh)

            sides_reweight = sum(which_side_vis_list)
            if sides_reweight > 0:
                which_side_weight_array = np.asarray(which_side_vis_list) / sides_reweight
            else:
                which_side_weight_array = [0., 0.]

            for which_side, which_side_vis, which_side_weight in zip(['left', 'right'], which_side_vis_list, which_side_weight_array):
                kps_hip = kps[:2, dataset_keypoints.index(which_side+'_hip')]
                kps_knee = kps[:2, dataset_keypoints.index(which_side+'_knee')]
                kps_ankle = kps[:2, dataset_keypoints.index(which_side+'_ankle')]
                dist_pred_side = np.abs(kps_hip[1] - kps_ankle[1])
                dist_straighten_side = pts_dist(kps_hip, kps_knee) + pts_dist(kps_ankle, kps_knee)
            #     print(dist_pred_side / dist_straighten_side)
                dist_pred += dist_pred_side * which_side_weight
                dist_straighten += dist_straighten_side * which_side_weight

            sides_reweight = sum(which_side_vis_list)
            if sides_reweight > 0:
                which_side_weight_array = np.asarray(which_side_vis_list) / sides_reweight
            else:
                which_side_weight_array = [0., 0.]

            # dist_pred_legs = []
            # dist_straighten_legs = []
            # for which_side, which_side_vis in zip(['left', 'right'], which_side_vis_list):
            #     kps_hip = kps[:2, dataset_keypoints.index(which_side+'_hip')]
            #     kps_knee = kps[:2, dataset_keypoints.index(which_side+'_knee')]
            #     kps_ankle = kps[:2, dataset_keypoints.index(which_side+'_ankle')]
            #     dist_pred_side = np.abs(kps_hip[1] - kps_ankle[1])
            #     dist_straighten_side = pts_dist(kps_hip, kps_knee) + pts_dist(kps_ankle, kps_knee)
            # #     print(dist_pred_side / dist_straighten_side)
            # #     dist_pred += dist_pred_side * which_side_weight
            # #     dist_straighten += dist_straighten_side * which_side_weight
            #     if which_side_vis:
            #         dist_pred_legs.append(dist_pred_side)
            #         dist_straighten_legs.append(dist_straighten_side)
            #
            # if dist_pred_legs:
            #     legs_ratios = [dist_pred_leg / dist_straighten_leg for dist_pred_leg, dist_straighten_leg in  zip(dist_pred_legs, dist_straighten_legs)]
            #     min_index = legs_ratios.index(min(legs_ratios))
            #     dist_pred += dist_pred_legs[min_index]
            #     dist_straighten += dist_straighten_legs[min_index]

            if dist_straighten == 0. or dist_pred == 0.:
                ratio = 1.
            else:
                ratio = np.clip(dist_pred / dist_straighten, 1e-5, 1.)
                assert ratio > 0. and ratio <= 1.01, 'ratio is %.2f!'%ratio
            ratio_batch.append(ratio)
        return ratio_batch

def pts_dist(pts1, pts2):
    return np.sqrt((pts1[0]-pts2[0])**2 + (pts1[1]-pts2[1])**2)



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


import numpy as np
import matplotlib.pyplot as plt
from maskrcnn_benchmark.structures.keypoint import PersonKeypoints



def vis_keypoints(img, kps, kp_thresh=2, alpha=0.7, if_fake_scores=False, if_show_kps_score=False):
    """Visualizes keypoints (adapted from vis_one_image).
    kps has shape (4, #keypoints) where 4 rows are (x, y, logit, prob).
    """
    dataset_keypoints = PersonKeypoints.NAMES
    kp_lines = PersonKeypoints.CONNECTIONS

    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kp_lines) + 2)]
    colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

    # Perform the drawing on a copy of the image, to allow for blending.
    kp_mask = np.copy(img)

    # Draw mid shoulder / mid hip first for better visualization.
    mid_shoulder = (
        kps[:2, dataset_keypoints.index('right_shoulder')] +
        kps[:2, dataset_keypoints.index('left_shoulder')]) / 2.0
    sc_mid_shoulder = np.minimum(
        kps[2, dataset_keypoints.index('right_shoulder')],
        kps[2, dataset_keypoints.index('left_shoulder')])
    mid_hip = (
        kps[:2, dataset_keypoints.index('right_hip')] +
        kps[:2, dataset_keypoints.index('left_hip')]) / 2.0
    sc_mid_hip = np.minimum(
        kps[2, dataset_keypoints.index('right_hip')],
        kps[2, dataset_keypoints.index('left_hip')])
    nose_idx = dataset_keypoints.index('nose')
    if sc_mid_shoulder > kp_thresh and kps[2, nose_idx] > kp_thresh:
        cv2.line(
            kp_mask, tuple(mid_shoulder), tuple(kps[:2, nose_idx]),
            color=colors[len(kp_lines)], thickness=4, lineType=cv2.LINE_AA)
    if sc_mid_shoulder > kp_thresh and sc_mid_hip > kp_thresh:
        cv2.line(
            kp_mask, tuple(mid_shoulder), tuple(mid_hip),
            color=colors[len(kp_lines) + 1], thickness=4, lineType=cv2.LINE_AA)

    # Draw the keypoints.
    for l in range(len(kp_lines)):
        i1 = kp_lines[l][0]
        i2 = kp_lines[l][1]
        p1 = kps[0, i1], kps[1, i1]
        p2 = kps[0, i2], kps[1, i2]
        if kps[2, i1] > kp_thresh and kps[2, i2] > kp_thresh:
            cv2.line(
                kp_mask, p1, p2,
                color=colors[l], thickness=4, lineType=cv2.LINE_AA)
        if kps[2, i1] > kp_thresh:
            cv2.circle(
                kp_mask, p1,
                radius=2, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)
            if (not if_fake_scores) and if_show_kps_score:
                cv2.putText(
                    kp_mask, dataset_keypoints[i1]+' %.2f'%kps[2, i1], p1, cv2.FONT_HERSHEY_SIMPLEX, .3, (255, 255, 255), 1
                )
                cv2.putText(
                    kp_mask, dataset_keypoints[i1], p1, cv2.FONT_HERSHEY_SIMPLEX, .3, (255, 255, 255), 1
                )
        if kps[2, i2] > kp_thresh:
            cv2.circle(
                kp_mask, p2,
                radius=2, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)
            if (not if_fake_scores) and if_show_kps_score:
                cv2.putText(
                    kp_mask, dataset_keypoints[i2]+' %.2f'%kps[2, i2], p2, cv2.FONT_HERSHEY_SIMPLEX, .3, (255, 255, 255), 1
                )
                cv2.putText(
                    kp_mask, dataset_keypoints[i2], p2, cv2.FONT_HERSHEY_SIMPLEX, .3, (255, 255, 255), 1
                )
    # Blend the keypoints.
    return cv2.addWeighted(img, 1.0 - alpha, kp_mask, alpha, 0)
