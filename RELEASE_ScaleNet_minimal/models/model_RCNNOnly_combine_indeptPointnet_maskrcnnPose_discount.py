import torch
import torch.nn as nn
from torchvision import models, transforms
# from torchvision import transforms as T
from torchvision.transforms import functional as F

from termcolor import colored
from utils.train_utils import print_white_blue, white_blue, print_red, red, green
import logging

# from torchvision.models.densenet import model_urls
# model_urls['densenet161'] = model_urls['densenet161'].replace('https://', 'http://')

# from maskrcnn_benchmark.structures.image_list import to_image_list
# from maskrcnn_benchmark.modeling.backbone import build_backbone
# from maskrcnn_benchmark.modeling.rpn.rpn import build_rpn
# from maskrcnn_benchmark.modeling.roi_heads.roi_heads import build_roi_heads
from utils.checkpointer import DetectronCheckpointer

from .model_part_GeneralizedRCNNRuiMod_cameraCalib_sep_maskrcnnPose_hybrid import GeneralizedRCNNRuiMod_cameraCalib_maskrcnnPose
from utils.utils_misc import *
import utils.model_utils as model_utils
import utils.geo_utils as geo_utils

import numpy as np
from utils import utils_coco


class RCNNOnly_combine(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self, opt, logger, printer, num_layers=2, modules_not_build=[]):
        super(RCNNOnly_combine, self).__init__()

        self.opt = opt
        self.cfg = opt.cfg
        self.good_num = self.cfg.DATA.COCO.GOOD_NUM
        self.if_print = self.opt.debug
        self.logger = logger
        self.printer = printer
        self.num_layers = num_layers

        self.cls_names = ['horizon', 'pitch', 'roll', 'vfov']
        # if self.opt.pointnet_camH:
        #     self.cls_names = ['horizon', 'pitch', 'roll', 'vfov']
        # else:
        #     self.cls_names = ['horizon', 'pitch', 'roll', 'vfov', 'camH']

        torch.manual_seed(12344)
        self.RCNN = GeneralizedRCNNRuiMod_cameraCalib_maskrcnnPose(self.cfg, self.opt, logger=self.logger, modules_not_build=modules_not_build)
        self.if_roi_h_heads = 'roi_h_heads' not in modules_not_build
        self.if_classifier_heads = 'classifier_heads' not in modules_not_build
        self.if_roi_bbox_heads = 'roi_bbox_heads' not in modules_not_build and opt.est_bbox

        if self.opt.train_cameraCls:
            if self.opt.pointnet_camH: # Only option for now
                import sys
                # sys.path.insert(0, self.cfg.MODEL.POINTNET.PATH)
                from models.model_part_pointnet_cls import CamHPointNet
                # if self.opt.direct_camH:
                #     out_channels_camH = 1
                # else:
                out_channels_camH = 1 if self.opt.direct_camH else self.cfg.MODEL.CLASSIFIER_HEAD.NUM_CLASSES
                out_channels_v0 = 1 if self.opt.direct_v0 else self.cfg.MODEL.CLASSIFIER_HEAD.NUM_CLASSES
                out_channels_fmm = 1 if self.opt.direct_fmm else self.cfg.MODEL.CLASSIFIER_HEAD.NUM_CLASSES

                self.roi_feat_dim = 16 if self.opt.pointnet_roi_feat_input else 0
                if self.opt.if_discount:
                    extra_input_dim = 2 # person_H
                else:
                    extra_input_dim = 1 # person_H, person_H_discount

                # self.net_CamHPointNet = CamHPointNet(in_channels=6, out_channels=out_channels, with_bn=True, with_FC=False, with_transform=False)
                self.net_PointNet = CamHPointNet(in_channels=7+self.roi_feat_dim+extra_input_dim, out_channels=out_channels_camH)

                if self.opt.pointnet_camH_refine or self.opt.pointnet_personH_refine:
                    self.net_PointNet_refine_layers = nn.ModuleDict([])
                    assert self.num_layers >= 2

                    # with_transform = False
                    with_transform = True


                    for layer_idx in range(self.num_layers-1):
                        # if not self.opt.pointnet_personH_refine:
                        self.net_PointNet_refine_layers.update({'net_PointNet_cls_layer_%d'%(layer_idx+1): CamHPointNet(in_channels=8+self.roi_feat_dim+extra_input_dim, out_channels=out_channels_camH, with_transform=with_transform)})
                        # else:
                        if self.opt.pointnet_personH_refine:
                            from models.model_part_pointnet_seg import CamHPersonHPointNet
                            # if self.opt.direct_camH:
                            #     num_seg_classes = 1
                            # else:
                            num_seg_classes_personH = self.cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES_h # personH logits
                            if not self.opt.pointnet_roi_feat_input_person3:
                                self.net_PointNet_refine_layers.update({'net_PointNet_seg_layer_%d'%(layer_idx+1): CamHPersonHPointNet(self.opt, in_channels=8+self.roi_feat_dim+extra_input_dim, \
                                    num_classes_camH=out_channels_camH, num_classes_v0=out_channels_v0, num_classes_fmm=out_channels_fmm, num_seg_classes=num_seg_classes_personH, with_transform=with_transform, \
                                    if_cls=False)})
                            else:
                                self.net_PointNet_refine_layers.update({'net_PointNet_seg_layer_%d'%(layer_idx+1): CamHPersonHPointNet(self.opt, in_channels=8+16+extra_input_dim, \
                                    num_classes_camH=out_channels_camH, num_classes_v0=out_channels_v0, num_classes_fmm=out_channels_fmm, num_seg_classes=num_seg_classes_personH, with_transform=with_transform, \
                                    if_cls=False)})

                        if self.opt.pointnet_v0_refine:
                            self.net_PointNet_refine_layers.update({'net_PointNet_cls_v0_layer_%d'%(layer_idx+1): CamHPointNet(in_channels=8+self.roi_feat_dim+extra_input_dim, out_channels=out_channels_v0, with_transform=with_transform)})

                        if self.opt.pointnet_fmm_refine:
                            self.net_PointNet_refine_layers.update({'net_PointNet_cls_fmm_layer_%d'%(layer_idx+1): CamHPointNet(in_channels=8+self.roi_feat_dim+extra_input_dim, out_channels=out_channels_fmm, with_transform=with_transform)})



    def init_restore(self, old=False, if_print=False):
        save_dir = self.cfg.OUTPUT_DIR
        checkpointer = DetectronCheckpointer(self.opt, self, checkpoint_all_dir=self.opt.checkpoints_folder, save_dir=save_dir, logger=self.logger, if_print=self.if_print)

        # Load backbone
        if 'SUN360RCNN' in self.cfg.MODEL.RCNN_WEIGHT_BACKBONE:
            _ = checkpointer.load(task_name=self.cfg.MODEL.RCNN_WEIGHT_BACKBONE, only_load_kws=['backbone'])
        else:
            _ = checkpointer.load(f=self.cfg.MODEL.RCNN_WEIGHT_BACKBONE, only_load_kws=['backbone'])

        # Load camera classifiers except camH
        if 'SUN360RCNN' in self.cfg.MODEL.RCNN_WEIGHT_CLS_HEAD:
            _ = checkpointer.load(task_name=self.cfg.MODEL.RCNN_WEIGHT_CLS_HEAD, only_load_kws=['classifier_heads'], skip_kws=['camH'])
        else:
            skip_kws_CLS_HEAD = ['classifier_%s.predictor'%cls_name for cls_name in self.cls_names]
            replace_kws_CLS_HEAD = ['classifier_heads.classifier_%s'%cls_name for cls_name in self.cls_names]
            replace_with_kws_CLS_HEAD = ['roi_heads.box'] * len(self.cls_names)
            _ = checkpointer.load(f=self.cfg.MODEL.RCNN_WEIGHT_CLS_HEAD, only_load_kws=replace_kws_CLS_HEAD, skip_kws=skip_kws_CLS_HEAD, replace_kws=replace_kws_CLS_HEAD, replace_with_kws=replace_with_kws_CLS_HEAD)

        # Initialize camH heads from bbox heads
        if not self.opt.pointnet_camH:
            if 'SUN360RCNN' in self.cfg.MODEL.RCNN_WEIGHT_BOX_HEAD:
                _ = checkpointer.load(task_name=self.cfg.MODEL.RCNN_WEIGHT_BOX_HEAD, only_load_kws=['camH'])
            else:
                cls_names_camH = ['camH']
                skip_kws_CLS_HEAD = ['classifier_%s.predictor'%cls_name for cls_name in cls_names_camH]
                replace_kws_CLS_HEAD = ['classifier_heads.classifier_%s'%cls_name for cls_name in cls_names_camH]
                replace_with_kws_CLS_HEAD = ['roi_heads.box']
                _ = checkpointer.load(f=self.cfg.MODEL.RCNN_WEIGHT_BOX_HEAD, only_load_kws=replace_kws_CLS_HEAD, skip_kws=skip_kws_CLS_HEAD, replace_kws=replace_kws_CLS_HEAD, replace_with_kws=replace_with_kws_CLS_HEAD)

        # # Load h heads
        if not self.RCNN.if_shared_kps_head:
            if 'SUN360RCNN' in self.cfg.MODEL.RCNN_WEIGHT_BOX_HEAD:
                _ = checkpointer.load(task_name=self.cfg.MODEL.RCNN_WEIGHT_BOX_HEAD, only_load_kws=['roi_h_heads.box'])
            else:
                _ = checkpointer.load(f=self.cfg.MODEL.RCNN_WEIGHT_BOX_HEAD, only_load_kws=['roi_h_heads.person_h'], skip_kws=['roi_h_heads.person_h.predictor'], replace_kws=['roi_h_heads.person_h'], replace_with_kws=['roi_heads.box'])

        # Load bbox heads
        if self.opt.est_bbox:
            # if 'SUN360RCNN' in self.cfg.MODEL.RCNN_WEIGHT_BOX_HEAD:
            #     # _ = checkpointer.load(task_name=self.cfg.MODEL.RCNN_WEIGHT_BOX_HEAD, only_load_kws=['roi_h_heads.box'])
            #     _ = checkpointer.load(task_name=self.cfg.MODEL.RCNN_WEIGHT_BOX_HEAD, only_load_kws=['roi_bbox_heads.box', 'rpn'])
            # else:
            #     # _ = checkpointer.load(f=self.cfg.MODEL.RCNN_WEIGHT_BOX_HEAD, only_load_kws=['roi_h_heads.box'], skip_kws=['box.predictor'], replace_kws=['roi_h_heads.box'], replace_with_kws=['roi_heads.box'])
            _ = checkpointer.load(f=self.cfg.MODEL.RCNN_WEIGHT_BOX_HEAD, only_load_kws=['roi_bbox_heads.box', 'rpn'], replace_kws=['roi_bbox_heads.box'], replace_with_kws=['roi_heads.box'])

        # if self.cfg.MODEL.KEYPOINT_ON:
        # Load h and kps head
        if self.RCNN.if_shared_kps_head:
            if self.opt.est_kps:
                if 'SUN360RCNN' in self.cfg.MODEL.RCNN_WEIGHT_KPS_HEAD:
                    _ = checkpointer.load(task_name=self.cfg.MODEL.RCNN_WEIGHT_KPS_HEAD, only_load_kws=['roi_bbox_heads.keypoint'], skip_kws=['roi_bbox_heads.keypoint.predictor_person_h'])
                else:
                    _ = checkpointer.load(f=self.cfg.MODEL.RCNN_WEIGHT_KPS_HEAD, only_load_kws=['roi_bbox_heads.keypoint'], skip_kws=['roi_bbox_heads.keypoint.predictor_person_h'], replace_kws=['roi_bbox_heads.keypoint'], replace_with_kws=['roi_heads.keypoint'])


    def forward(self, input_dict_misc=None, input_dict=None, image_batch_list=None, list_of_bbox_list_cpu=None, list_of_oneLargeBbox_list=None, im_filename=None):
        """
        :param images224: torch.Size([8, 3, 224, 224])
        :param image_batch_list: List(np.array)
        :return:
        """
        device = input_dict_misc['device']
        rank = input_dict_misc['rank']
        tid = input_dict_misc['tid']
        
        track_list = {}
        if im_filename is not None and self.if_print:
            print('in model: im_filename', colored(im_filename, 'white', 'on_red'))

        reduce_method = 'argmax' if (self.opt.argmax_val and not input_dict_misc['is_training']) else 'softmax'

        assert image_batch_list is not None

        if input_dict_misc['data'] in ['coco', 'IMDB-23K']:
            list_of_box_list_kps_gt = input_dict['target_maskrcnnTransform_list']
            list_of_box_list_kps_gt = [box_list_kps_gt.to(self.opt.device) for box_list_kps_gt in list_of_box_list_kps_gt]

            if self.RCNN.if_shared_kps_head:
                if self.opt.if_discount and self.opt.discount_from == 'GT':
                    straighten_ratios_list_GT = []
                    for box_list_kps in list_of_box_list_kps_gt:
                        straighten_ratios = self.RCNN.get_straighten_ratio_from_pred(box_list_kps, gt_input=True)
                        straighten_ratios_list_GT.append(straighten_ratios)
        else:
            list_of_box_list_kps_gt = None

        list_of_box_list_kps_gt_clone = [box_list_kps_gt.copy_with_fields(box_list_kps_gt.fields()) for box_list_kps_gt in list_of_box_list_kps_gt] if list_of_box_list_kps_gt is not None else list_of_box_list_kps_gt
        output_RCNN = self.RCNN(image_batch_list, list_of_bbox_list_cpu, list_of_oneLargeBbox_list, list_of_box_list_kps_gt=list_of_box_list_kps_gt_clone, targets=list_of_box_list_kps_gt_clone, input_data=input_dict_misc['data'])

        # if input_dict_misc is not None:
        if input_dict_misc['data'] not in ['coco', 'IMDB-23K']:
            return output_RCNN

        if self.RCNN.if_shared_kps_head:
            if self.opt.if_discount and self.opt.discount_from == 'pred':
                straighten_ratios_list_pred = []
                for box_list_kps in output_RCNN['predictions']:
                    print('==', box_list_kps.fields())
                    straighten_ratios = self.RCNN.get_straighten_ratio_from_pred(box_list_kps, gt_input=False)
                    straighten_ratios_list_pred.append(straighten_ratios)

            if self.opt.if_discount:
                if self.opt.discount_from == 'GT':
                    where_to_cal_ratios = straighten_ratios_list_GT
                elif self.opt.discount_from == 'pred':
                    where_to_cal_ratios = straighten_ratios_list_pred
                else:
                    raise ValueError('opt.discount_from must be in (\'GT\', \'pred\')!')

                if input_dict_misc['is_training']:
                    print('+++++where_to_cal_ratios', where_to_cal_ratios, self.opt.discount_from)

        straighten_ratios_list = []
        for image_idx, box_list_kps in enumerate(list_of_box_list_kps_gt_clone):
            if self.opt.if_discount:
                # straighten_ratios = self.RCNN.get_straighten_ratio_from_pred(box_list_kps, gt_input=gt_input)
                straighten_ratios = torch.tensor(where_to_cal_ratios[image_idx]).to(device).float()
            else:
                straighten_ratios = torch.ones((box_list_kps.bbox.shape[0])).to(device).float() # !!!!!!!
            straighten_ratios_list.append(straighten_ratios)
        straighten_ratios_concat = torch.cat(straighten_ratios_list)

        preds_RCNN = {}
        if list_of_oneLargeBbox_list is not None and self.if_classifier_heads:
            preds_RCNN = self.get_RCNN_predictions(output_RCNN, input_dict_misc['bins'], input_dict_misc['H_batch'], reduce_method=reduce_method, straighten_ratios_list=straighten_ratios_list)
            preds_RCNN.update({'f_pixels_est_batch_np_list': [preds_RCNN['f_pixels_batch_est'].detach().cpu().numpy()], \
                               'v0_01_est_batch_np_list': [((input_dict_misc['H_batch'] - preds_RCNN['v0_batch_est'])/input_dict_misc['H_batch']).detach().cpu().numpy()]})

        # PERSON LOSS =================
        if (not self.opt.not_rcnn) and list_of_bbox_list_cpu is not None and self.if_roi_h_heads:
            if not self.opt.loss_last_layer or (self.opt.loss_last_layer and not self.opt.pointnet_personH_refine) or self.opt.loss_person_all_layers:
                output_RCNN.update({'loss_all_person_h_list': [preds_RCNN['loss_all_person_h']]})
            else:
                output_RCNN.update({'loss_all_person_h_list': []})
            # output_RCNN.update({'loss_all_person_h_list': [preds_RCNN['loss_all_person_h']]})

        preds_RCNN['straighten_ratios_list'] = straighten_ratios_list
        if rank == 0 and self.opt.if_discount and tid % 20 == 0:
            print(straighten_ratios_list)

        if self.opt.pointnet_camH and self.if_classifier_heads:
            # bboxes_padded = bbox_list_to_bboxes_padded(list_of_bbox_list_cpu, cfg.MODEL.POINTNET.N_PAD_TO, input_dict['H_batch_array'])
            list_of_bboxes = bbox_list_to_list_of_bboxes(list_of_bbox_list_cpu)
            list_of_bboxes_cat = []
            for bbox, v0, H in zip(list_of_bboxes, preds_RCNN['v0_batch_est'], input_dict_misc['H_batch']):
                bbox_y1y2 = bbox[:, [1, 3]] # [top 0 , bottom H] [N', 2]
                bbox_y1y2_offset = bbox_y1y2 - (H - v0)  # [top 0 , bottom H]
                bbox_concat = torch.cat((bbox, bbox_y1y2_offset), 1)
                list_of_bboxes_cat.append(bbox_concat)
            bboxes_cat_padded = list_of_bboxes_to_bboxes_padded(list_of_bboxes_cat, self.good_num, input_dict_misc['H_batch'], normalize_with_H=True) # [batchsize, N, 6] (x1, y1, x2, y2, y1-v0, y2-v0], normalized by H

            if not self.opt.not_rcnn:
                person_h_list = preds_RCNN['person_h_list']
                person_h_list_input = [person_h / self.cfg.MODEL.HUMAN.MEAN - 1. for person_h in person_h_list]
                person_h_padded_input = list_of_tensor_to_tensor_padded(person_h_list_input, self.good_num)
                person_h_list_input_discount = [person_h*straighten_ratios / self.cfg.MODEL.HUMAN.MEAN - 1. for person_h, straighten_ratios in zip(person_h_list, straighten_ratios_list)]
                person_h_padded_input_discount = list_of_tensor_to_tensor_padded(person_h_list_input_discount, self.good_num)
                # person_h_padded_input = person_h_padded_input.detach() # !!!!!
            else:
                # person_h_padded_input = torch.zeros((bboxes_cat_padded.shape[0], bboxes_cat_padded.shape[1], 1), device=bboxes_cat_padded.device, dtype=bboxes_cat_padded.dtype) + 1.75
                # person_h_padded_input = person_h_padded_input / self.cfg.MODEL.HUMAN.MEAN - 1.
                person_h_list_175 = [torch.zeros((length), device=bboxes_cat_padded.device, dtype=bboxes_cat_padded.dtype) + 1.75 for length in output_RCNN['bbox_lengths']]
                person_h_list_input_175 = [person_h / self.cfg.MODEL.HUMAN.MEAN - 1. for person_h in person_h_list_175]
                person_h_padded_input = list_of_tensor_to_tensor_padded(person_h_list_input_175, self.good_num)

            v0_01_batch_est = ((input_dict_misc['H_batch'] - preds_RCNN['v0_batch_est']) / input_dict_misc['H_batch']).view(-1, 1, 1).repeat(1, self.good_num, 1)
            if self.opt.if_discount:
                input_list = [v0_01_batch_est, bboxes_cat_padded, person_h_padded_input, person_h_padded_input_discount]
            else:
                input_list = [v0_01_batch_est, bboxes_cat_padded, person_h_padded_input]
            if not self.opt.not_pointnet_detach_input:
                input_list = [input.detach() for input in input_list]
            if self.opt.pointnet_roi_feat_input or self.opt.pointnet_roi_feat_input_person3:
                roi_feats = preds_RCNN['roi_feats'] # L37, maskrcnn_rui/roi_heads_rui/box_head/roi_box_predictors.py
                roi_feats_list = roi_feats.split(output_RCNN['bbox_lengths'])
                roi_feats_padded_input = list_of_tensor_to_tensor_padded(roi_feats_list, self.good_num)
                if self.opt.pointnet_roi_feat_input:
                    input_list.append(roi_feats_padded_input)

            points_input = torch.cat(input_list, 2).permute(0, 2, 1)

            camH_cls_logits = self.net_PointNet({'points': points_input})['cls_logit']
            yc_est_batch = self.yc_logits_to_est_yc(camH_cls_logits, input_dict_misc['bins']['yc_bins_layers_list_torch'][0], input_dict_misc['bins']['yc_bins_lowHigh_list'][0], preds_RCNN['reduce_method'], direct=self.opt.direct_camH, debug=False)
            preds_RCNN.update({'yc_est_batch': yc_est_batch, 'output_yc_batch': camH_cls_logits})
            preds_RCNN.update({'yc_est_batch_np_list': [yc_est_batch.detach().cpu().numpy()]})
            if not self.opt.not_rcnn:
                   preds_RCNN.update({'person_hs_est_np_list': [[(person_hs*straighten_ratios).detach().cpu().numpy() for person_hs, straighten_ratios in zip(person_h_list, straighten_ratios_list)]]})
                   preds_RCNN.update({'person_hs_est_np_list_canonical': [[(person_hs).detach().cpu().numpy() for person_hs, straighten_ratios in zip(person_h_list, straighten_ratios_list)]]})

            # if self.opt.direct_camH:
            #     print('--tanh(camH_cls_logits)', camH_cls_logits.shape, torch.tanh(camH_cls_logits).reshape([-1]))
            #     print('------yc_est_batch', yc_est_batch.shape, yc_est_batch.reshape([-1]))

            if self.opt.pointnet_v0_refine:
                preds_RCNN['v0_batch_est_0'] = preds_RCNN['v0_batch_est'].clone()

            # vt_camEst_N_delta_paded, loss_vt, list_of_vt_camEst_N_delta = self.fit(input_dict, input_dict_misc, preds_RCNN, layer_num=layer_idx+1, if_detach=True)  #!!!!!!!!

            return_dict_fit, vt_camEst_N_delta_paded, list_of_vt_camEst_N_delta = self.fit_batch(input_dict, input_dict_misc, preds_RCNN, layer_num=0, if_detach=self.opt.pointnet_camH_refine, \
                                                                                                 if_vis=input_dict_misc['if_vis'] and not self.opt.pointnet_camH_refine,
                                                                                                 if_fit_derek=(not self.opt.pointnet_camH_refine) and self.opt.fit_derek)  #!!!!!!!!
            output_RCNN.update({'loss_vt_list': []})
            if not self.opt.loss_last_layer:
                output_RCNN['loss_vt_list'].append(return_dict_fit['loss_vt'])

            output_RCNN.update(return_dict_fit)

            track_list.update({'0': {'if_detach': self.opt.pointnet_camH_refine, \
                    'if_vis': input_dict_misc['if_vis'] and not self.opt.pointnet_camH_refine, \
                    'if_fit_derek': (not self.opt.pointnet_camH_refine) and self.opt.fit_derek, \
                    'if_append_loss': not self.opt.loss_last_layer}})

            if input_dict is not None and self.opt.pointnet_camH_refine:
                preds_RCNN.update({'vt_camEst_N_delta_np_list': []})
                for layer_idx in range(self.num_layers-1):
                    is_last_layer = layer_idx+1 == self.num_layers-1
                    # vt_camEst_N_delta_paded, loss_vt, list_of_vt_camEst_N_delta = self.fit(input_dict, input_dict_misc, preds_RCNN, layer_num=layer_idx+1, if_detach=True)  #!!!!!!!!
                    preds_RCNN['vt_camEst_N_delta_np_list'].append([vt_camEst_N_delta.detach().cpu().numpy().reshape([-1]) for vt_camEst_N_delta in list_of_vt_camEst_N_delta])

                    v0_01_batch_est = ((input_dict_misc['H_batch'] - preds_RCNN['v0_batch_est']) / input_dict_misc['H_batch']).view(-1, 1, 1).repeat(1, self.good_num, 1)

                    if self.opt.if_discount:
                        input_list2 = [v0_01_batch_est, bboxes_cat_padded, vt_camEst_N_delta_paded, person_h_padded_input, person_h_padded_input_discount]
                    else:
                        input_list2 = [v0_01_batch_est, bboxes_cat_padded, vt_camEst_N_delta_paded, person_h_padded_input]

                    # if not(layer_idx+1 == self.num_layers-1 and self.opt.loss_last_layer):
                    if not self.opt.not_pointnet_detach_input:
                        input_list2 = [input.detach() for input in input_list2]

                    if self.opt.pointnet_roi_feat_input:
                        input_list2.append(roi_feats_padded_input)
                    points_input2 = torch.cat(input_list2, 2).permute(0, 2, 1)

                    if self.opt.pointnet_roi_feat_input_person3:
                        input_list3_person = [v0_01_batch_est.detach(), bboxes_cat_padded.detach(), vt_camEst_N_delta_paded.detach(), person_h_padded_input.detach(), roi_feats_padded_input]
                        points_input3_person = torch.cat(input_list3_person, 2).permute(0, 2, 1)

                    net_PointNet_cls_layer_output = self.net_PointNet_refine_layers['net_PointNet_cls_layer_%d'%(layer_idx+1)]({'points': points_input2})
                    camH_cls_logits_delta = net_PointNet_cls_layer_output['cls_logit']
                    yc_est_batch_delta = self.yc_logits_to_est_yc(camH_cls_logits_delta, input_dict_misc['bins']['yc_bins_layers_list_torch'][layer_idx+1], input_dict_misc['bins']['yc_bins_lowHigh_list'][layer_idx+1], preds_RCNN['reduce_method'], direct=self.opt.direct_camH)
                    preds_RCNN['yc_est_batch'] = preds_RCNN['yc_est_batch'] + yc_est_batch_delta
                    preds_RCNN['yc_est_batch_np_list'].append(yc_est_batch_delta.detach().cpu().numpy())
                    # if self.opt.direct_camH:
                    #     print('=====tanh(camH_cls_logits_delta)', layer_idx, camH_cls_logits_delta.shape, torch.tanh(camH_cls_logits_delta).reshape([-1]))
                    #     print('=========yc_est_batch_delta', layer_idx, yc_est_batch_delta.shape, yc_est_batch_delta.reshape([-1]))


                    if self.opt.pointnet_fmm_refine:
                        net_PointNet_cls_fmm_layer_output = self.net_PointNet_refine_layers['net_PointNet_cls_fmm_layer_%d'%(layer_idx+1)]({'points': points_input2})

                        fmm_cls_logits_delta = net_PointNet_cls_fmm_layer_output['cls_logit']
                        fmm_est_batch_delta_percent = self.fmm_logits_to_est_fmm_delta(fmm_cls_logits_delta, input_dict_misc['bins']['fmm_bins_layers_list_torch'][layer_idx+1], \
                                                                                       input_dict_misc['bins']['fmm_bins_lowHigh_list'][layer_idx+1], reduce_method, direct=self.opt.direct_fmm)
                        f_pixels_batch_est_mm = utils_coco.fpix_to_fmm_th(preds_RCNN['f_pixels_batch_est'], input_dict_misc['H_batch'].float(), input_dict_misc['W_batch'].float()) # [batchsize]

                        fmm_est_batch_delta = fmm_est_batch_delta_percent * f_pixels_batch_est_mm.detach()
                        fpix_est_batch_delta = utils_coco.fmm_to_fpix_th(fmm_est_batch_delta, input_dict_misc['H_batch'].float(), input_dict_misc['W_batch'].float()) # [batchsize]

                        # if self.opt.direct_fmm:
                        #     print('-----tanh(fmm_est_batch_delta_percent)', layer_idx, fmm_est_batch_delta_percent.shape, torch.tanh(fmm_est_batch_delta_percent).reshape([-1]))
                        # print('---------fmm_est_batch_delta', layer_idx, fmm_est_batch_delta.shape, fmm_est_batch_delta.reshape([-1]))
                        # print('---------f_pixels_batch_est_mm', layer_idx, f_pixels_batch_est_mm.shape, f_pixels_batch_est_mm.reshape([-1]))

                        preds_RCNN['f_pixels_batch_est'] = preds_RCNN['f_pixels_batch_est'] + fpix_est_batch_delta
                        preds_RCNN['f_pixels_est_batch_np_list'].append(fpix_est_batch_delta.detach().cpu().numpy())

                    if self.opt.pointnet_v0_refine:
                        net_PointNet_cls_v0_layer_output = self.net_PointNet_refine_layers['net_PointNet_cls_v0_layer_%d'%(layer_idx+1)]({'points': points_input2})

                        v0_cls_logits_delta = net_PointNet_cls_v0_layer_output['cls_logit']
                        v0_est_batch_delta = self.v0_logits_to_est_v0_delta(v0_cls_logits_delta, input_dict_misc['bins']['v0_bins_layers_list_torch'][layer_idx+1], \
                                                                            input_dict_misc['bins']['v0_bins_lowHigh_list'][layer_idx+1], reduce_method, direct=self.opt.direct_v0, debug=False)
                        v0_est_batch_delta_H0 = - v0_est_batch_delta * input_dict_misc['H_batch'].float() # (H = top of the image, 0 = bottom of the image)
                        preds_RCNN['v0_batch_est'] = preds_RCNN['v0_batch_est'] + v0_est_batch_delta_H0

                        preds_RCNN['v0_01_est_batch_np_list'].append(v0_est_batch_delta.detach().cpu().numpy())
                        # if self.opt.direct_v0:
                        #     print('-----tanh(v0_cls_logits_delta)', layer_idx, v0_cls_logits_delta.shape, torch.tanh(v0_cls_logits_delta).reshape([-1]))
                        # print('---------v0_est_batch_delta', layer_idx, v0_est_batch_delta.shape, v0_est_batch_delta.reshape([-1]))

                    if self.opt.pointnet_personH_refine:
                        if not self.opt.pointnet_roi_feat_input_person3:
                            net_PointNet_seg_layer_output = self.net_PointNet_refine_layers['net_PointNet_seg_layer_%d'%(layer_idx+1)]({'points': points_input2})
                        else:
                            net_PointNet_seg_layer_output = self.net_PointNet_refine_layers['net_PointNet_seg_layer_%d'%(layer_idx+1)]({'points': points_input3_person})


                        personH_cls_logits_delta = net_PointNet_seg_layer_output['seg_logit'].permute(0, 2, 1) # [batchsize, N, 256]
                        personH_cls_logits_delta_N_all = torch.cat([personH_cls_logits_delta_single[:length] for personH_cls_logits_delta_single, length in zip(personH_cls_logits_delta, output_RCNN['bbox_lengths'])], 0)  # [N_all, 256]

                        all_person_hs_delta, person_h_delta_list = self.person_h_logits_to_person_h_list(personH_cls_logits_delta_N_all, input_dict_misc['bins']['human_bins_layers_list_torch'][layer_idx+1], reduce_method, output_RCNN['bbox_lengths'])

                        all_person_hs = preds_RCNN['all_person_hs'] + all_person_hs_delta
                        preds_RCNN['all_person_hs'] = all_person_hs
                        preds_RCNN['all_person_hs_layers'].append(all_person_hs)
                        preds_RCNN['person_h_list'] = all_person_hs.split(output_RCNN['bbox_lengths'])
                        loss_all_person_h = self.person_h_list_loss(all_person_hs, output_RCNN['bbox_lengths'])

                        # PERSON LOSS =================
                        if (not self.opt.loss_last_layer) or (is_last_layer and self.opt.loss_last_layer) or self.opt.loss_person_all_layers:
                            output_RCNN['loss_all_person_h_list'].append(loss_all_person_h)
                        # output_RCNN['loss_all_person_h_list'].append(loss_all_person_h)

                        preds_RCNN['person_hs_est_np_list'].append([(person_hs*straighten_ratios).detach().cpu().numpy() for person_hs, straighten_ratios in zip(person_h_delta_list, straighten_ratios_list)])
                        preds_RCNN['person_hs_est_np_list_canonical'].append([(person_hs).detach().cpu().numpy() for person_hs, straighten_ratios in zip(person_h_delta_list, straighten_ratios_list)])


                        person_h_list_input = [person_h / self.cfg.MODEL.HUMAN.MEAN - 1. for person_h in preds_RCNN['person_h_list']]
                        person_h_padded_input = list_of_tensor_to_tensor_padded(person_h_list_input, self.good_num)

                        person_h_list_input_discount = [person_h*straighten_ratios / self.cfg.MODEL.HUMAN.MEAN - 1. for person_h, straighten_ratios in zip(person_h_list, straighten_ratios_list)]
                        person_h_padded_input_discount = list_of_tensor_to_tensor_padded(person_h_list_input_discount, self.good_num)

                    # fit
                    if_vis = input_dict_misc['if_vis'] and is_last_layer
                    return_dict_fit, vt_camEst_N_delta_paded, list_of_vt_camEst_N_delta = self.fit_batch(input_dict, input_dict_misc, preds_RCNN, layer_num=layer_idx+1, if_detach=not is_last_layer, \
                                                                                           if_vis=if_vis, if_fit_derek=is_last_layer and self.opt.fit_derek)  #!!!!!!!!
                    if_append_loss = (self.opt.loss_last_layer and is_last_layer) or (not self.opt.loss_last_layer)
                    if if_append_loss:
                        output_RCNN['loss_vt_list'].append(return_dict_fit['loss_vt'])

                    track_list.update({'%d'%(layer_idx+1): {'if_detach': not is_last_layer, \
                        'if_vis': if_vis, \
                        'if_fit_derek': is_last_layer and self.opt.fit_derek, \
                        'if_append_loss': if_append_loss}})

                    for key in return_dict_fit:
                        if key in output_RCNN and key in ['camH_fit_batch', 'input_dict_show']:
                            raise ValueError('Key of %s exists in output_RCNN!'%key)

                    output_RCNN.update(return_dict_fit)

        if rank == 0 and tid % 20 == 0:
            print(track_list)

        output_RCNN.update(preds_RCNN)

        return output_RCNN

    def fit_batch(self, input_dict, input_dict_misc, preds_RCNN, layer_num=1, if_detach=False, if_vis=False, if_fit_derek=False):
        list_of_vt_camEst_N = []
        vt_loss_allBoxes_dict_cpu = {}
        vt_loss_sample_batch_list = []
        if if_fit_derek:
            camH_fit_batch = []
            vt_error_fit_allBoxes_dict_cpu = {}

        if if_vis:
            input_dict_show = {}
            input_dict_show['bbox_gt'] = []
            input_dict_show['bbox_est'] = []
            if if_fit_derek:
                input_dict_show['bbox_fit'] = []
            input_dict_show['bbox_h'] = []
            input_dict_show['bbox_geo'] = []
            input_dict_show['bbox_loss'] = []

        # loss_func = input_dict_misc['loss_func']
        loss_func = torch.nn.L1Loss(reduction='none')
        for idx, bboxes_length in enumerate(input_dict['bboxes_length_batch_array']):
            bboxes = input_dict_misc['bboxes_batch'][idx][:bboxes_length] # [N, 4]
            labels = input_dict['labels_list']
            H = input_dict_misc['H_batch'][idx]
            vc = H / 2.
            v0_est = preds_RCNN['v0_batch_est'][idx]
            pitch_est = preds_RCNN['pitch_batch_est'][idx]
            f_pixels_yannick_est = preds_RCNN['f_pixels_batch_est'][idx]
            inv_f2 = 1. / (f_pixels_yannick_est * f_pixels_yannick_est)
            yc_est = preds_RCNN['yc_est_batch'][idx]

            H_np = input_dict_misc['H_batch'][idx].cpu().numpy()
            W_np = input_dict_misc['W_batch'][idx].cpu().numpy()

            if self.opt.not_rcnn:
                h_human_s = torch.from_numpy(np.asarray([1.75] * bboxes.shape[0], dtype=np.float32)).float().to(self.opt.device)\
                        # + 0. * preds_RCNN['person_h_list'][idx]
            else:
                h_human_s = preds_RCNN['person_h_list'][idx] * preds_RCNN['straighten_ratios_list'][idx]

            if if_detach: # !!!!!!!
                v0_est = v0_est.detach()
                pitch_est = pitch_est.detach()
                f_pixels_yannick_est = f_pixels_yannick_est.detach()
                # yc_est = yc_est.detach()
                # h_human_s = h_human_s.detach()

            if if_fit_derek:
                # Fitting 现场: getting camera heights
                camH_fit_list = []
                for bbox_idx, bbox in enumerate(bboxes):
                    y_person_fit = 1.75
                    camH_fit_bbox = geo_utils.fit_camH(bbox.cpu(), H.cpu(), v0_est.cpu(), vc.cpu(), f_pixels_yannick_est.cpu(), y_person_fit)
                    camH_fit_list.append(camH_fit_bbox.detach().numpy())
                camH_fit = np.median(np.array(camH_fit_list))
                camH_fit_batch.append(camH_fit)

            vt_loss_sample_list = []
            vt_loss_ori_sample_list = []
            if if_fit_derek:
                vt_error_sample_fit_list = []

            # vt_camEst_list = []
            bbox_gt_sample = []
            bbox_est_sample = []
            if if_fit_derek:
                bbox_fit_sample = []
            bbox_h_sample = []
            bbox_geo_sample = []
            bbox_loss_sample = []


            vb_batch = H - (bboxes[:, 1] + bboxes[:, 3]) # [top H bottom 0]
            vt_gt_batch = H - bboxes[:, 1] # [top H bottom 0]
            # print(h_human_s.shape, pitch_est.shape, vb.shape)
            geo_model_input_dict = {'yc_est': yc_est, 'vb': vb_batch, 'y_person': h_human_s*torch.cos(pitch_est), 'v0': v0_est, 'vc': vc, 'f_pixels_yannick': f_pixels_yannick_est, 'pitch_est': pitch_est}
            # geo_model_input_dict = {'yc_est': yc_est, 'vb': vb, 'y_person': y_person, 'v0': v0, 'vc': vc, 'f_pixels_yannick': f_pixels_yannick}
            if self.opt.accu_model:
                vt_camEst_batch, z_batch, _ = model_utils.accu_model_batch(geo_model_input_dict, if_debug=False)  # [top H bottom 0]
            else:
                # vt_camEst = model_utils.approx_model(geo_model_input_dict)
                pass
            # print(vt_gt_batch.shape, vt_camEst_batch.shape)

            vt_loss_ori_batch = loss_func(vt_gt_batch, vt_camEst_batch) / bboxes[:, 3]
            # print('====', loss_func(vt_gt_batch, vt_camEst_batch).shape, vt_gt_batch.shape, vt_camEst_batch.shape, bboxes[:, 3].shape, vt_loss_ori_batch.shape)
            vt_loss_ori_batch = torch.where(torch.isnan(vt_loss_ori_batch), torch.zeros_like(vt_loss_ori_batch), vt_loss_ori_batch)
            # vt_loss_ori_sample_list.append(vt_loss_ori_batch.clone().detach())
            vt_loss_batch = torch.clamp(vt_loss_ori_batch, 0., self.opt.cfg.MODEL.LOSS.VT_LOSS_CLAMP)
            # vt_loss_sample_list.append(vt_loss)

            for bbox_idx, vt_loss in enumerate(vt_loss_batch.cpu()):
                vt_loss_allBoxes_dict_cpu.update({'bbox_vt_loss_layer%d-tid%04d_rank%d_%02d-%02d'%(layer_num, input_dict_misc['tid'], input_dict_misc['rank'], idx, bbox_idx): vt_loss})

            if if_vis and if_fit_derek:
                for bbox_idx, (bbox, y_person_np, vb, vt_gt, vt_camEst_np, vt_loss) in enumerate(zip(bboxes, h_human_s.cpu().detach().numpy(), vb_batch, vt_gt_batch, vt_camEst_batch.detach().cpu().numpy(), vt_loss_batch.to(input_dict_misc['cpu_device']))):
                    # vt_loss_ori = loss_func(vt_gt, vt_camEst.reshape([])) / bbox[3]
                    # vt_loss_ori = torch.where(torch.isnan(vt_loss_ori), torch.zeros_like(vt_loss_ori), vt_loss_ori)
                    # vt_loss_ori_sample_list.append(vt_loss_ori.clone().detach().reshape([]))
                    # vt_loss = torch.clamp(vt_loss_ori, 0., self.opt.cfg.MODEL.LOSS.VT_LOSS_CLAMP)
                    # vt_loss_sample_list.append(vt_loss)
                    # vt_loss_allBoxes_dict_cpu.update({'bbox_vt_loss_layer%d-tid%04d_rank%d_%02d-%02d'%(layer_num, input_dict_misc['tid'], input_dict_misc['rank'], idx, bbox_idx): vt_loss})

                    # Fitting 现场: getting vt
                    y_person_fit = 1.75
                    vt_camFit = geo_utils.fit_vt(camH_fit, vb, v0_est, vc, y_person_fit, 1. / (f_pixels_yannick_est  * f_pixels_yannick_est))
                    vt_error_fit = loss_func(vt_gt, vt_camFit) / bbox[3]
                    vt_error_fit_allBoxes_dict_cpu.update({'bbox_vt_error_fit_tid%04d_rank%d_%02d-%02d'%(input_dict_misc['tid'], input_dict_misc['rank'], idx, bbox_idx): vt_error_fit.to(input_dict_misc['cpu_device'])})
                    vt_error_sample_fit_list.append(vt_error_fit.detach().cpu().numpy().item())

                    bbox_np = bbox.cpu().numpy()
                    #     # vt_camEst_np = vt_camEst.detach().cpu().numpy()
                    vt_camFit_np = vt_camFit.detach().cpu().numpy()
                    bbox_fit_sample.append([bbox_np[0], H_np - vt_camFit_np, bbox_np[2], bbox_np[1]+bbox_np[3]-(H_np - vt_camFit_np)])
                    #     bbox_gt_sample.append([bbox_np[0], bbox_np[1], bbox_np[2], bbox_np[3]]) # [x, y (top), w, h]
                    #     bbox_est_sample.append([bbox_np[0], H_np - vt_camEst_np, bbox_np[2], bbox_np[1]+bbox_np[3]-(H_np - vt_camEst_np)])
                    #
                    #     bbox_h_sample.append(y_person_np)
                    #     bbox_geo_sample.append(geo_model_input_dict)
                    #     bbox_loss_sample.append(vt_loss.item())

            if if_vis:
                bboxes_np = bboxes.detach().cpu().numpy()
                vt_camEst_batch_np = vt_camEst_batch.detach().cpu().numpy().reshape(-1, 1)
                bbox_est_sample = np.hstack((bboxes_np[:, 0:1], H_np-vt_camEst_batch_np, bboxes_np[:, 2:3], bboxes_np[:, 1:2]+bboxes_np[:, 3:4]-(H_np - vt_camEst_batch_np)))
                bbox_h_sample = h_human_s.detach().cpu().numpy()
                bbox_loss_sample = vt_loss_batch.cpu().detach().numpy()

                input_dict_show['bbox_gt'].append(bboxes_np)
                input_dict_show['bbox_est'].append(bbox_est_sample)
                if if_fit_derek:
                    input_dict_show['bbox_fit'].append(bbox_fit_sample)
                input_dict_show['bbox_h'].append(bbox_h_sample)
                input_dict_show['bbox_geo'].append(geo_model_input_dict)
                input_dict_show['bbox_loss'].append(bbox_loss_sample)

            vt_loss_sample = torch.mean(vt_loss_batch)
            vt_loss_sample_batch_list.append(vt_loss_sample)

            vt_camEst_N = torch.clamp(vt_loss_ori_batch.reshape((-1, 1)), -self.opt.cfg.MODEL.LOSS.VT_LOSS_CLAMP, self.opt.cfg.MODEL.LOSS.VT_LOSS_CLAMP)
            list_of_vt_camEst_N.append(vt_camEst_N)

        vt_loss_batch = torch.stack(vt_loss_sample_batch_list)
        loss_vt = torch.mean(vt_loss_batch)

        vt_camEst_N_paded = list_of_tensor_to_tensor_padded(list_of_vt_camEst_N, self.good_num) # [batchsize, N, 1] (vt_est-vt), normalized by H

        return_dict = {'vt_loss_batch': vt_loss_batch, 'loss_vt': loss_vt, 'vt_loss_allBoxes_dict': vt_loss_allBoxes_dict_cpu}
        if if_fit_derek:
            return_dict.update({'camH_fit_batch': camH_fit_batch, 'vt_error_fit_allBoxes_dict': vt_error_fit_allBoxes_dict_cpu})
        if if_vis:
            return_dict.update({'input_dict_show': input_dict_show})

        return return_dict, vt_camEst_N_paded, list_of_vt_camEst_N


    # def fit(self, input_dict, input_dict_misc, preds_RCNN, layer_num=1, if_detach=False, if_vis=False, if_fit_derek=False):
    #     list_of_vt_camEst_N = []
    #     vt_loss_allBoxes_dict_cpu = {}
    #     vt_loss_sample_batch_list = []
    #     if if_fit_derek:
    #         camH_fit_batch = []
    #         vt_error_fit_allBoxes_dict_cpu = {}
    #
    #     if if_vis:
    #         input_dict_show = {}
    #         input_dict_show['bbox_gt'] = []
    #         input_dict_show['bbox_est'] = []
    #         if if_fit_derek:
    #             input_dict_show['bbox_fit'] = []
    #         input_dict_show['bbox_h'] = []
    #         input_dict_show['bbox_geo'] = []
    #         input_dict_show['bbox_loss'] = []
    #
    #     loss_func = input_dict_misc['loss_func']
    #     for idx, bboxes_length in enumerate(input_dict['bboxes_length_batch_array']):
    #         bboxes = input_dict_misc['bboxes_batch'][idx][:bboxes_length] # [N, 4]
    #         H = input_dict_misc['H_batch'][idx]
    #         vc = H / 2.
    #         v0_est = preds_RCNN['v0_batch_est'][idx]
    #         pitch_est = preds_RCNN['pitch_batch_est'][idx]
    #         f_pixels_yannick_est = preds_RCNN['f_pixels_batch_est'][idx]
    #         inv_f2 = 1. / (f_pixels_yannick_est * f_pixels_yannick_est)
    #         yc_est = preds_RCNN['yc_est_batch'][idx]
    #
    #         H_np = input_dict_misc['H_batch'][idx].cpu().numpy()
    #         W_np = input_dict_misc['W_batch'][idx].cpu().numpy()
    #
    #         if self.opt.not_rcnn:
    #             h_human_s = torch.from_numpy(np.asarray([1.75] * bboxes.shape[0], dtype=np.float32)).float().to(self.opt.device)\
    #                     # + 0. * preds_RCNN['person_h_list'][idx]
    #         else:
    #             h_human_s = preds_RCNN['person_h_list'][idx]
    #
    #         if if_detach: # !!!!!!!
    #             v0_est = v0_est.detach()
    #             pitch_est = pitch_est.detach()
    #             f_pixels_yannick_est = f_pixels_yannick_est.detach()
    #             # yc_est = yc_est.detach()
    #             # h_human_s = h_human_s.detach()
    #
    #         if if_fit_derek:
    #             # Fitting 现场: getting camera heights
    #             camH_fit_list = []
    #             for bbox_idx, bbox in enumerate(bboxes):
    #                 y_person_fit = 1.75
    #                 camH_fit_bbox = geo_utils.fit_camH(bbox.cpu(), H.cpu(), v0_est.cpu(), vc.cpu(), f_pixels_yannick_est.cpu(), y_person_fit)
    #                 camH_fit_list.append(camH_fit_bbox.detach().numpy())
    #             camH_fit = np.median(np.array(camH_fit_list))
    #             camH_fit_batch.append(camH_fit)
    #
    #         vt_loss_sample_list = []
    #         vt_loss_ori_sample_list = []
    #         if if_fit_derek:
    #             vt_error_sample_fit_list = []
    #
    #         # vt_camEst_list = []
    #         bbox_gt_sample = []
    #         bbox_est_sample = []
    #         bbox_fit_sample = []
    #         bbox_h_sample = []
    #         bbox_geo_sample = []
    #         bbox_loss_sample = []
    #
    #         for bbox_idx, (bbox, y_person) in enumerate(zip(bboxes, h_human_s)):
    #             vb = H - (bbox[1] + bbox[3]) # [top H bottom 0]
    #             vt_gt = H - bbox[1] # [top H bottom 0]
    #             geo_model_input_dict = {'yc_est': yc_est, 'vb': vb, 'y_person': y_person*torch.cos(pitch_est), 'v0': v0_est, 'vc': vc, 'f_pixels_yannick': f_pixels_yannick_est, 'pitch_est': pitch_est}
    #             # geo_model_input_dict = {'yc_est': yc_est, 'vb': vb, 'y_person': y_person, 'v0': v0, 'vc': vc, 'f_pixels_yannick': f_pixels_yannick}
    #             if self.opt.accu_model:
    #                 vt_camEst, z, negative_z = model_utils.accu_model(geo_model_input_dict, if_debug=False)  # [top H bottom 0]
    #             else:
    #                 vt_camEst = model_utils.approx_model(geo_model_input_dict)
    #
    #             print(vt_camEst, vt_camEst.shape)
    #             vt_loss_ori = loss_func(vt_gt, vt_camEst.reshape([])) / bbox[3]
    #             vt_loss_ori = torch.where(torch.isnan(vt_loss_ori), torch.zeros_like(vt_loss_ori), vt_loss_ori)
    #             vt_loss_ori_sample_list.append(vt_loss_ori.clone().detach().reshape([]))
    #             vt_loss = torch.clamp(vt_loss_ori, 0., self.opt.cfg.MODEL.LOSS.VT_LOSS_CLAMP)
    #             vt_loss_sample_list.append(vt_loss)
    #             vt_loss_allBoxes_dict_cpu.update({'bbox_vt_loss_layer%d-tid%04d_rank%d_%02d-%02d'%(layer_num, input_dict_misc['tid'], input_dict_misc['rank'], idx, bbox_idx): vt_loss.to(input_dict_misc['cpu_device'])})
    #
    #             if if_fit_derek:
    #                 # Fitting 现场: getting vt
    #                 y_person_fit = 1.75
    #                 vt_camFit = geo_utils.fit_vt(camH_fit, vb, v0_est, vc, y_person_fit, 1. / (f_pixels_yannick_est  * f_pixels_yannick_est))
    #                 vt_error_fit = loss_func(vt_gt, vt_camFit) / bbox[3]
    #                 vt_error_fit_allBoxes_dict_cpu.update({'bbox_vt_error_fit_tid%04d_rank%d_%02d-%02d'%(input_dict_misc['tid'], input_dict_misc['rank'], idx, bbox_idx): vt_error_fit.to(input_dict_misc['cpu_device'])})
    #                 vt_error_sample_fit_list.append(vt_error_fit.detach().cpu().numpy().item())
    #
    #             if if_vis:
    #                 bbox_np = bbox.cpu().numpy()
    #                 vt_camEst_np = vt_camEst.detach().cpu().numpy()
    #                 vt_camFit_np = vt_camFit.detach().cpu().numpy()
    #                 bbox_gt_sample.append([bbox_np[0], bbox_np[1], bbox_np[2], bbox_np[3]]) # [x, y (top), w, h]
    #                 bbox_est_sample.append([bbox_np[0], H_np - vt_camEst_np, bbox_np[2], bbox_np[1]+bbox_np[3]-(H_np - vt_camEst_np)])
    #                 bbox_fit_sample.append([bbox_np[0], H_np - vt_camFit_np, bbox_np[2], bbox_np[1]+bbox_np[3]-(H_np - vt_camFit_np)])
    #                 bbox_h_sample.append(y_person.cpu().detach().numpy())
    #                 bbox_geo_sample.append(geo_model_input_dict)
    #                 bbox_loss_sample.append(vt_loss.item())
    #
    #         vt_loss_sample = torch.mean(torch.stack(vt_loss_sample_list))
    #         vt_loss_sample_batch_list.append(vt_loss_sample)
    #
    #         vt_camEst_N = torch.clamp(torch.stack(vt_loss_ori_sample_list).reshape((-1, 1)), -self.opt.cfg.MODEL.LOSS.VT_LOSS_CLAMP, self.opt.cfg.MODEL.LOSS.VT_LOSS_CLAMP)
    #         list_of_vt_camEst_N.append(vt_camEst_N)
    #
    #     vt_loss_batch = torch.stack(vt_loss_sample_batch_list)
    #     loss_vt = torch.mean(vt_loss_batch)
    #
    #     vt_camEst_N_paded = list_of_tensor_to_tensor_padded(list_of_vt_camEst_N, self.good_num) # [batchsize, N, 1] (vt_est-vt), normalized by H
    #
    #     return_dict = {'vt_loss_batch': vt_loss_batch, 'loss_vt': loss_vt, 'vt_loss_allBoxes_dict': vt_loss_allBoxes_dict_cpu}
    #     if if_fit_derek:
    #         return_dict.update({'camH_fit_batch': camH_fit_batch, 'vt_error_fit_allBoxes_dict': vt_error_fit_allBoxes_dict_cpu})
    #
    #     return return_dict, vt_camEst_N_paded, list_of_vt_camEst_N


    def yc_logits_to_est_yc(self, output_yc_batch, bins, low_high, reduce_method, direct, debug=False):
        if debug:
            print('[debug yc_logits_to_est_yc]: direct', direct)
        if direct:
            # if self.opt.pointnet_camH_refine:
            #     yc_est_batch = torch.tanh(output_yc_batch) + self.cfg.MODEL.HUMAN.MEAN
            # else:
            #     yc_est_batch = torch.tanh(output_yc_batch) * 1.70 + self.cfg.MODEL.HUMAN.MEAN
            midway = (low_high[0] + low_high[1]) / 2.
            half_range = low_high[1] - midway
            yc_est_batch = torch.tanh(torch.squeeze(output_yc_batch, 1)) * half_range + midway
            if debug:
                print('[debug yc_logits_to_est_yc]: midway, low_high:', midway, low_high, low_high[1], low_high[0])
        else:
            # if self.opt.pointnet_camH_refine:
            #     yc_est_batch = prob_to_est(output_yc_batch, bins, reduce_method)
            # else:
            yc_est_batch = prob_to_est(output_yc_batch, bins, reduce_method)
        return yc_est_batch

    # def yc_logits_to_est_yc_delta(self, output_yc_batch_delta, bins, reduce_method):
    #     if self.opt.direct_camH:
    #         yc_est_batch_delta = torch.tanh(output_yc_batch_delta) * 0.70
    #     else:
    #         yc_est_batch_delta = prob_to_est(output_yc_batch_delta, bins, reduce_method)
    #     return yc_est_batch_delta

    def person_h_logits_to_person_h_list(self, class_person_H_logits, human_bins_torch, reduce_method, bbox_lengths):
        all_person_hs = prob_to_est(class_person_H_logits, human_bins_torch, reduce_method, debug=False)  # [N_bbox,]
        person_h_list = all_person_hs.split(bbox_lengths)
        # if tid % opt.summary_every_iter == 0 and if_print:
        #     print(white_blue('>>>> person_h_list:'), [person_h.detach().cpu().numpy() for person_h in person_h_list])

        return all_person_hs, person_h_list

    def person_h_list_loss(self, all_person_hs, bbox_lengths):
        prob_all_person_hs = model_utils.human_prior(all_person_hs, mean=self.cfg.MODEL.HUMAN.MEAN, std=self.cfg.MODEL.HUMAN.STD)
        prob_all_person_h_list = prob_all_person_hs.split(bbox_lengths)
        loss_all_person_h = - torch.mean(torch.stack([torch.mean(prob_all_person_h) for prob_all_person_h in prob_all_person_h_list]))
        loss_all_person_h = loss_all_person_h * self.cfg.SOLVER.PERSON_WEIGHT

        return loss_all_person_h

    def fmm_logits_to_est_fmm_delta(self, fmm_cls_logits_delta, bins, low_high, reduce_method, direct, debug=False):
        # assert not self.opt.direct_v0
        # pass
        # fmm_est_batch_delta = prob_to_est(fmm_cls_logits_delta, bins, reduce_method)
        # return fmm_est_batch_delta
        return self.yc_logits_to_est_yc(fmm_cls_logits_delta, bins, low_high, reduce_method, direct=direct, debug=debug)


    def v0_logits_to_est_v0_delta(self, v0_cls_logits_delta, bins, low_high, reduce_method, direct, debug=False):
        # assert not self.opt.direct_v0
        # v0_est_batch_delta = prob_to_est(v0_cls_logits_delta, bins, reduce_method)
        # return v0_est_batch_delta
        return self.yc_logits_to_est_yc(v0_cls_logits_delta, bins, low_high, reduce_method, direct=direct, debug=debug)

    def get_RCNN_predictions(self, output_RCNN, bins, H_batch, reduce_method, straighten_ratios_list=None):
        output_horizon = output_RCNN['output_horizon']
        output_pitch = output_RCNN['output_pitch']
        # output_roll = output_RCNN['output_roll']
        output_vfov = output_RCNN['output_vfov']

        predictions = {}

        if not self.opt.not_rcnn and self.opt.train_roi_h:
            # # if opt.direct_h:
            # #     all_person_hs = torch.tanh(output_RCNN['class_logits']) + cfg.MODEL.HUMAN.MEAN
            # # else:
            # all_person_hs = prob_to_est(output_RCNN['class_person_H_logits'], bins['human_bins_torch'], reduce_method)  # [N_bbox,]
            # person_h_list = all_person_hs.split(output_RCNN['bbox_lengths'])
            # # if tid % opt.summary_every_iter == 0 and if_print:
            # #     print(white_blue('>>>> person_h_list:'), [person_h.detach().cpu().numpy() for person_h in person_h_list])
            #
            # prob_all_person_hs = model_utils.human_prior(all_person_hs, mean=self.cfg.MODEL.HUMAN.MEAN, std=self.cfg.MODEL.HUMAN.STD)
            # prob_all_person_h_list = prob_all_person_hs.split(output_RCNN['bbox_lengths'])
            # loss_all_person_h = - torch.mean(torch.stack([torch.mean(prob_all_person_h) for prob_all_person_h in prob_all_person_h_list]))
            # loss_all_person_h = loss_all_person_h * self.cfg.SOLVER.PERSON_WEIGHT

            all_person_hs, person_h_list = self.person_h_logits_to_person_h_list(output_RCNN['class_person_H_logits'], bins['human_bins_torch'], reduce_method, output_RCNN['bbox_lengths'])
            # print([a.shape for a in straighten_ratios_list], all_person_hs.shape, [a.shape for a in person_h_list])
            # print(torch.cat(straighten_ratios_list).shape)

            loss_all_person_h = self.person_h_list_loss(all_person_hs, output_RCNN['bbox_lengths'])

            # straighten_ratios_concat = torch.cat(straighten_ratios_list)
            # all_person_hs_2 = all_person_hs * straighten_ratios_concat
            # print('--', all_person_hs.shape, all_person_hs_2.shape)
            # person_h_list_2 = [person_h * straighten_ratios for person_h, straighten_ratios in zip(person_h_list, straighten_ratios_list)]
            # print('---', [a.shape for a in person_h_list], [a.shape for a in person_h_list_2])

        # if self.opt.direct_h:
        #     yc_est_batch = torch.tanh(output_yc_batch) + self.cfg.MODEL.HUMAN.MEAN
        # else:
        #     yc_est_batch = prob_to_est(output_yc_batch, bins['yc_bins_centers_torch'], reduce_method)
        if not self.opt.pointnet_camH:
            # output_yc_batch = output_RCNN['output_camH']
            # yc_est_batch = self.yc_logits_to_est(output_yc_batch, bins['yc_bins_lowHigh_list'][0], reduce_method)
            # predictions.update({'yc_est_batch': yc_est_batch, 'output_yc_batch': output_yc_batch})
            pass

        ## Yannick/s module
        vfov_estim = prob_to_est(output_vfov, bins['vfov_bins_centers_torch'], reduce_method)
        f_estim = H_batch.float()/torch.tan(vfov_estim/2)/2
        # f_pixels_yannick_batch_ori = f_pixels_yannick_batch.clone()
        f_pixels_yannick_batch_est = f_estim
        # if tid % opt.summary_every_iter == 0 and if_print:
        #     f_mm_array_yannick, f_mm_array_est = print_f_info(f_estim, f_pixels_yannick_batch, input_dict)

        horizon_estim = prob_to_est(output_horizon, bins['horizon_bins_centers_torch'], reduce_method) # # ([Yannick] 0 = top of the image, 1 = bottom of the image)
        v0_batch_predict = H_batch.float() - horizon_estim * H_batch.float() # (H = top of the image, 0 = bottom of the image)
        # v0_batch_ori = v0_batch.clone()
        # if tid % opt.summary_every_iter == 0 and if_print:
        #     print_v0_info(v0_batch_est, v0_batch, output_pitch, H_batch)
        # v0_batch = v0_batch_est

        pitch_estim_yannick = prob_to_est(output_pitch, bins['pitch_bins_centers_torch'], reduce_method) # ([Yannick] negative of our def! horizon above center: positive)
        pitch_batch_est = - pitch_estim_yannick # (horizon above center: NEGative)
        v0_batch_from_pitch_vfov_01 = geo_utils.horizon_from_pitch_vfov(pitch_estim_yannick, vfov_estim)
        v0_batch_from_pitch_vfov = H_batch.float() - v0_batch_from_pitch_vfov_01 * H_batch.float() # (H = top of the image, 0 = bottom of the image)

        # v0_batch_est = v0_batch_predict
        v0_batch_est = v0_batch_from_pitch_vfov

        predictions.update({'f_pixels_batch_est': f_pixels_yannick_batch_est, 'vfov_estim': vfov_estim, \
                'v0_batch_predict': v0_batch_predict, 'v0_batch_from_pitch_vfov': v0_batch_from_pitch_vfov, 'v0_batch_est': v0_batch_est, \
                'pitch_batch_est': pitch_batch_est, 'pitch_estim_yannick': pitch_estim_yannick, 'reduce_method': reduce_method})
        if not self.opt.not_rcnn and self.opt.train_roi_h:
                predictions.update({'person_h_list': person_h_list, 'all_person_hs': all_person_hs, 'all_person_hs_layers': [all_person_hs], 'loss_all_person_h': loss_all_person_h})

        if self.opt.pointnet_roi_feat_input or self.opt.pointnet_roi_feat_input_person3:
            predictions.update({'roi_feats': output_RCNN['roi_feats']})

        return predictions




    def turn_off_all_params(self):
        for name, param in self.named_parameters():
            param.requires_grad = False
        self.logger.info(colored('only_enable_camH_bboxPredictor', 'white', 'on_red'))

    def turn_on_all_params(self):
        for name, param in self.named_parameters():
            param.requires_grad = True
        self.logger.info(colored('turned on all params', 'white', 'on_red'))

    def turn_on_names(self, in_names):
        for name, param in self.named_parameters():
            for in_name in in_names:
            # if 'roi_heads.box.predictor' in name or 'classifier_c' in name:
                if in_name in name:
                    param.requires_grad = True
                    self.logger.info(colored('turn_ON_in_names: ' + in_name, 'white', 'on_red'))


    def turn_off_names(self, in_names):
        for name, param in self.named_parameters():
            for in_name in in_names:
            # if 'roi_heads.box.predictor' in name or 'classifier_c' in name:
                if in_name in name:
                    param.requires_grad = False
                    self.logger.info(colored('turn_False_in_names: ' + in_name, 'white', 'on_red'))

    def print_net(self):
        count_grads = 0
        for name, param in self.named_parameters():
            # self.logger.info(name + str(param.shape) + white_blue('True') if param.requires_grad else green('False'))
            if_trainable_str = white_blue('True') if param.requires_grad else green('False')
            # if_trainable_str = 'True' if param.requires_grad else 'False'
            self.logger.info(name + str(param.shape) + if_trainable_str)
            if param.requires_grad:
                count_grads += 1
        self.logger.info(white_blue('---> ALL %d params; %d trainable'%(len(list(self.named_parameters())), count_grads)))
        return count_grads

    def turn_off_print(self):
        self.if_print = False

    def turn_on_print(self):
        self.if_print = True

if __name__ == '__main__':
    from torchsummary import summary
    summary(Densenet, (3, 224, 224))

