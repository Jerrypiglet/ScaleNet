import torch
import torch.nn as nn
from torchvision import models, transforms
# from torchvision import transforms as T
from torchvision.transforms import functional as F

from termcolor import colored
from utils.utils_misc import *
import logging

# from torchvision.models.densenet import model_urls
# model_urls['densenet161'] = model_urls['densenet161'].replace('https://', 'http://')

# from maskrcnn_benchmark.structures.image_list import to_image_list
# from maskrcnn_benchmark.modeling.backbone import build_backbone
# from maskrcnn_benchmark.modeling.rpn.rpn import build_rpn
# from maskrcnn_benchmark.modeling.roi_heads.roi_heads import build_roi_heads
from utils.checkpointer import DetectronCheckpointer

from .model_part_GeneralizedRCNNRuiMod_cameraCalib_sep import GeneralizedRCNNRuiMod_cameraCalib
# from utils.train_utils import load_densenet, checkpoints_folder

class RCNN_only(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self, cfg, opt, logger, printer, rank=-1):
        super(RCNN_only, self).__init__()

        self.opt = opt
        self.cfg = cfg
        self.if_print = self.opt.debug
        self.logger = logger
        self.printer = printer
        self.rank = rank

        self.cls_names = ['horizon', 'pitch', 'roll', 'vfov', 'camH']

        torch.manual_seed(12344)
        self.RCNN = GeneralizedRCNNRuiMod_cameraCalib(cfg, opt, modules_not_build=['roi_heads'], logger=self.logger, rank=self.rank)

    def init_restore(self, old=False, if_print=False):
        save_dir = self.cfg.OUTPUT_DIR
        checkpointer = DetectronCheckpointer(self.opt, self.RCNN, save_dir=save_dir, logger=self.logger, if_print=self.if_print)
        _ = checkpointer.load(self.cfg.MODEL.RCNN_WEIGHT_BACKBONE, only_load_kws=['backbone'])

        skip_kws_CLS_HEAD = ['classifier_%s.predictor'%cls_name for cls_name in self.cls_names]
        replace_kws_CLS_HEAD = ['classifier_heads.classifier_%s'%cls_name for cls_name in self.cls_names]
        replace_with_kws_CLS_HEAD = ['roi_heads.box'] * 5
        _ = checkpointer.load(self.cfg.MODEL.RCNN_WEIGHT_CLS_HEAD, only_load_kws=replace_kws_CLS_HEAD, skip_kws=skip_kws_CLS_HEAD, replace_kws=replace_kws_CLS_HEAD, replace_with_kws=replace_with_kws_CLS_HEAD)

        # _ = checkpointer.load(self.cfg.MODEL.RCNN_WEIGHT_BOX_HEAD, only_load_kws=['roi_heads.box'], skip_kws=['box.predictor'])

        # , replace_kw=[''], replace_with_kw='roi_heads.box')
        # self.logger.info(colored("Resuming RCNN from %s"%(cfg.MODEL.RCNN_WEIGHT), 'white', 'on_magenta'))

        # for name, param in self.RCNN.named_parameters():
        #     if 'roi_heads.box.predictor' not in name and 'box.feature_extractor' not in name:
        #     # if 'roi_heads.box.predictor' not in name:
        #         param.requires_grad = False

    def turn_off_all_params(self):
        for name, param in self.named_parameters():
            param.requires_grad = False
        self.logger.info(colored('only_enable_camH_bboxPredictor', 'white', 'on_red'))

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
            print(name, param.shape, white_blue('True') if param.requires_grad else green('False'))
            if param.requires_grad:
                count_grads += 1
        self.logger.info(white_blue('---> ALL %d params; %d trainable'%(len(list(self.named_parameters())), count_grads)))
        return count_grads

    def forward(self, image_batch_list=None, list_of_bbox_list_cpu=None, list_of_oneLargeBbox_list=None, im_filename=None):
        """
        :param images224: torch.Size([8, 3, 224, 224])
        :param image_batch_list: List(np.array)
        :return:
        """
        if im_filename is not None and self.if_print:
            print('in model: im_filename', colored(im_filename, 'white', 'on_red'))

        # # print(images224.shape)
        # output_Densenet = self.forward_densenet(images224)
        # if self.if_print:
        #     print('in model: images224', colored(images224.shape, 'yellow', 'on_red'))
        #     print('in model: output_Densenet[0]', colored(output_Densenet[0].shape, 'yellow', 'on_red'))
        # # return output_Densenet

        if image_batch_list is not None:
            output_RCNN = self.RCNN(image_batch_list, list_of_bbox_list_cpu, list_of_oneLargeBbox_list)
            # prediction_list, prediction_list_ori = self.RCNN.post_process(output_RCNN['predictions'], image_batch_list)
            # result_list, top_prediction_list = self.RCNN.select_and_vis_bbox(prediction_list, prediction_list_ori, image_batch_list)
            # output_RCNN.update({'result_list': result_list, 'top_predictions': [proposal.bbox.detach().cpu().numpy() for proposal in top_prediction_list]})
            # if self.if_print:
            #     print('in model', colored(output_RCNN['class_logits_softmax'].shape, 'yellow', 'on_red'))
            return output_RCNN
        else:
            return None

    def turn_off_print(self):
        self.if_print = False

    def turn_on_print(self):
        self.if_print = True

if __name__ == '__main__':
    from torchsummary import summary
    summary(Densenet, (3, 224, 224))

