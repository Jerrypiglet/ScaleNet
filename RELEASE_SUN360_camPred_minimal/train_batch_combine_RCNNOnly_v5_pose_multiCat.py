import torch
import torch.nn as nn
import numpy as np
from termcolor import colored
# from model_utils import bboxArray_to_bboxList, get_pitch_est_v0, get_pitch_est_v0_mid, accu_model, approx_model, human_prior
import utils.model_utils as model_utils

from utils.utils_misc import *
# from dataset_coco import bin2midpointpitch
from utils import utils_coco
from utils.train_utils import print_f_info, f_pixels_to_mm
import utils.geo_utils as geo_utils
import matplotlib.pyplot as plt
 
cpu_device = torch.device("cpu")

def train_batch_combine(input_dict, model, device, opt, is_training, epoch=-1, tid=-1, loss_func=None, rank=0, logger=None, printer=None, if_SUN360=True, if_vis=False, if_debug=False):
    # if_vis = True
    if_print = is_training
    cfg = opt.cfg
    bins = input_dict['bins']

    # ========= Rui's inputs
    inputCOCO_Image_maskrcnnTransform_list = input_dict['inputCOCO_Image_maskrcnnTransform_list']
    bboxes_batch, v0_batch_offline, f_pixels_yannick_batch_offline, W_batch, H_batch, yc_batch_offline = \
        torch.from_numpy(input_dict['bboxes_batch_array']).float().to(device), input_dict['v0_batch'].to(device), input_dict['f_pixels_yannick_batch'].to(device), \
        torch.from_numpy(input_dict['W_batch_array']).to(device), torch.from_numpy(input_dict['H_batch_array']).to(device), input_dict['yc_batch'].to(device)

    # if not opt.not_rcnn:
    list_of_bbox_list_cpu = model_utils.bboxArray_to_bboxList(bboxes_batch, input_dict['bboxes_length_batch_array'], input_dict['W_batch_array'], input_dict['H_batch_array'])
    # else:
    #     list_of_bbox_list_cpu = []

    list_of_oneLargeBbox_list_cpu = model_utils.oneLargeBboxList(input_dict['W_batch_array'], input_dict['H_batch_array']) # in original image size
    list_of_oneLargeBbox_list = [bbox_list_array.to(device) for bbox_list_array in list_of_oneLargeBbox_list_cpu]


    if if_vis:
        input_dict_show = {'H': input_dict['H_batch_array'], 'W': input_dict['W_batch_array'], 'v0_cocoPredict': input_dict['v0_batch'].numpy()}

    input_dict_misc = {'bins': bins, 'is_training': is_training, 'H_batch': H_batch, 'W_batch': W_batch, 'bboxes_batch': bboxes_batch, 'loss_func': loss_func, \
                       'cpu_device': cpu_device, 'device': device, 'tid': tid, 'rank': rank, 'data': 'coco', 'if_vis': if_vis}

    output_RCNN = model(input_dict_misc=input_dict_misc, input_dict=input_dict, image_batch_list=inputCOCO_Image_maskrcnnTransform_list, \
                        list_of_bbox_list_cpu=list_of_bbox_list_cpu, list_of_oneLargeBbox_list=list_of_oneLargeBbox_list, \
                        )
    loss_dict = {}
    return_dict = {}

    # if not is_training:
    #     import matplotlib.pyplot as plt
    #     from matplotlib.patches import Rectangle
    #     image_sizes_ori = [(input_dict['W_batch_array'][0], input_dict['H_batch_array'][0])]
    #     predictions = output_RCNN['predictions']
    #     # print('====predictions', predictions, predictions[0].fields(), predictions[0].get_field('keypoints'))
    #     prediction_list, prediction_list_ori = model.RCNN.post_process(predictions, image_sizes_ori)
    #     # print('====prediction_list', prediction_list, prediction_list[0].fields(), prediction_list[0].get_field('keypoints'))
    #     image_batch_list_ori = [plt.imread(input_dict['im_file'][0])]
    #     result_list, top_prediction_list = model.RCNN.select_and_vis_bbox(prediction_list_ori, image_batch_list_ori)
    #     # print('====top_prediction_list', top_prediction_list, top_prediction_list[0].fields(), top_prediction_list[0].get_field('keypoints'))
    #
    #     target_list = input_dict['target_maskrcnnTransform_list']
    #     bboxes_gt = target_list[0].get_field('boxlist_ori').convert("xywh").bbox.numpy()
    #     plt.figure(figsize=(20, 20))
    #     plt.imshow(result_list[0])
    #     ax = plt.gca()
    #     for bbox_gt in bboxes_gt:
    #         # print(bbox_gt)
    #         rect = Rectangle((bbox_gt[0], bbox_gt[1]), bbox_gt[2], bbox_gt[3], linewidth=10, edgecolor='lime', facecolor='none')
    #         ax.add_patch(rect)
    #     plt.show()

    if opt.est_kps:
        loss_dict.update({'loss_kp': output_RCNN['detector_losses']['loss_kp'] * opt.weight_kps})
    if opt.est_bbox:
        loss_dict.update({'loss_bbox_cls': output_RCNN['detector_losses']['loss_classifier'] * opt.weight_kps})
        loss_dict.update({'loss_bbox_reg': output_RCNN['detector_losses']['loss_box_reg'] * opt.weight_kps})

    if opt.train_cameraCls:
        output_horizon = output_RCNN['output_horizon']
        output_pitch = output_RCNN['output_pitch']
        # output_roll = output_RCNN['output_roll']
        output_vfov = output_RCNN['output_vfov']

        f_pixels_yannick_batch_ori = f_pixels_yannick_batch_offline.clone()
        v0_batch_ori = v0_batch_offline.clone()

        v0_batch_est = output_RCNN['v0_batch_est']
        v0_batch_predict = output_RCNN['v0_batch_predict']
        v0_batch_from_pitch_vfov = output_RCNN['v0_batch_from_pitch_vfov']

        vfov_estim = output_RCNN['vfov_estim']
        f_pixels_yannick_batch_est = output_RCNN['f_pixels_batch_est']

        pitch_batch_est = output_RCNN['pitch_batch_est']
        pitch_estim_yannick = output_RCNN['pitch_estim_yannick']

        reduce_method = output_RCNN['reduce_method']

        if tid % opt.summary_every_iter == 0 and if_print:
            f_mm_array_est = f_pixels_to_mm(f_pixels_yannick_batch_est, input_dict)

        if opt.pointnet_camH:
            output_yc_batch = output_RCNN['output_yc_batch']
            yc_est_batch = output_RCNN['yc_est_batch']
            # yc_est_batch_list = output_RCNN['yc_est_batch_list']
            return_dict.update({'yc_est_batch': yc_est_batch, 'vt_loss_allBoxes_dict': output_RCNN['vt_loss_allBoxes_dict']})

        if opt.train_roi_h and not opt.not_rcnn:
            person_h_list = output_RCNN['person_h_list']
            all_person_hs = output_RCNN['all_person_hs']


        if opt.fit_derek:
            return_dict.update({'vt_error_fit_allBoxes_dict': output_RCNN['vt_error_fit_allBoxes_dict'], \
                            'yc_fit_batch': np.array(output_RCNN['camH_fit_batch'])})

        if if_vis and 'input_dict_show' in output_RCNN:
            input_dict_show.update(output_RCNN['input_dict_show'])

        if tid % opt.summary_every_iter == 0 and if_print:
            return_dict.update({'f_mm_batch': f_mm_array_est.reshape(-1, 1)})

        # if 'loss_vt_list' in output_RCNN:
        if opt.pointnet_camH:
            loss_vt_list = output_RCNN['loss_vt_list']
            assert len(loss_vt_list) != 0 or opt.loss_last_layer, 'len loss_vt_list cannot be 0 when not loss_last_layer!'
            # else:
            #     loss_vt_list = []
            # loss_vt_list.append(loss_vt)
            return_dict.update({'loss_vt_list': loss_vt_list})
            if tid % 20 == 0:
                print('loss_vt_list length: ', len(loss_vt_list))

            if not opt.loss_last_layer:
                loss_dict.update({'loss_vt': sum(loss_vt_list)/len(loss_vt_list)}) # mean of layers; for optimization and scheduler
            else:
                loss_dict.update({'loss_vt': loss_vt_list[-1]}) # ONLY LAST layer; for optimization and scheduler


            if opt.pointnet_camH_refine:
                loss_vt_layers_dict = {}
                for loss_idx, loss in enumerate(loss_vt_list):
                    loss_vt_layers_dict['loss_vt_layer_%d'%(loss_idx-len(loss_vt_list))] = loss

                # print('loss_vt_layers_dict', loss_vt_layers_dict)
                return_dict.update({'loss_vt_layers_dict': loss_vt_layers_dict})

            if not opt.not_rcnn:
                return_dict.update({'all_person_hs': all_person_hs})
                return_dict.update({'all_person_hs_layers': output_RCNN['all_person_hs_layers']})
                # loss_dict.update({'loss_person': loss_all_person_h})
                loss_all_person_h_list = output_RCNN['loss_all_person_h_list']
                return_dict.update({'loss_all_person_h_list': loss_all_person_h_list})
                if tid % 20 == 0:
                    print('loss_all_person_h_list length: ', len(loss_all_person_h_list))
                if not opt.loss_last_layer:
                    loss_dict.update({'loss_person': sum(loss_all_person_h_list)/len(loss_all_person_h_list)}) # mean of layers; for optimization and scheduler
                else:
                    loss_dict.update({'loss_person': loss_all_person_h_list[-1]}) # ONLY LAST layer;; for optimization and scheduler


    # ========= Yannick's inputs
    if opt.train_cameraCls and if_SUN360:
        inputSUN360_Image_maskrcnnTransform_list = input_dict['inputSUN360_Image_maskrcnnTransform_list']
        horizon_dist_gt = input_dict['horizon_dist_gt'].to(device)
        pitch_dist_gt = input_dict['pitch_dist_gt'].to(device)
        roll_dist_gt = input_dict['roll_dist_gt'].to(device)
        vfov_dist_gt = input_dict['vfov_dist_gt'].to(device)

        list_of_oneLargeBbox_list_cpu = model_utils.oneLargeBboxList(input_dict['W_list'], input_dict['H_list']) # in original image size
        list_of_oneLargeBbox_list = [bbox_list_array.to(device) for bbox_list_array in list_of_oneLargeBbox_list_cpu]

        output_RCNN_SUN360 = model(input_dict_misc={'rank': rank, 'data': 'SUN360', 'device': device, 'tid': tid}, image_batch_list=inputSUN360_Image_maskrcnnTransform_list, list_of_oneLargeBbox_list=list_of_oneLargeBbox_list)
        output_horizon = output_RCNN_SUN360['output_horizon']
        output_pitch = output_RCNN_SUN360['output_pitch']
        output_roll = output_RCNN_SUN360['output_roll']
        output_vfov = output_RCNN_SUN360['output_vfov']
        loss_horizon = nn.functional.kl_div(nn.functional.log_softmax(output_horizon, dim=1), horizon_dist_gt, reduction='batchmean')
        loss_pitch = nn.functional.kl_div(nn.functional.log_softmax(output_pitch, dim=1), pitch_dist_gt, reduction='batchmean')
        loss_roll = nn.functional.kl_div(nn.functional.log_softmax(output_roll, dim=1), roll_dist_gt, reduction='batchmean')
        loss_vfov = nn.functional.kl_div(nn.functional.log_softmax(output_vfov, dim=1), vfov_dist_gt, reduction='batchmean')

        loss_dict.update({'loss_horizon': loss_horizon * opt.weight_SUN360, \
                          'loss_pitch': loss_pitch * opt.weight_SUN360, \
                          'loss_roll': loss_roll * opt.weight_SUN360, \
                          'loss_vfov': loss_vfov * opt.weight_SUN360})
        return_dict.update({'output_horizon_SUN360': output_horizon, 'output_pitch_SUN360': output_pitch, 'output_roll_SUN360': output_roll, 'output_vfov_SUN360': output_vfov})


    # ========== Some vis
    if if_vis:
        H_nps = H_batch.cpu().numpy()
        W_nps = W_batch.cpu().numpy()
        input_dict_show['im_path'] = list(input_dict['im_file'])
        input_dict_show['im_filename'] = list(input_dict['im_filename'])
        num_samples = len(input_dict['im_file'])
        input_dict_show['tid'] = [tid] * num_samples
        input_dict_show['task_name'] = [opt.task_name] * num_samples
        input_dict_show['num_samples'] = num_samples

        if opt.est_kps:
            input_dict_show['W_batch_array'] = input_dict['W_batch_array']
            input_dict_show['H_batch_array'] = input_dict['H_batch_array']
            input_dict_show['predictions'] = output_RCNN['predictions']
            input_dict_show['target_maskrcnnTransform_list'] = input_dict['target_maskrcnnTransform_list']
            # import matplotlib.pyplot as plt
            # from matplotlib.patches import Rectangle
            # image_sizes_ori = [(w, h) for w, h in zip(input_dict['W_batch_array'], input_dict['H_batch_array'])]
            # predictions = output_RCNN['predictions']
            # # print('====predictions', predictions, predictions[0].fields(), predictions[0].get_field('keypoints'))
            # prediction_list, prediction_list_ori = model.RCNN.post_process(predictions, image_sizes_ori)
            # # print('====prediction_list', prediction_list, prediction_list[0].fields(), prediction_list[0].get_field('keypoints'))
            # image_batch_list_ori = [plt.imread(im_file) for im_file in input_dict['im_file']]
            # result_list, top_prediction_list = model.RCNN.select_and_vis_bbox(prediction_list_ori, image_batch_list_ori)
            # # print('====top_prediction_list', top_prediction_list, top_prediction_list[0].fields(), top_prediction_list[0].get_field('keypoints'))
            # input_dict_show['result_list_pose'] = result_list
            # target_list = input_dict['target_maskrcnnTransform_list']
            # for idx, (target, result) in enumerate(zip(target_list, result_list)):
            #     bboxes_gt = target.get_field('boxlist_ori').convert("xywh").bbox.numpy()
            #     plt.figure(figsize=(20, 20))
            #     plt.imshow(result)
            #     # ax = plt.gca()
            #     # for bbox_gt in bboxes_gt:
            #     #     # print(bbox_gt)
            #     #     rect = Rectangle((bbox_gt[0], bbox_gt[1]), bbox_gt[2], bbox_gt[3], linewidth=2, edgecolor='lime', facecolor='none')
            #     #     ax.add_patch(rect)
            #     plt.title('%d'%idx)
            #     plt.show()

        if opt.train_cameraCls:
            input_dict_show['reduce_method'] = [reduce_method] * num_samples
            input_dict_show['v0_batch_predict'] = v0_batch_predict.detach().cpu().numpy()  # (H = top of the image, 0 = bottom of the image)
            input_dict_show['v0_batch_from_pitch_vfov'] = v0_batch_from_pitch_vfov.detach().cpu().numpy()
            input_dict_show['v0_batch_est'] = v0_batch_est.detach().cpu().numpy()
            if 'v0_batch_est_0' in output_RCNN:
                input_dict_show['v0_batch_est_0'] = output_RCNN['v0_batch_est_0'].detach().cpu().numpy()
            f_pixels_yannick_single_est = f_pixels_yannick_batch_est.detach().cpu().numpy()
            f_pixels_yannick_single_est_mm = [utils_coco.fpix_to_fmm(f_pixels_yannick_single_est_0, H_np, W_np) for f_pixels_yannick_single_est_0, H_np, W_np in zip(f_pixels_yannick_single_est, H_nps, W_nps)]
            f_pixels_yannick_single_ori = f_pixels_yannick_batch_ori.detach().cpu().numpy()
            f_pixels_yannick_single_ori_mm = [utils_coco.fpix_to_fmm(f_pixels_yannick_single_ori_0, H_np, W_np) for f_pixels_yannick_single_ori_0, H_np, W_np in zip(f_pixels_yannick_single_ori, H_nps, W_nps)]
            # input_dict_show.update({'yc_fit': yc_batch_offline.detach().cpu().numpy(), 'yc_est': yc_est_batch.detach().cpu().numpy()})

            if len(output_RCNN['f_pixels_est_batch_np_list']) > 1: # more than one layers
                input_dict_show.update({'f_pixels_est_mm_list': [utils_coco.fpix_to_fmm(f_pixels_est_0, H_np, W_np) for f_pixels_est_0, H_np, W_np in zip(output_RCNN['f_pixels_est_batch_np_list'], H_nps, W_nps)]})
            if len(output_RCNN['v0_01_est_batch_np_list']) > 1: # more than one layers
                input_dict_show.update({'v0_est_list': output_RCNN['v0_01_est_batch_np_list']})

            input_dict_show.update({'f_est_px': f_pixels_yannick_single_est, 'f_est_mm': f_pixels_yannick_single_est_mm})
            input_dict_show.update({'f_cocoPredict': f_pixels_yannick_single_ori, 'f_cocoPredict_mm': f_pixels_yannick_single_ori_mm})
            input_dict_show.update({'pitch_est_angle': pitch_batch_est.detach().cpu().numpy()/np.pi*180.})


            # input_dict_show['horizon_est'] = {'output_horizon': nn.functional.softmax(output_horizon, dim=1).detach().cpu().numpy(), \
            #                                   'horizon_estim': horizon_estim.detach().cpu().numpy(), \
            #                                   'horizon_bins': [bins['horizon_bins_centers_torch'].cpu().numpy()] * num_samples, 'num_samples': num_samples}
            input_dict_show['output_horizon_COCO'] = output_RCNN['output_horizon'].detach().cpu().numpy()
            input_dict_show['horizon_bins'] = [bins['horizon_bins_centers_torch'].cpu().numpy()] * num_samples

            input_dict_show.update({'vfov_est': vfov_estim.detach().cpu().numpy(), 'pitch_est_yannick': pitch_estim_yannick.detach().cpu().numpy()})

            if opt.train_roi_h and opt.pointnet_camH:
                if opt.fit_derek:
                    input_dict_show.update({'yc_fit': output_RCNN['camH_fit_batch']})
                input_dict_show.update({'yc_est': yc_est_batch.detach().cpu().numpy()})

                if len(output_RCNN['yc_est_batch_np_list']) > 1: # more than one layers
                    input_dict_show.update({'yc_est_list': output_RCNN['yc_est_batch_np_list']})
                if not opt.not_rcnn:
                    if len(output_RCNN['person_hs_est_np_list']) > 1:
                        input_dict_show.update({'person_hs_est_list': output_RCNN['person_hs_est_np_list']})
                        input_dict_show.update({'labels_list': input_dict['labels_list']})
                if opt.pointnet_camH_refine:
                    if len(output_RCNN['vt_camEst_N_delta_np_list']) > 1:
                        input_dict_show.update({'vt_camEst_N_delta_est_list': output_RCNN['vt_camEst_N_delta_np_list']})
                # print('||||||||output_RCNN vt_camEst_N_delta_np_list', output_RCNN['vt_camEst_N_delta?_np_list'])
                # print('||||||||output_RCNN person_hs_est_np_list', output_RCNN['person_hs_est_np_list'])

                if not opt.direct_camH:
                    input_dict_show['output_camH_COCO'] = output_RCNN['output_yc_batch'].detach().cpu().numpy()
                    input_dict_show['camH_bins'] = [bins['yc_bins_centers_torch'].cpu().numpy()] * num_samples

        return_dict['input_dict_show'] = input_dict_show

    return loss_dict, return_dict