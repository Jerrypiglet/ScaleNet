# Evaluate
import os
import torch
import torch.nn as nn
import numpy as np
import shutil
from tqdm import tqdm
import utils.model_utils as model_utils
import torch.distributed as dist
from utils.train_utils import sum_bbox_ratios, reduce_loss_dict

import utils.vis_utils as vis_utils
from maskrcnn_benchmark.utils.comm import synchronize, get_rank
from utils.utils_misc import *




def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', epoch=0, tid=0, checkpoint_path='.'):
    if epoch is not None:
        filename = os.path.join(checkpoint_path, filename.replace('.pth.tar', '_epoch%d_tid%d_.pth.tar'%(epoch, tid)))
    torch.save(state, filename)
    print('Saved to ...' + filename)
    if is_best:
        print("best", state["eval_loss"])
        shutil.copyfile(filename, os.path.join(checkpoint_path, 'model_best.pth.tar'))
    else:
        print("NOT best", state["eval_loss"])

def eval_epoch_cvpr_RCNN(model, validation_loader, epoch, tid, device, writer, scheduler, best_loss, logger, opt, max_iter=-1, if_vis=False, if_loss=True, prepostfix='', savepath=''):
    eval_loss_list, eval_loss_horizon_list, eval_loss_pitch_list, eval_loss_roll_list, eval_loss_vfov_list = [], [], [], [], []
    if opt.distributed:
        rank = dist.get_rank()
    else:
        rank = 0

    loss_func = nn.CrossEntropyLoss()
    model.eval()

    return_dict_list = []
    return_dict_epoch = {}

    with torch.no_grad():
        with tqdm(total=len(validation_loader)) as t:
            t.set_description('Ep.{} Eval'.format(epoch))

            for i, (im_paths_SUN360, inputSUN360_Image_yannickTransform_list, \
                horizon_dist_gt, pitch_dist_gt, roll_dist_gt, vfov_dist_gt, metadata, \
                pitch_list, roll_list, vfov_list, horizon_list, focal_length_35mm_eq_list, sensor_size_list, W_list, H_list, idx1, idx2, idx3, idx4) in enumerate(validation_loader):

                horizon_dist_gt, pitch_dist_gt, roll_dist_gt, vfov_dist_gt = horizon_dist_gt.to(device), pitch_dist_gt.to(device), roll_dist_gt.to(device), vfov_dist_gt.to(device)
                horizon_idx_gt, pitch_idx_gt, roll_idx_gt, vfov_idx_gt = idx1.to(device), idx2.to(device), idx3.to(device), idx4.to(device)

                list_of_oneLargeBbox_list_cpu = model_utils.oneLargeBboxList(W_list, H_list)
                list_of_oneLargeBbox_list = [bbox_list_array.to(device) for bbox_list_array in list_of_oneLargeBbox_list_cpu]

                input_dict_misc = {'rank': rank, 'data': 'SUN360', 'device': device, 'tid': tid}
                output_RCNN = model(input_dict_misc = input_dict_misc, image_batch_list=inputSUN360_Image_yannickTransform_list, list_of_oneLargeBbox_list=list_of_oneLargeBbox_list)
                output_horizon = output_RCNN['output_horizon']
                output_pitch = output_RCNN['output_pitch']
                output_roll = output_RCNN['output_roll']
                output_vfov = output_RCNN['output_vfov']

                if if_loss:
                    # loss_horizon = nn.functional.kl_div(nn.functional.log_softmax(output_horizon, dim=1), horizon_dist_gt, reduction='batchmean')
                    # loss_pitch = nn.functional.kl_div(nn.functional.log_softmax(output_pitch, dim=1), pitch_dist_gt, reduction='batchmean')
                    # loss_roll = nn.functional.kl_div(nn.functional.log_softmax(output_roll, dim=1), roll_dist_gt, reduction='batchmean')
                    # loss_vfov = nn.functional.kl_div(nn.functional.log_softmax(output_vfov, dim=1), vfov_dist_gt, reduction='batchmean')
                    loss_horizon = loss_func(output_horizon, horizon_idx_gt)
                    loss_pitch = loss_func(output_pitch, pitch_idx_gt)
                    loss_roll = loss_func(output_roll, roll_idx_gt)
                    loss_vfov = loss_func(output_vfov, vfov_idx_gt)

                    loss_dict = {'loss_horizon': loss_horizon, 'loss_pitch': loss_pitch, \
                                 'loss_roll': loss_roll, 'loss_vfov': loss_vfov}
                    loss_dict_reduced = reduce_loss_dict(loss_dict, mark=i, logger=logger)
                    loss_reduced = sum(loss for loss in loss_dict_reduced.values())

                    # loss = loss_horizon + loss_pitch + loss_roll + loss_vfov

                    eval_loss_list.append(loss_reduced.item())
                    eval_loss_horizon_list.append(loss_dict_reduced['loss_horizon'].item())
                    eval_loss_pitch_list.append(loss_dict_reduced['loss_pitch'].item())
                    eval_loss_roll_list.append(loss_dict_reduced['loss_roll'].item())
                    eval_loss_vfov_list.append(loss_dict_reduced['loss_vfov'].item())
                    toreport = {
                         "loss": loss_reduced.item(),
                         "horizon": loss_dict_reduced['loss_horizon'].item(),
                         "pitch": loss_dict_reduced['loss_pitch'].item(),
                         "roll": loss_dict_reduced['loss_roll'].item(),
                         "vfov": loss_dict_reduced['loss_vfov'].item(),
                     }
                    t.set_postfix(**toreport)
                    t.update()

                if i < 5 and if_vis:
                    _, return_dict = vis_utils.vis_SUN360(tid, savepath, im_paths_SUN360, output_horizon, output_pitch, output_roll, output_vfov, horizon_list, pitch_list, roll_list, vfov_list, focal_length_35mm_eq_list, sensor_size_list, rank, \
                               if_vis=i < 10, if_save=True, logger=logger, prepostfix=prepostfix, idx_sample=i)
                    return_dict_list.append(return_dict)

                synchronize()

                if max_iter != -1 and i > max_iter:
                    break

            if if_loss:
                eval_loss_sum_SUN360 = sum(eval_loss_list) / len(validation_loader)
                eval_loss_horizon = sum(eval_loss_horizon_list) / len(validation_loader)
                eval_loss_pitch = sum(eval_loss_pitch_list) / len(validation_loader)
                eval_loss_roll = sum(eval_loss_roll_list) / len(validation_loader)
                eval_loss_vfov = sum(eval_loss_vfov_list) / len(validation_loader)
                t.set_postfix(loss=eval_loss_sum_SUN360)

                if rank == 0:
                    writer.add_scalar('loss_eval/eval_loss_sum_SUN360', eval_loss_sum_SUN360, tid)
                    writer.add_scalar('loss_eval/eval_loss_horizon', eval_loss_horizon, tid)
                    writer.add_scalar('loss_eval/eval_loss_pitch', eval_loss_pitch, tid)
                    writer.add_scalar('loss_eval/eval_loss_roll', eval_loss_roll, tid)
                    writer.add_scalar('loss_eval/eval_loss_vfov', eval_loss_vfov, tid)

                    writer.add_histogram('loss/eval_loss_hist', np.asarray(eval_loss_list), tid, bins="doane")
                    writer.add_histogram('loss/eval_loss_horizon_hist', np.asarray(eval_loss_horizon_list), tid, bins="doane")
                    writer.add_histogram('loss/eval_loss_pitch_hist', np.asarray(eval_loss_pitch_list), tid, bins="doane")
                    writer.add_histogram('loss/eval_loss_roll_hist', np.asarray(eval_loss_roll_list), tid, bins="doane")
                    writer.add_histogram('loss/eval_loss_vfov_hist', np.asarray(eval_loss_vfov_list), tid, bins="doane")

                    return_dict_epoch.update({'eval_loss_sum_SUN360': eval_loss_sum_SUN360})

                    # writer.flush()

                    if if_vis and rank == 0:
                        horizon_all = merge_list_of_lists([return_dict['horizon_list'] for return_dict in return_dict_list])
                        pitch_all = merge_list_of_lists([return_dict['pitch_list'] for return_dict in return_dict_list])
                        roll_all = merge_list_of_lists([return_dict['roll_list'] for return_dict in return_dict_list])
                        vfov_all = merge_list_of_lists([return_dict['vfov_list'] for return_dict in return_dict_list])
                        f_mm_all = merge_list_of_lists([return_dict['f_mm_list'] for return_dict in return_dict_list])

                        writer.add_histogram('dist/horizon_all', np.asarray(horizon_all), tid, bins="doane")
                        writer.add_histogram('dist/pitch_all', np.asarray(pitch_all)/np.pi*180., tid, bins="doane")
                        writer.add_histogram('dist/roll_all', np.asarray(roll_all)/np.pi*180., tid, bins="doane")
                        writer.add_histogram('dist/vfov_all', np.asarray(vfov_all)/np.pi*180., tid, bins="doane")
                        writer.add_histogram('dist/f_mm_all', np.asarray(f_mm_all), tid, bins="doane")

                        # writer.flush()

    # if if_loss and scheduler is not None:
    #     scheduler.step(eval_loss)
    #
    #     # Save checkpoint
    #     is_best = False
    #     if eval_loss < best_loss:
    #         is_best = True
    #         best_loss = eval_loss
    #
    #     # checkpoint = {
    #     #      'epoch': epoch,
    #     #      'tid': tid,
    #     #      'state_dict': model.state_dict(),
    #     #      'train_loss': train_loss,
    #     #      'eval_loss': eval_loss,
    #     #      'optimizer': optimizer.state_dict(),
    #     # }
    #     #
    #     # save_checkpoint(checkpoint, is_best, epoch=epoch, tid=tid, checkpoint_path=checkpoint_path)
    #     # del checkpoint
    #
    #     model.train()

    return return_dict_epoch
