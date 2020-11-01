import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from scipy.special import softmax
import os
from utils.utils_misc import *
from imageio import imread, imsave
from panorama_cropping_dataset_generation.debugging import drawLine, showHorizonLine, showHorizonLineFromHorizon
from dataset_cvpr import bins2roll, bins2vfov, bins2horizon, bins2pitch
from PIL import Image, ImageDraw, ImageFont
import ntpath
import logging
import random
import string

def show_cam_bbox(img, input_dict_show, save_path='.', save_name='tmp', if_show=False, if_save=True, figzoom=1., if_pause=True, if_return=False, if_not_detail=False, idx_sample=0):
    input_dict = {'tid': -1, 'yc_fit': -1, 'yc_est': -1, 'f_est_px': -1, 'f_est_mm': -1, 'pitch_est_angle': -1}
    # SHOW IMAGE, HORIZON FROM YANNICK
    input_dict.update(input_dict_show)
    # Turn interactive plotting off
    if if_show == False:
        plt.ioff()
    fig = plt.figure(figsize=(10*figzoom, 10*figzoom))
    # plt.subplot(4, 1, [1, 2, 3])
    gs = fig.add_gridspec(4, 2)
    ax1 = fig.add_subplot(gs[0:3, :])
    plt.imshow(img)
    if 'yc_est_list' in input_dict:
        add_yc_list_str = '(%s)'%(', '.join((['%.2f'%yc_est for yc_est in input_dict['yc_est_list']])))
        # print(input_dict['yc_est_list'], add_yc_list_str)
    else:
        add_yc_list_str = ''

    if 'f_pixels_est_mm_list' in input_dict:
        add_fmm_list_str = '(%s)'%(', '.join((['%.2f'%fmm_est for fmm_est in input_dict['f_pixels_est_mm_list']])))
    else:
        add_fmm_list_str = ''

    if 'v0_est_list' in input_dict:
        add_v0_list_str = '(%s)'%(', '.join((['%.2f'%v0_est for v0_est in input_dict['v0_est_list']])))
        add_v0_list_str = 'v0=%.2f'%sum(input_dict['v0_est_list']) + add_v0_list_str
        # print(input_dict['yc_est_list'], add_yc_list_str)
    else:
        add_v0_list_str = ''

    plt.title('[%s-%d-%s] H_camFit=%.2f, H_camEst=%.2f %s; f_est=%.2f mm %s; pitch=%.2f degree; %s'%\
          (input_dict['task_name'].split('_')[0], input_dict['tid'], input_dict['im_filename'][-6:], \
           input_dict['yc_fit'], input_dict['yc_est'], add_yc_list_str, input_dict['f_est_mm'], add_fmm_list_str, input_dict['pitch_est_angle'], add_v0_list_str), fontsize='small')

    W = input_dict['W']
    H = input_dict['H']

    if 'v0_cocoPredict' in input_dict:
        v0_cocoPredict = input_dict['v0_cocoPredict'] # # [top H, bottom 0],
        plt.plot([0., W/2.], [H - v0_cocoPredict, H - v0_cocoPredict], 'w--', 'linewidth', 50)

    # SHOW HORIZON EST
    if 'v0_batch_predict' in input_dict:
        v0_batch_predict = input_dict['v0_batch_predict'] # (H = top of the image, 0 = bottom of the image)
        plt.plot([0., W-1.], [H - v0_batch_predict, H - v0_batch_predict], linestyle='-', linewidth=2, color='black')
        if not if_not_detail:
            plt.text(W/4., H - v0_batch_predict, 'v0_predict %.2f'%(1. - v0_batch_predict/H), fontsize=8, weight="bold", color='black', bbox=dict(facecolor='w', alpha=0.5, boxstyle='round', edgecolor='none'))

    if 'v0_batch_est' in input_dict:
        v0_batch_est = input_dict['v0_batch_est'] # (H = top of the image, 0 = bottom of the image)
        plt.plot([0., W-1.], [H - v0_batch_est, H - v0_batch_est], linestyle='-', linewidth=2, color='lime')
        if not if_not_detail:
            plt.text(W/2., H - v0_batch_est, 'v0_est %.2f'%(1. - v0_batch_est/H), fontsize=8, weight="bold", color='lime', bbox=dict(facecolor='w', alpha=0.5, boxstyle='round', edgecolor='none'))

    if 'v0_batch_est_0' in input_dict:
        v0_batch_est = input_dict['v0_batch_est_0'] # (H = top of the image, 0 = bottom of the image)
        plt.plot([0., W-1.], [H - v0_batch_est, H - v0_batch_est], linestyle='-', linewidth=2, color='aquamarine')
        if not if_not_detail:
            plt.text(0., H - v0_batch_est, 'v0_est_0 %.2f'%(1. - v0_batch_est/H), fontsize=8, weight="bold", color='aquamarine', bbox=dict(facecolor='w', alpha=0.5, boxstyle='round', edgecolor='none'))

    ax = plt.gca()

    if 'bbox_gt' in input_dict:
        for bbox in input_dict['bbox_gt']:
            rect = Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=5, edgecolor='lime', facecolor='none')
            ax.add_patch(rect)

    if 'bbox_est' in input_dict:
        for bbox in input_dict['bbox_est']:
            rect = Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=2, edgecolor='b', facecolor='none')
            ax.add_patch(rect)

    if 'bbox_fit' in input_dict:
        for bbox in input_dict['bbox_fit']:
            rect = Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=2, edgecolor='r', facecolor='none', linestyle='--')
            ax.add_patch(rect)

    if 'bbox_h' in input_dict:
        if 'vt_camEst_N_delta_est_list' in input_dict:
            vt_camEst_N_delta_list = input_dict['vt_camEst_N_delta_est_list']
            vt_camEst_N_delta_list_sample = []
            for sample_idx in range(len(input_dict['bbox_h'])):
                vt_camEst_N_delta_list_sample.append([vt_camEst_N_delta_layer[sample_idx] for vt_camEst_N_delta_layer in vt_camEst_N_delta_list])
            if not if_not_detail:
                for bbox, vt_camEst_person_delta_person_layers in zip(input_dict['bbox_gt'], vt_camEst_N_delta_list_sample):
                    add_vt_camEst_person_delta_person_list_str = '(%s)'%(', '.join((['%.2f'%vt_camEst_person_delta_person for vt_camEst_person_delta_person in vt_camEst_person_delta_person_layers])))
                    plt.text(bbox[0], bbox[1]+bbox[3], 'Err %s'%(add_vt_camEst_person_delta_person_list_str), fontsize=7, bbox=dict(facecolor='aquamarine', alpha=0.5))

        if 'person_hs_est_list' not in input_dict:
            for y_person, bbox in zip(input_dict['bbox_h'], input_dict['bbox_gt']):
                plt.text(bbox[0], bbox[1], '%.2f'%(y_person), fontsize=12, weight="bold", bbox=dict(facecolor='white', alpha=0.5))
        else:
            person_hs_est_list = input_dict['person_hs_est_list']
            person_hs_est_list_sample = []
            for sample_idx in range(len(input_dict['bbox_h'])):
                person_hs_est_list_sample.append([person_hs_est_layer[sample_idx] for person_hs_est_layer in person_hs_est_list])
            for y_person, bbox, y_person_layers in zip(input_dict['bbox_h'], input_dict['bbox_gt'], person_hs_est_list_sample):
                if not if_not_detail:
                    add_y_person_list_str = '(%s)'%(', '.join((['%.2f'%y_person for y_person in y_person_layers])))
                else:
                    add_y_person_list_str = ''
                plt.text(bbox[0], bbox[1], '%.2f %s'%(y_person, add_y_person_list_str), fontsize=7, bbox=dict(facecolor='white', alpha=0.5))
            if 'bbox_h_canonical' in input_dict:
                for y_person, y_person_canonical, bbox, y_person_layers in zip(input_dict['bbox_h'], input_dict['bbox_h_canonical'], input_dict['bbox_gt'], person_hs_est_list_sample):
                    plt.text(bbox[0], bbox[1]+15, '%.2f C'%(y_person_canonical), fontsize=10, bbox=dict(facecolor='white', alpha=0.5))

    if 'bbox_loss' in input_dict and not if_not_detail:
        for vt_loss, bbox in zip(input_dict['bbox_loss'], input_dict['bbox_est']):
            plt.text(bbox[0]+bbox[2]-8, bbox[1]+bbox[3]/2.*1.5-8, '%.2f'%(vt_loss), fontsize=8, weight="bold", color='white', bbox=dict(facecolor='b', alpha=0.5))

    # if 'vfov_est' in input_dict:
    #     pitch = input_dict_show['pitch_est_yannick']
    #     vfov = input_dict_show['vfov_est']
    #     ctr = H*( 0.5 - 0.5*np.tan(pitch) / np.tan(vfov/2) )
    #     plt.plot([W/4., W/4.*3.], [ctr, ctr], linestyle='-.', linewidth=2, color='royalblue')
    #     # l = ctr - w*np.tan(roll)/2
    #     # r = ctr + w*np.tan(roll)/2
    #     # if debug:
    #     #     draw.text((0, 0), "vfov:{0:.2f}, pitch:{1:.2f}, roll:{2:.2f}, f_mm:{3:.2f}".format(vfov*180/np.pi, pitch*180/np.pi, roll*180/np.pi, focal_length), (255, 255, 255))
    #     # draw.line((0, l, w, r), fill=color, width=width)

    if 'v0_batch_from_pitch_vfov' in input_dict:
        v0_batch_from_pitch_vfov = input_dict_show['v0_batch_from_pitch_vfov']
        plt.plot([0., W], [H - v0_batch_from_pitch_vfov, H - v0_batch_from_pitch_vfov], linestyle='-.', linewidth=2, color='blue')
        plt.text(W/4.*3., H - v0_batch_from_pitch_vfov, 'v0_from_pitch_vfov %.2f'%(1. - v0_batch_from_pitch_vfov/H), fontsize=8, weight="bold", color='blue', bbox=dict(facecolor='w', alpha=0.5, boxstyle='round', edgecolor='none'))

    plt.xlim([0, W])
    plt.ylim([H, 0])

    # SHOW HORIZON ARGMAX
    if 'output_horizon_COCO' in input_dict and not if_not_detail:
        output_horizon = input_dict['output_horizon_COCO']
        horizon_bins = input_dict['horizon_bins']
        ax2 = fig.add_subplot(gs[3, 0])
        vis_output_softmax_argmax(output_horizon, horizon_bins, ax2, title='horizon-'+input_dict['reduce_method'])


    if 'output_camH_COCO' in input_dict and not if_not_detail:
        output_camH = input_dict['output_camH_COCO']
        camH_bins = input_dict['camH_bins']
        ax3 = fig.add_subplot(gs[3, 1])
        vis_output_softmax_argmax(output_camH, camH_bins, ax3, title='camH-'+input_dict['reduce_method'])


    # plt.axis('off')
    # ax1.set_xlim(0, W)
    # ax1.set_xlim(0, H)
    if if_show and not if_return:
        plt.show()
        print('plt.show()')
        if if_pause:
            if_delete = input(colored('Pause', 'white', 'on_blue'))


    if if_save:
        vis_path = os.path.join(save_path, save_name+'.jpg')
        fig.savefig(vis_path)
        if idx_sample == 0:
            print('Vis saved to ' + vis_path)
        # npy_path = os.path.join(save_path, 'zzNpy-' + save_name+'.npy')
        # np.save(npy_path, input_dict_show)

    if if_return:
        return fig, ax1
    else:
        plt.close(fig)

def vis_output_softmax_argmax(output_camH, camH_bins, ax, title=''):
    camH_softmax_prob = softmax(output_camH, axis=0)
    camH_max_prob = np.max(camH_softmax_prob)
    plt.plot(camH_bins, camH_softmax_prob)
    camH_softmax_estim = np.sum(camH_softmax_prob * camH_bins)
    plt.plot([camH_softmax_estim, camH_softmax_estim], [0, camH_max_prob*1.05], '--')
    camH_argmax = camH_bins[np.argmax(output_camH)]
    plt.plot([camH_argmax, camH_argmax], [0, camH_max_prob*1.05])
    plt.grid()
    ax.set_title('%s: softmax %.2f, argmax %.2f'%(title, camH_softmax_estim, camH_argmax), fontsize='small')

def show_box_kps(opt, model, img, input_dict_show, save_path='.', save_name='tmp', if_show=False, if_save=True, figzoom=1., if_pause=True, if_return=False, if_not_detail=False, idx_sample=0, select_top=True, predictions_override=None):
    image_sizes_ori = [(input_dict_show['W_batch_array'], input_dict_show['H_batch_array'])]
    if predictions_override is None:
        if 'predictions' not in input_dict_show or input_dict_show['predictions'] is None:
            return
        predictions = [input_dict_show['predictions']]
    else:
        predictions = predictions_override
    if opt.distributed:
        prediction_list, prediction_list_ori = model.module.RCNN.post_process(predictions, image_sizes_ori)
        image_batch_list_ori = [img]
        result_list, top_prediction_list = model.module.RCNN.select_and_vis_bbox(prediction_list_ori, image_batch_list_ori)
    else:
        prediction_list, prediction_list_ori = model.RCNN.post_process(predictions, image_sizes_ori)
        image_batch_list_ori = [img]
        result_list, top_prediction_list = model.RCNN.select_and_vis_bbox(prediction_list_ori, image_batch_list_ori, select_top=select_top)
    input_dict_show['result_list_pose'] = result_list
    target_list = [input_dict_show['target_maskrcnnTransform_list']]
    for idx, (target, result) in enumerate(zip(target_list, result_list)):
        # bboxes_gt = target.get_field('boxlist_ori').convert("xywh").bbox.numpy()
        if if_show == False:
            plt.ioff()
        fig = plt.figure(figsize=(10*figzoom, 10*figzoom))
        plt.imshow(result)
        plt.title('[%d-%s]'%(input_dict_show['tid'], input_dict_show['im_filename'][-6:]))
        # ax = plt.gca()
        # for bbox_gt in bboxes_gt:
        #     # print(bbox_gt)
        #     rect = Rectangle((bbox_gt[0], bbox_gt[1]), bbox_gt[2], bbox_gt[3], linewidth=2, edgecolor='lime', facecolor='none')
        #     ax.add_patch(rect)
        # plt.title('%d'%idx)
        # plt.show()
        if if_show and not if_return:
            plt.show()
            # print('plt.show()')
            if if_pause:
                if_delete = input(colored('Pause', 'white', 'on_blue'))
        if if_save:
            vis_path = os.path.join(save_path, save_name+'.jpg')
            fig.savefig(vis_path)
            if idx_sample == 0:
                print('Vis saved to ' + vis_path)
            # npy_path = os.path.join(save_path, 'zzNpy-' + save_name+'.npy')
            # np.save(npy_path, input_dict_show)
        # if if_return:
        #     return fig
        # else:
        plt.close(fig)
    return result_list, top_prediction_list

# def vis_pose(tid, save_path, im_paths)
def vis_SUN360(tid, save_path, im_paths, output_horizon, output_pitch, output_roll, output_vfov, horizon_num, pitch_num, roll_num, vfov_num, f_num, sensor_size_num, rank, \
               if_vis=True, if_save=False, min_samples=5, logger=None, prepostfix='', idx_sample=0):
    # if not(epoch != 0 and epoch != epoch_start and not opt.not_val and epoch not in epochs_evaled):
    # if not(not opt.not_val and (epoch < opt.save_every_epoch or epoch % opt.save_every_epoch == 0) and epoch not in epochs_evaled):
    #     return

    if logger is None:
        logger = logging.getLogger("vis_SUN360")
    
    if rank == 0 and if_vis:
        logger.info(green('Visualizing SUN360..... potentially save to' + save_path))
    im2_list = []
    horizon_list = []
    pitch_list = []
    roll_list = []
    vfov_list = []
    f_mm_list = []

    for idx in range(min(min_samples, len(im_paths))):
        im = Image.fromarray(imread(im_paths[idx])[:,:,:3])
        if len(im.getbands()) == 1:
            im = Image.fromarray(np.tile(np.asarray(im)[:,:,np.newaxis], (1, 1, 3)))

        horizon_disc = output_horizon[idx].detach().cpu().numpy().squeeze()
        pitch_disc = output_pitch[idx].detach().cpu().numpy().squeeze()
        roll_disc = output_roll[idx].detach().cpu().numpy().squeeze()
        vfov_disc = output_vfov[idx].detach().cpu().numpy().squeeze()
        # distortion_disc = distortion_disc.detach().cpu().numpy().squeeze()
        vfov_disc[...,0] = -35
        vfov_disc[...,-1] = -35

        horizon = bins2horizon(horizon_disc)
        pitch = bins2pitch(pitch_disc)
        roll = bins2roll(roll_disc)
        vfov = bins2vfov(vfov_disc)
        h, w = im.size
        f_pix = h / 2. / np.tan(vfov / 2.)
        sensor_size = sensor_size_num[idx]
        # sensor_size = 24 # !!!!!!
        f_mm = f_pix / h * sensor_size

        horizon_list.append(horizon)
        pitch_list.append(pitch)
        roll_list.append(roll)
        vfov_list.append(vfov)
        f_mm_list.append(f_mm)

        # horizon_from_pitch = 0.5 - 0.5*np.tan(pitch) / np.tan(vfov/2)

        if if_vis:
            im2 = np.asarray(im).copy()
            im2, _ = showHorizonLine(im2, vfov, pitch, 0., focal_length=f_mm, color=(0, 0, 255), width=4) # Blue: horizon converted from camera params WITHOUT roll
            im2 = showHorizonLineFromHorizon(im2, horizon, color=(255, 255, 0), width=4, debug=True) # Yellow: est horizon v0

            horizon_gt = horizon_num[idx]
            pitch_gt = pitch_num[idx]
            roll_gt = roll_num[idx]
            vfov_gt = vfov_num[idx]
            f_gt = f_num[idx]
            im2 = showHorizonLineFromHorizon(im2, horizon_gt, color=(255, 255, 255), width=3, debug=True, GT=True, ) # White: GT horizon without roll
            im2, _ = showHorizonLine(im2, vfov, pitch, roll, focal_length=f_mm, debug=True, color=(0, 0, 255), width=2) # Blue: horizon converted from camera params with roll
            im2, _ = showHorizonLine(im2, vfov_gt, pitch_gt, roll_gt, focal_length=f_gt, debug=True, GT=True, color=(255, 255, 255), width=1) # White: GT horizon

            if if_save:
                prefix, postfix = prepostfix.split('|')
                im_save_path = os.path.join(save_path, prefix+'tid%d-rank%d-idx%d'%(tid, rank, idx) + postfix + '-' +ntpath.basename(im_paths[idx])+'-f%.2f-GT%.2f.jpg'%(f_mm, f_gt))
                imsave(im_save_path, im2)
                if idx_sample == 0:
                    print('Vis saved to ' + im_save_path)

            im2_list.append(im2)

    # epochs_evaled.append(epoch)
    return_dict = {'horizon_list': horizon_list, 'pitch_list': pitch_list, 'roll_list': roll_list, 'vfov_list': vfov_list, 'f_mm_list': f_mm_list}
    return im2_list, return_dict

def blender_render(input_dict_show, output_RCNN, im_file, save_path='./', if_show=False, idx=0, save_name='', tmp_code='iamgroot', show_bbox=True):

    tmp_code = ''.join([random.choice(string.ascii_letters + string.digits) for n in range(32)])

    currentdir = '/home/ruizhu/Documents/Projects/adobe_rui_camera-calibration-redux'

    blender_path = '/home/ruizhu/Downloads/blender-2.79b-linux-glibc219-x86_64/blender'
    # scene_path = currentdir + '/rendering/scene_cone.blend'
    scene_path = currentdir + '/rendering/scene_chair_fix.blend'
    # script_path = currentdir + '/rendering/render_coco_rui.py'
    # script_path = currentdir + '/rendering/render_coco_rui_chair.py'
    script_path = currentdir + '/rendering/render_coco_rui_chair_all_fix.py'
    # render_file = currentdir + '/rendering/render/render_0.png'

    insertion_points_xy_list = []
    bboxes_filter = input_dict_show['bbox_gt']
    # bboxes_filter = [bboxes_filter[0]]
    for bbox in bboxes_filter:
#         insertion_points_xy_list.append([bbox[0]+bbox[2]/2., bbox[1]+bbox[3]])
        insertion_points_xy_list.append([bbox[0], bbox[1]+bbox[3]])

    # insertion_points_xy_list = insertion_points_xy_list[:1]
    npy_path = '/home/ruizhu/tmp_insert_pts_'+tmp_code
    np.save(npy_path, insertion_points_xy_list)
    bbox_hs_list = [a.item() for a in input_dict_show['bbox_h']]
    npy_path = '/home/ruizhu/tmp_bbox_hs_'+tmp_code
    np.save(npy_path, bbox_hs_list)

    im_filepath = im_file[0]
    # W = W_batch_array[0]
    # H = H_batch_array[0]
    im_ori = plt.imread(im_filepath)
    H, W = im_ori.shape[:2]
    insertion_points_x = -1
    insertion_points_y = -1
    ppitch = output_RCNN['pitch_batch_est'].cpu().numpy()[0]
    ffpixels = output_RCNN['f_pixels_batch_est'].cpu().numpy()[0]
    vvfov = output_RCNN['vfov_estim'].cpu().numpy()[0]
    hhfov = np.arctan(W / 2. / ffpixels) * 2.
    h_cam = output_RCNN['yc_est_batch'].cpu().numpy()[0]

    print('======blender, h_cam: %.2f'%h_cam)

#     if replace_v0_01 is not None:
#         ppitch = np.arctan((replace_v0_01 - 0.5) / 0.5 * np.tan(vvfov/2.))

    rendering_command = '%s %s --background --python %s'%(blender_path, scene_path, script_path)
    rendering_command_append = ' -- -img_path %s -tmp_code %s -H %d -W %d -insertion_points_x %d -insertion_points_y %d -pitch %.6f -fov_h %.6f -fov_v %.6f -cam_h %.6f'%\
        (im_filepath, tmp_code, H, W, insertion_points_x, insertion_points_y, ppitch, hhfov, vvfov, h_cam)
    rendering_command = rendering_command + rendering_command_append
#     print(rendering_command)

    os.system(rendering_command)

    render_file = currentdir + '/rendering/render/render_all_%s.png'%tmp_code

    if if_show == False:
        plt.ioff()
    fig = plt.figure(figsize=(15, 15), frameon=False)
    def full_frame(width=None, height=None):
        import matplotlib as mpl
        mpl.rcParams['savefig.pad_inches'] = 0
        figsize = None if width is None else (width, height)
        fig = plt.figure(figsize=figsize)
        ax = plt.axes([0,0,1,1], frameon=False)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.autoscale(tight=True)
    full_frame(15, 15)
    im_render = plt.imread(render_file)
    plt.imshow(im_render)
    plt.axis('off')
    plt.xlim([0, W])
    plt.ylim([H, 0])
    ax = plt.gca()
    ax.set_axis_off()
    plt.autoscale(tight=True)
#     plt.savefig(save_path + '/%s_noBbox.jpg'%(save_name), dpi = 100, bbox_inches='tight')

    input_dict = input_dict_show
    if 'bbox_gt' in input_dict:
        for bbox in input_dict['bbox_gt']:
            rect = Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=5, edgecolor='lime', facecolor='none')
            print((bbox[0], bbox[1]+bbox[3]))
            ax.add_patch(rect)


    if 'bbox_est' in input_dict:
        for bbox in input_dict['bbox_est']:
            rect = Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=3, edgecolor='b', facecolor='none')
            ax.add_patch(rect)

    v0_batch_from_pitch_vfov = input_dict_show['v0_batch_from_pitch_vfov']
    plt.plot([0., W], [H - v0_batch_from_pitch_vfov, H - v0_batch_from_pitch_vfov], linestyle='-.', linewidth=5, color='blue')

    for y_person, bbox in zip(input_dict['bbox_h'], input_dict['bbox_gt']):
        plt.text(bbox[0], bbox[1], '%.2fm'%(y_person), fontsize=20, bbox=dict(facecolor='white', alpha=0.5))

    plt.text(30, 100, r'$y_c$: %.2fm'%input_dict_show['yc_est'] + '\n' + \
                  r'$f_{mm}$: %.2fmm'%input_dict_show['f_est_mm'] + '\n' + \
                  r'$\theta$: %.2f$^\circ$'%input_dict_show['pitch_est_angle'], fontsize=50, color='white', bbox=dict(facecolor='black', alpha=0.4))

    ax.set_axis_off()
    plt.autoscale(tight=True)

    plt.savefig(save_path + '/%s.jpg'%(save_name), dpi = 100, bbox_inches='tight')

    if if_show:
        plt.show()

    plt.close(fig)


def blender_render_reverse(input_dict_show, output_RCNN, im_file, save_path='./', if_show=False, idx=0, save_name='', tmp_code='iamgroot', show_bbox=True, show_render=False, paper_zoom=False):

    tmp_code = ''.join([random.choice(string.ascii_letters + string.digits) for n in range(32)])

    currentdir = '/home/ruizhu/Documents/Projects/adobe_rui_camera-calibration-redux'
    blender_path = '/home/ruizhu/Downloads/blender-2.79b-linux-glibc219-x86_64/blender'
    # scene_path = currentdir + '/rendering/scene_cone.blend'
    scene_path = currentdir + '/rendering/scene_chair_fix.blend'
    # script_path = currentdir + '/rendering/render_coco_rui.py'
    # script_path = currentdir + '/rendering/render_coco_rui_chair.py'
    script_path = currentdir + '/rendering/render_coco_rui_chair_all_fix.py'
    # render_file = currentdir + '/rendering/render/render_0.png'

    render_file = currentdir + '/rendering/render/render_all_%s.png'%tmp_code
    reproj_file = currentdir + '/rendering/render/render_all_%s_reproj.png'%tmp_code
    from os import path
    if path.exists(render_file):
        os.system('rm ' + render_file)

    render_file = currentdir + '/rendering/render/render_all_%s.png'%tmp_code

    if if_show == False:
        plt.ioff()
    insertion_points_xy_list = []
    bboxes_filter = input_dict_show['bbox_gt']
    # bboxes_filter = [bboxes_filter[0]]
    for bbox in bboxes_filter:
#         insertion_points_xy_list.append([bbox[0]+bbox[2]/2., bbox[1]+bbox[3]])
        insertion_points_xy_list.append([bbox[0], bbox[1]+bbox[3]])

    # insertion_points_xy_list = insertion_points_xy_list[:1]
    npy_path = '/home/ruizhu/tmp_insert_pts_'+tmp_code
    np.save(npy_path, insertion_points_xy_list)
    bbox_hs_list = [a.item() for a in input_dict_show['bbox_h']]
    npy_path = '/home/ruizhu/tmp_bbox_hs_'+tmp_code
    np.save(npy_path, bbox_hs_list)

    im_filepath = im_file[0]
    im_ori = plt.imread(im_filepath)
    H, W = im_ori.shape[:2]
#     print(im_ori.shape)
#     plt.imshow(im_ori)
#     plt.show()
#     W = W_batch_array[0]
#     H = H_batch_array[0]
    insertion_points_x = -1
    insertion_points_y = -1
    ppitch = output_RCNN['pitch_batch_est'].cpu().numpy()[0]
    ffpixels = output_RCNN['f_pixels_batch_est'].cpu().numpy()[0]
    vvfov = output_RCNN['vfov_estim'].cpu().numpy()[0]
    hhfov = np.arctan(W / 2. / ffpixels) * 2.
    h_cam = output_RCNN['yc_est_batch'].cpu().numpy()[0]

    print('===fsdfs===blender, h_cam: %.2f'%h_cam)

    fig = plt.figure(figsize=(15, 15), frameon=False)
    def full_frame(width=None, height=None):
        import matplotlib as mpl
        mpl.rcParams['savefig.pad_inches'] = 0
        figsize = None if width is None else (width, height)
        fig = plt.figure(figsize=figsize)
        ax = plt.axes([0,0,1,1], frameon=False)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.autoscale(tight=True)
    full_frame(15, 15)
    im_boxes = plt.imread(im_filepath)
#     im_boxes = plt.imread(im_filepath)
    plt.imshow(im_ori)
    plt.axis('off')
    plt.xlim([0, W])
    plt.ylim([H, 0])
    ax = plt.gca()
    ax.set_axis_off()
    plt.autoscale(tight=True)
    plt.savefig(save_path + '/%s_noBbox.jpg'%(save_name), dpi = 100, bbox_inches='tight')
    plt.savefig(save_path + '/%s.jpg'%(save_name), dpi = 100, bbox_inches='tight')
    print('Original image saved to '+save_path + '/%s.jpg'%(save_name))

    if show_bbox:
        input_dict = input_dict_show
        print(input_dict.keys(), 'bbox_gt' in input_dict)
        if 'bbox_gt' in input_dict:
            for bbox in input_dict['bbox_gt']:
                rect = Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=8, edgecolor='lime', facecolor='none')
                print((bbox[0], bbox[1]+bbox[3]))
                ax.add_patch(rect)


        if 'bbox_est' in input_dict:
            for bbox in input_dict['bbox_est']:
                rect = Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=4, edgecolor='b', facecolor='none')
                ax.add_patch(rect)

        v0_batch_from_pitch_vfov = input_dict_show['v0_batch_from_pitch_vfov']
        plt.plot([0., W], [H - v0_batch_from_pitch_vfov, H - v0_batch_from_pitch_vfov], linestyle='-.', linewidth=8, color='blue')

        for y_person, bbox in zip(input_dict['bbox_h'], input_dict['bbox_gt']):
            plt.text(bbox[0], bbox[1]-10, '%.2fm'%(y_person), fontsize=40 if paper_zoom else 25, bbox=dict(facecolor='white', alpha=0.8))

        plt.text(30, 200, r'$y_c$: %.2fm'%input_dict_show['yc_est'] + '\n' + \
                  r'$f_{mm}$: %.2fmm'%input_dict_show['f_est_mm'] + '\n' + \
                  r'$\theta$: %.2f$^\circ$'%input_dict_show['pitch_est_angle'], fontsize=50, color='white', bbox=dict(facecolor='black', alpha=0.4))
#         plt.text(0, H-50, r'$y_c$: %.2fm'%input_dict_show['yc_est'] + ', ' + \
#                   r'$f_{mm}$: %.2fmm'%input_dict_show['f_est_mm']  + ', \n' + \
#                   r'$\theta$: %.2f$^\circ$'%input_dict_show['pitch_est_angle'], fontsize=50, color='white', bbox=dict(facecolor='black', alpha=0.4))


    ax.set_axis_off()
    plt.autoscale(tight=True)


    plt.savefig(reproj_file, dpi = 100, bbox_inches='tight')
    print('Rerpojection sabved to '+reproj_file)

    if if_show:
        plt.show()

    plt.close(fig)



    if not show_render:
        render_file = reproj_file
    else:
    #     if replace_v0_01 is not None:
    #         ppitch = np.arctan((replace_v0_01 - 0.5) / 0.5 * np.tan(vvfov/2.))

    #     reproj_file = '/home/ruizhu/Downloads/plain-white-background.jpg'
        rendering_command = '%s %s --background --python %s'%(blender_path, scene_path, script_path)
        rendering_command_append = ' -- -img_path %s -tmp_code %s -H %d -W %d -insertion_points_x %d -insertion_points_y %d -pitch %.6f -fov_h %.6f -fov_v %.6f -cam_h %.6f'%\
            (reproj_file, tmp_code, H, W, insertion_points_x, insertion_points_y, ppitch, hhfov, vvfov, h_cam)
        rendering_command = rendering_command + rendering_command_append
    #     print(rendering_command)

        os.system(rendering_command)


    #     plt.savefig(save_path + '/%s_reproj.jpg'%(save_name), dpi = 100, bbox_inches='tight')
    #     print('Blender rendering saved to '+save_path + '/%s.jpg'%(save_name))

        print('Loading rendered file: ', render_file)
        im_render = plt.imread(render_file)
        plt.figure(figsize=(10, 25))
        plt.imshow(im_render)
        plt.show()

    os.system('cp %s %s'%(render_file, save_path + '/%s_reproj.jpg'%(save_name)))
    print('Moded to : ', save_path + '/%s_reproj.jpg'%(save_name))

