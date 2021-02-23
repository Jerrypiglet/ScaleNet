import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from imageio import imread, imsave
from PIL import Image, ImageDraw, ImageFont
import os
import torch

def merge_bboxes(bboxes):
    max_x1y1x2y2 = [np.inf, np.inf, -np.inf, -np.inf]
    for bbox in bboxes:
        max_x1y1x2y2 = [min(max_x1y1x2y2[0], bbox[0]), min(max_x1y1x2y2[1], bbox[1]),
                        max(max_x1y1x2y2[2], bbox[2]+bbox[0]), max(max_x1y1x2y2[3], bbox[3]+bbox[1])]
    return [max_x1y1x2y2[0], max_x1y1x2y2[1], max_x1y1x2y2[2]-max_x1y1x2y2[0], max_x1y1x2y2[3]-max_x1y1x2y2[1]]

def check_clear(ann, vis=False, debug=False):
    kps = np.asarray(ann['keypoints']).reshape(-1, 3)
    if debug:
        print(np.hstack((np.arange(kps.shape[0]).reshape((-1, 1)), kps)))

    if vis:
        plt.figure(figsize=(20, 20))
        plt.imshow(I); plt.axis('off')
        for idx, kp in enumerate(kps):
            plt.scatter(kp[0], kp[1], )
            plt.text(kp[0], kp[1], '%d'%idx, weight='bold')

    eyes_ys = kps[1:5, 1]
    eyes_ys_valid_idx = eyes_ys!=0
    eyes_ys_valid = eyes_ys[eyes_ys_valid_idx]
    ankles_ys = kps[15:17, 1]
    ankles_ys_valid_idx = ankles_ys!=0
    ankles_ys_valid = ankles_ys[ankles_ys_valid_idx]
    if eyes_ys_valid.size==0 or ankles_ys_valid.size==0:
        return False

    should_min_y_idx = np.argmin(eyes_ys_valid) # two eyes
    should_max_y_idx = np.argmax(ankles_ys_valid) # two ankles

    kps_valid = kps[kps[:, 1]!=0, :]

    if debug:
        print(eyes_ys_valid[should_min_y_idx], np.min(kps_valid[:, 1]), kps[15:17, 1][should_max_y_idx], np.max(kps_valid[:, 1]), kps[1:5, 2], kps[15:17, 2])

    return eyes_ys_valid[should_min_y_idx]==np.min(kps_valid[:, 1]) and ankles_ys_valid[should_max_y_idx]==np.max(kps_valid[:, 1]) \
        and np.any(np.logical_or(kps[1:5, 2]==1, kps[1:5, 2]==2)) and np.any(np.logical_or(kps[15:17, 2]==1, kps[15:17, 2]==2))

def check_valid_surface(cats):
    green_cats_exception = {'water':'', 'ground':'', 'solid':'', 'vegetation':['-', 'flower', 'tree'], 'floor':'', 'plant':['+', 'grass', 'leaves']}
    if_green = False
    for super_cat in green_cats_exception.keys():
        if cats[0] == super_cat:
            sub_cats = green_cats_exception[super_cat]
            if sub_cats == '':
                if_green = True
            elif sub_cats[0] == '-':
                if cats[1] not in sub_cats[1:]:
                    if_green = True
            elif sub_cats[0] == '+':
                if cats[1] in sub_cats[1:]:
                    if_green = True
    return if_green

def fpix_to_fmm(f, H, W):
    sensor_diag = 43 # full-frame sensor: 43mm
    img_diag = np.sqrt(H**2 + W**2)
    f_mm = f / img_diag * sensor_diag
    return f_mm

def fpix_to_fmm_croped(f, H, W):
    sensor_size = [24, 36]

    if H / W < 1.:
        if H / W < sensor_size[0] / sensor_size[1]:
            f_mm = f / W * sensor_size[1]
        else:
            f_mm = f / H * sensor_size[0]
    else:
        if H / W < sensor_size[0] / sensor_size[1]:
            f_mm = f / W * sensor_size[1]
        else:
            f_mm = f / H * sensor_size[0]

    return f_mm


def fmm_to_fpix(f, H, W):
    sensor_diag = np.sqrt(36**2+24**2) # full-frame sensor: 43mm (36mm * 24mm)
    img_diag = np.sqrt(H**2 + W**2)
    f_pix = f / sensor_diag * img_diag
    return f_pix

def fmm_to_fpix_th(f, H, W):
    sensor_diag = np.sqrt(36**2+24**2) # full-frame sensor: 43mm (36mm * 24mm)
    img_diag = torch.sqrt(H**2 + W**2)
    f_pix = f / sensor_diag * img_diag
    return f_pix

def fpix_to_fmm_th(f, H, W):
    sensor_diag = np.sqrt(36**2+24**2) # full-frame sensor: 43mm (36mm * 24mm)
    img_diag = torch.sqrt(H**2 + W**2)
    f_mm = f * sensor_diag / img_diag
    return f_mm

def drawLine(image, hl, hr, leftright=(None, None), color=(0,255,0), width=5):
    if np.isnan([hl, hr]).any():
        return image

    h, w, c = image.shape
    if image.dtype in (np.float32, np.float64):
        image = (image * 255).astype('uint8')

    im = Image.fromarray(image)
    draw = ImageDraw.Draw(im)

    #l = (1-hl)*h
    #r = (1-hr)*h
    l = hl*h
    r = hr*h

    b = 0
    #if leftright[0] is not None:
    #    b = leftright[0]
    #if leftright[1] is not None:
    #    w = leftright[1]

    draw.line((b, l, w, r), fill=color, width=width)
    return np.array(im)

def vis_yannick(yannick_results, image_file):
    output_dir = '/home/ruizhu/tmpdir_adobe'

    horizon_visible = yannick_results['horizon_visible'][0][0]
    pitch = yannick_results['pitch'][0][0]
    roll = yannick_results['roll'][0][0]
    vfov = yannick_results['vfov'][0][0]
    distortion = yannick_results['distortion'][0][0]
    p_bins = yannick_results['p_bins'][0]
    r_bins = yannick_results['r_bins'][0]

    im = Image.fromarray(imread(image_file))
    if len(im.getbands()) == 1:
        im = Image.fromarray(np.tile(np.asarray(im)[:,:,np.newaxis], (1, 1, 3)))
    imh, imw = im.size[:2]

    plt.figure(figsize=(30, 10))
    if horizon_visible:
        hl, hr = pitch - np.tan(roll)/2, pitch + np.tan(roll)/2
        im = drawLine(np.asarray(im), hl, hr)
        #im = Image.fromarray(im)
        #draw = ImageDraw.Draw(im)
        #draw.text((10, 10), "midpoint:{0:.2f}, roll:{1:.2f}, vfov:{2:.2f}, xi:{3:.2f}".format(float(pitch), float(roll*180/np.pi), float(vfov*180/np.pi), distortion), font=fnt, fill=(255, 255, 255, 255))
    #else:
        #draw = ImageDraw.Draw(im)
        #draw.text((10, 10), "pitch:{0:.2f}, roll:{1:.2f}, vfov:{2:.2f}, xi:{3:.2f}".format(float(pitch*180/np.pi), float(roll*180/np.pi), float(vfov*180/np.pi), distortion), font=fnt, fill=(255, 255, 255, 255))
    im = np.asarray(im)
    imsave(os.path.join(output_dir, image_file), im)

    plt.subplot(131);
    plt.imshow(im)

    # imsave(os.path.join(output_dir, debug_filename), im)

    # plt.clf()
    plt.subplot(132); plt.plot(p_bins)
    plt.subplot(133); plt.plot(r_bins)
    # plt.savefig(os.path.join(output_dir, os.path.splitext(debug_filename)[0] + "_prob.png"), bbox_inches="tight", dpi=150)
    plt.show()
    plt.close()

    # print("{}: {}, {}, {}, {}, {}".format(debug_filename, horizon_visible, pitch, roll, vfov, distortion))
