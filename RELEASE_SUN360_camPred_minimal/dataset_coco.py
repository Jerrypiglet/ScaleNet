import os, sys
import numpy as np
import torch
import json
import random
from PIL import Image
from glob import glob
from scipy.stats import norm
# from torchvision import transforms

from imageio import imread
from tqdm import tqdm
from scipy.io import loadmat
from termcolor import colored
import time


def getBins(minval, maxval, sigma, alpha, beta, kappa):
    """Remember, bin 0 = below value! last bin mean >= maxval"""
    x = np.linspace(minval, maxval, 255)

    rv = norm(0, sigma)
    pdf = rv.pdf(x)
    pdf /= (pdf.max())
    pdf *= alpha
    pdf = pdf.max()*beta - pdf
    cumsum = np.cumsum(pdf)
    cumsum = cumsum / cumsum.max() * kappa
    cumsum -= cumsum[pdf.size//2]

    return cumsum


# def getHorizonLineFromAngles(pitch, roll, FoV, im_h, im_w):
#     midpoint = getMidpointFromAngle(pitch, FoV)
#     dh = getDeltaHeightFromRoll(roll, im_h, im_w)
#     return midpoint + dh, midpoint - dh


# def getMidpointFromAngle(pitch, FoV):
#     return ( 0.5 + 0.5*np.tan(pitch) / np.tan(FoV/2) )


# def getDeltaHeightFromRoll(roll, im_h, im_w):
#     "The height distance of horizon from the midpoint at image left/right border intersection."""
#     return im_w/im_h*np.tan(roll) / 2


# def getOffset(pitch, roll, vFoV, im_h, im_w):
#     hl, hr = getHorizonLineFromAngles(pitch, roll, vFoV, im_h, im_w)
#     midpoint = (hl + hr) / 2.
#     #slope = np.arctan(hr - hl)
#     offset = (midpoint - 0.5) / np.sqrt( 1 + (hr - hl)**2 )
#     return offset


# def midpointpitch2bin(midpoint, pitch):
#     if np.isnan(midpoint):
#         if pitch < 0:
#             return np.digitize(pitch, pitch_bins_low)
#         else:
#             return np.digitize(pitch, pitch_bins_high) + 224
#     assert 0 <= midpoint <= 1
#     return int(midpoint*192) + 32


def bin2midpointpitch(bins):
    pos = np.squeeze(bins.argmax(axis=-1))
    if pos < 31:
        return False, pitch_bins_low[pos]
    elif pos == 255:
        return False, np.pi/6
    elif pos >= 224:
        return False, pitch_bins_high[pos - 224]
    else:
        return True, (pos - 32)/192


# def bins2roll(bins):
#     idxes = np.argmax(bins, axis=bins.ndim - 1)
#     return roll_bins_centers[idxes]


# def bins2vfov(bins):
#     idxes = np.argmax(bins, axis=bins.ndim - 1)
#     return vfov_bins_centers[idxes]


# def bins2distortion(bins):
#     idxes = np.argmax(bins, axis=bins.ndim - 1)
#     return distortion_bins_centers[idxes]


# pitch_bins_low = np.linspace(-np.pi/2 + 1e-5, -5*np.pi/180., 31)
# pitch_bins_low[0] = 0.
# pitch_bins_high = np.linspace(5*np.pi/180., np.pi/6, 31)
# pitch_bins_v0_wide = np.concatenate((np.linspace(-1.5, 0., 31), np.linspace(0., 1., 193), np.linspace(1., 1.5, 32)), 0)
# roll_bins = getBins(-np.pi/4, np.pi/4, 0.5, 0.04, 1.1, np.pi)
# roll_bins_centers = roll_bins.copy()
# roll_bins_centers[:-1] += np.diff(roll_bins_centers)/2
# roll_bins_centers = np.append(roll_bins_centers, roll_bins[-1])
# vfov_bins = np.linspace(0.33, 1.22, 255)
# vfov_bins_centers = vfov_bins.copy()
# vfov_bins_centers[:-1] += np.diff(vfov_bins_centers)/2
# vfov_bins_centers = np.append(vfov_bins_centers, vfov_bins[-1])
# distortion_bins = np.linspace(0, 1, 255)
# distortion_bins_centers = distortion_bins.copy()
# distortion_bins_centers[:-1] += np.diff(distortion_bins_centers)/2
# distortion_bins_centers = np.append(distortion_bins_centers, distortion_bins[-1])

def make_bins_layers_list(x_bins_lowHigh_list):
    x_bins_layers_list = []
    for layer_idx, x_bins_lowHigh in enumerate(x_bins_lowHigh_list):
        x_bins = np.linspace(x_bins_lowHigh[0], x_bins_lowHigh[1], 255)
        x_bins_centers = x_bins.copy()
        x_bins_centers[:-1] += np.diff(x_bins_centers)/2
        x_bins_centers = np.append(x_bins_centers, x_bins_centers[-1]) # 42 bins
        x_bins_layers_list.append(x_bins_centers)
    return x_bins_layers_list


# yc_bins = np.linspace(0.5, 3., 255)
# # print(vfov_bins, vfov_bins.shape)
# yc_bins_centers = yc_bins.copy()
# yc_bins_centers[:-1] += np.diff(yc_bins_centers)/2
# # print(vfov_bins_centers)
# yc_bins_centers = np.append(yc_bins_centers, yc_bins_centers[-1]) # 42 bins
# # print(vfov_bins_centers, vfov_bins_centers.shape)

# yc_bins_0 = np.linspace(0.5, 0.5, 255)
# # print(vfov_bins, vfov_bins.shape)
# yc_bins_centers_0 = yc_bins_0.copy()
# yc_bins_centers_0[:-1] += np.diff(yc_bins_centers_0)/2
# # print(vfov_bins_centers)
# yc_bins_centers_0 = np.append(yc_bins_centers_0, yc_bins_centers_0[-1]) # 42 bins

# yc_bins_1 = np.linspace(-0.45, 0.45, 255)
# # print(vfov_bins, vfov_bins.shape)
# yc_bins_centers_1 = yc_bins_1.copy()
# yc_bins_centers_1[:-1] += np.diff(yc_bins_centers_1)/2
# # print(vfov_bins_centers)
# yc_bins_centers_1 = np.append(yc_bins_centers_1, yc_bins_centers_1[-1]) # 42 bins

bins_lowHigh_list_dict = {}

# yc_bins_lowHigh_list = [[0.5, 3.], [-0.3, 0.3], [-0.15, 0.15], [-0.15, 0.15], [-0.05, 0.05]] # 'SmallerBins'
yc_bins_lowHigh_list = [[0.5, 5.], [-0.3, 0.3], [-0.15, 0.15], [-0.3, 0.3], [-0.15, 0.15]] # 'YcLargeBins'
# yc_bins_lowHigh_list = [[0.5, 10.], [-0.3, 0.3], [-0.15, 0.15], [-0.3, 0.3], [-0.15, 0.15]] # 'YcLargerBinsV2'

bins_lowHigh_list_dict['yc_bins_lowHigh_list'] = yc_bins_lowHigh_list
# yc_bins_lowHigh_list = [[0.5, 3.], [-0.45, 0.45], [-0.45, 0.45]] # 'OriginalBins'
# yc_bins_layers_list = []
# for layer_idx, yc_bins_lowHigh in enumerate(yc_bins_lowHigh_list):
#     yc_bins = np.linspace(yc_bins_lowHigh[0], yc_bins_lowHigh[1], 255)
#     yc_bins_centers = yc_bins.copy()
#     yc_bins_centers[:-1] += np.diff(yc_bins_centers)/2
#     yc_bins_centers = np.append(yc_bins_centers, yc_bins_centers[-1]) # 42 bins
#     yc_bins_layers_list.append(yc_bins_centers)
yc_bins_layers_list = make_bins_layers_list(yc_bins_lowHigh_list)
# yc_bins_centers = np.sum(np.stack(yc_bins_layers_list), 0)
yc_bins_centers = yc_bins_layers_list[0]


fmm_bins_lowHigh_list = [[0., 0.], [-0.2, 0.2], [-0.05, 0.05], [-0.05, 0.05], [-0.05, 0.05]] # percentage!!
bins_lowHigh_list_dict['fmm_bins_lowHigh_list'] = fmm_bins_lowHigh_list
# yc_bins_lowHigh_list = [[0.5, 3.], [-0.45, 0.45], [-0.45, 0.45]] # 'OriginalBins'
# fmm_bins_layers_list = []
# for layer_idx, fmm_bins_lowHigh in enumerate(fmm_bins_lowHigh_list):
#     fmm_bins = np.linspace(fmm_bins_lowHigh[0], fmm_bins_lowHigh[1], 255)
#     fmm_bins_centers = fmm_bins.copy()
#     fmm_bins_centers[:-1] += np.diff(fmm_bins_centers)/2
#     fmm_bins_centers = np.append(fmm_bins_centers, fmm_bins_centers[-1]) # 42 bins
#     fmm_bins_layers_list.append(fmm_bins_centers)
fmm_bins_layers_list = make_bins_layers_list(fmm_bins_lowHigh_list)


v0_bins_lowHigh_list = [[0., 0.], [-0.15, 0.15], [-0.05, 0.05], [-0.05, 0.05], [-0.05, 0.05]] # 'SmallerBins'
bins_lowHigh_list_dict['v0_bins_lowHigh_list'] = v0_bins_lowHigh_list
# v0_bins_layers_list = []
# for layer_idx, v0_bins_lowHigh in enumerate(v0_bins_lowHigh_list):
#     v0_bins = np.linspace(v0_bins_lowHigh[0], v0_bins_lowHigh[1], 255)
#     v0_bins_centers = v0_bins.copy()
#     v0_bins_centers[:-1] += np.diff(v0_bins_centers)/2
#     v0_bins_centers = np.append(v0_bins_centers, v0_bins_centers[-1]) # 42 bins
#     v0_bins_layers_list.append(v0_bins_centers)
v0_bins_layers_list = make_bins_layers_list(v0_bins_lowHigh_list)


# human_bins = np.linspace(1., 2., 256)
human_bins = np.linspace(1., 1.9, 256) #  'SmallerPersonBins'
# human_bins = np.linspace(1., 2.5, 256) #  'V2PersonCenBins'
# human_bins = np.linspace(0.7, 1.9, 256) #  'V3PersonCenBins'
human_bins_1 = np.linspace(-0.2, 0.2, 256)
human_bins_lowHigh_list = [[0., 0.], [-0.3, 0.15], [-0.10, 0.10], [-0.10, 0.10], [-0.05, 0.05]] # 'SmallerBins'
bins_lowHigh_list_dict['human_bins_lowHigh_list'] = human_bins_lowHigh_list
human_bins_layers_list = make_bins_layers_list(human_bins_lowHigh_list)


results_path_yannick = '/home/ruizhu/Documents/Projects/adobe_scale_est/data/yannick_results_train2017_filtered'
# image_path = '/home/ruizhu/Documents/Projects/adobe_scale_est/data/COCO/train2017'
# bbox_path = '/home/ruizhu/Documents/Projects/adobe_scale_est/data/coco_results/imgs_with_morethan2_standing_persons_allVis_train2017_2'
image_path = '/data/COCO/train2017'

# bbox_path = '/data/COCO/coco_results/imgs_with_morethan2_standing_persons_allVis_train2017_2'



# new dataset 2020
# bbox_path = '/data/COCO/coco_results/imgs_with_morethan2_standing_persons_train2017_20200101-2'
# bbox_path = '/data/COCO/coco_results/imgs_with_morethan2_standing_persons_train2017_20200103-v4'
bbox_path = '/data/COCO/coco_results/imgs_with_morethan2_standing_persons_train2017_20200103-v5_ratio2-8'

class COCO2017Scale:
    def __init__(self, transforms_maskrcnn=None, transforms_yannick=None, train=True, shuffle=True, logger=None, opt=None):
        self.opt = opt
        self.cfg = self.opt.cfg

        self.GOOD_NUM = self.cfg.DATA.COCO.GOOD_NUM

        self.transforms_maskrcnn = transforms_maskrcnn
        self.transforms_yannick = transforms_yannick
        ts = time.time()
        # try:
        #     with open("filelist_spherical.json", "r") as fhdl:
        #         self.data = json.load(fhdl)
        # except FileNotFoundError:
        self.yannick_mat_files = glob(os.path.join(results_path_yannick, "*.mat"), recursive=True)
        self.yannick_mat_files.sort()
        random.seed(123456)
        random.shuffle(self.yannick_mat_files)


        num_mat_files = len(self.yannick_mat_files)
        if train:
            self.yannick_mat_files = self.yannick_mat_files[:int(num_mat_files*0.8)]
        else:
            self.yannick_mat_files = self.yannick_mat_files[-int(num_mat_files*0.2):]
        logger.info(self.yannick_mat_files[0])
        # self.yannick_mat_files = self.yannick_mat_files[:100]
        # with open("filelist_spherical.json", "w") as fhdl:
        #     json.dump(self.data, fhdl)

        # self.yannick_mat_files_filtered = []
        # for yannick_mat_file in tqdm(self.yannick_mat_files):
        #     yannick_results = loadmat(yannick_mat_file)
        #     horizon_visible = yannick_results['horizon_visible'][0][0]
        #     if horizon_visible:
        #         self.yannick_mat_files_filtered.append(yannick_mat_file)
        #         os.system('cp %s %s'%(yannick_mat_file, yannick_mat_file.replace('yannick_results_train2017', 'yannick_results_train2017_filtered')))
        # self.yannick_mat_files = self.yannick_mat_files_filtered
        
        self.img_filenames = [os.path.basename(yannick_mat_file).split('.')[0] for yannick_mat_file in self.yannick_mat_files]
        self.img_files = [os.path.join(image_path, img_filename+'.jpg') for img_filename in self.img_filenames]
        self.bbox_npy_files = [os.path.join(bbox_path, img_filename+'_bboxes_valid.npy') for img_filename in self.img_filenames]
        
        # self.bbox_npy_files = glob(os.path.join(bbox_path, "*.npy"), recursive=True)
        # # print(len(self.bbox_npy_files))
        # for bbox_npy_file in tqdm(self.bbox_npy_files):
        #     # os.system('rm %s'%bbox_npy_file)
        #     bbox_file_id = os.path.basename(bbox_npy_file).split('_')[1]
        #     bbox_new_file = bbox_npy_file.replace(os.path.basename(bbox_npy_file), '%012d_bboxes_valid.npy'%(int(bbox_file_id)))
        #     os.system('cp %s %s'%(bbox_npy_file, bbox_new_file.replace('imgs_with_morethan2_standing_persons_allVis_train2017', 'imgs_with_morethan2_standing_persons_allVis_train2017_2')))
        #     os.system('cp %s %s'%(bbox_npy_file.replace('_bboxes_valid.npy', '.png'), bbox_new_file.replace('_bboxes_valid.npy', '.png').replace('imgs_with_morethan2_standing_persons_allVis_train2017', 'imgs_with_morethan2_standing_persons_allVis_train2017_2')))

        assert len(self.bbox_npy_files) == len(self.img_files) == len(self.yannick_mat_files)
        bbox_npy_files_filtered = []
        img_files_filtered = []
        yannick_mat_files_filtered = []
        for bbox_npy_file, img_file, yannick_mat_file in zip(self.bbox_npy_files, self.img_files, self.yannick_mat_files):
            if os.path.isfile(bbox_npy_file):
                assert os.path.basename(bbox_npy_file)[:12] == os.path.basename(img_file)[:12] == os.path.basename(yannick_mat_file)[:12]
                bbox_npy_files_filtered.append(bbox_npy_file)
                img_files_filtered.append(img_file)
                yannick_mat_files_filtered.append(yannick_mat_file)
        self.bbox_npy_files = bbox_npy_files_filtered
        self.img_files = img_files_filtered
        self.yannick_mat_files = yannick_mat_files_filtered
        assert len(self.bbox_npy_files) == len(self.img_files) == len(self.yannick_mat_files)

        logger.info(colored("[COCO dataset] Loaded %d files in %.4fs for %s set from %s."%(len(self.bbox_npy_files), time.time()-ts, 'train' if train else 'val', bbox_path), 'white', 'on_blue'))

        # from scipy.io import savemat
        # savemat('val_set.mat', {'img_files': self.img_files})

        if shuffle:
            random.seed(314159265)
            list_zip = list(zip(self.img_files, self.bbox_npy_files, self.yannick_mat_files))
            random.shuffle(list_zip)
            self.img_files, self.bbox_npy_files, self.yannick_mat_files = zip(*list_zip)
            assert os.path.basename(self.img_files[0])[:12] == os.path.basename(self.bbox_npy_files[0])[:12] == os.path.basename(self.yannick_mat_files[0])[:12]
            # print(self.img_files[:2])
            # print(self.bbox_npy_files[:2])
        # if train:
        #     self.data = self.data[:-2000]
        # else:
        #     self.data = self.data[-2000:]

        if not train:
            print([os.path.basename(img_file) for img_file in self.img_files[:12]])

    def __getitem__(self, k):
        im_ori_RGB = Image.open(self.img_files[k]).convert('RGB')
        bboxes = np.load(self.bbox_npy_files[k]).astype(np.float32) # [xywh]
        if bboxes.shape[0] > self.cfg.DATA.COCO.GOOD_NUM:
            bboxes = bboxes[:self.cfg.DATA.COCO.GOOD_NUM, :]
        yannick_results = loadmat(self.yannick_mat_files[k])
        horizon_visible = yannick_results['horizon_visible'][0][0].astype(np.float32)
        assert horizon_visible == 1
        horizon = yannick_results['pitch'][0][0].astype(np.float32)
        W, H = im_ori_RGB.size[:2]
        horizon_pixels_yannick = H * horizon
        v0 = H - horizon_pixels_yannick
        vfov = yannick_results['vfov'][0][0].astype(np.float32)
        f_pixels_yannick = H/2./(np.tan(vfov/2.))

        # print('-', im.size, np.asarray(im).shape)
        im_yannickTransform = self.transforms_yannick(im_ori_RGB) # [0., 1.] by default
        im_maskrcnnTransform = self.transforms_maskrcnn(im_ori_RGB) # [0., 1.] by default
        # print('---', im.size(), np.asarray(im).shape)
        # im_array = np.asarray(im)
        # if len(im_array.shape)==2:
        #     im_array = np.stack((im_array,)*3, axis=-1)
        #     # print(im_array.shape)
        # x = torch.from_numpy(im_array.transpose((2,0,1)))

        y_person = 1.75
        bbox_good_list = bboxes
        vc = H / 2.
        inv_f2_yannick = 1./ (f_pixels_yannick * f_pixels_yannick)
        yc_list = []
        for bbox in bbox_good_list:
            vt = H - bbox[1]
            vb = H - (bbox[1] + bbox[3])
        #     v0_single = yc * (vt - vb) / y_person + vb
            yc_single = y_person * (v0 - vb) / (vt - vb) / (1. + (vc - v0) * (vc - vt) / f_pixels_yannick**2)
            yc_list.append(yc_single)
        yc_estCam = np.median(np.asarray(yc_list))
        # idx3 = np.digitize(yc_estCam, yc_bins)
        # yc_onehot = np.zeros((256,), dtype=np.float32)
        # yc_onehot[idx3] = 1.

        # im_ori_BGR_array = np.array(im_ori_RGB.copy())[:,:,::-1]
        return im_yannickTransform, im_maskrcnnTransform, W, H, \
               yc_estCam.astype(np.float32), \
               self.pad_bbox(bboxes, self.GOOD_NUM).astype(np.float32), bboxes.shape[0], v0.astype(np.float32), f_pixels_yannick.astype(np.float32), \
               os.path.basename(self.img_files[k])[:12], self.img_files[k]

    def __len__(self):
        return len(self.img_files)

    def pad_bbox(self, bboxes, max_length):
        bboxes_padded = np.zeros((max_length, bboxes.shape[1]))
        assert bboxes.shape[0]<=max_length, 'bboxes length %d > max_length %d!'%(bboxes.shape[0], max_length)
        bboxes_padded[:bboxes.shape[0], :] = bboxes
        return bboxes_padded

def my_collate(batch):
    # Refer to https://discuss.pytorch.org/t/how-to-create-batches-of-a-list-of-varying-dimension-tensors/50773/14
    im_yannickTransform_list, im_maskrcnnTransform_list, W_batch_list, H_batch_list, yc_batch_list, \
            bboxes_batch_list, bboxes_length_batch_list, v0_batch_list, f_pixels_yannick_batch_list, im_filename_list, im_file_list = zip(*batch)
    # input_yannickTransform = torch.stack(im_yannickTransform_list)
    # input_maskrcnnTransform = torch.stack(im_maskrcnnTransform_list)
    W_batch_array = np.stack(W_batch_list).copy()
    H_batch_array = np.stack(H_batch_list).copy()
    # yc_onehot_batch = torch.stack(yc_onehot_batch_list)
    yc_batch = torch.tensor(yc_batch_list)
    bboxes_batch_array = np.stack(bboxes_batch_list).copy()
    bboxes_length_batch_array = np.stack(bboxes_length_batch_list).copy()
    v0_batch = torch.tensor(v0_batch_list)
    f_pixels_yannick_batch = torch.tensor(f_pixels_yannick_batch_list)
    # idx3_batch_list = [idx3.item() for idx3 in idx3_batch_list]
    # idx3_batch = torch.tensor(idx3_batch_list)
    return im_yannickTransform_list, im_maskrcnnTransform_list, W_batch_array, H_batch_array, yc_batch, \
            bboxes_batch_array, bboxes_length_batch_array, v0_batch, f_pixels_yannick_batch, im_filename_list, im_file_list

    # # batch contains a list of tuples of structure (sequence, target)
    # data = [item[0] for item in batch]
    # data = pack_sequence(data, enforce_sorted=False)
    # targets = [item[1] for item in batch]
    # return [data, targets]

def collate_fn_padd(batch):
    '''
    Padds batch of variable length

    note: it converts things ToTensor manually here since the ToTensor transform
    assume it takes in images rather than arbitrary tensors.
    '''
    ## get sequence lengths
    # lengths = torch.tensor([ t.shape[0] for t in batch ]).to(device)
    # ## padd
    # batch = [ torch.Tensor(t).to(device) for t in batch ]
    # batch = torch.nn.utils.rnn.pad_sequence(batch)
    # ## compute mask
    # mask = (batch != 0).to(device)
    # return batch, lengths, mask

    ims = [torch.Tensor(item[0]) for item in batch]
    bboxes = [torch.Tensor(item[1]) for item in batch]
    v0s = [torch.Tensor(np.asarray(item[2])) for item in batch]
    f_pixels_yannicks = [torch.Tensor(np.asarray(item[3])) for item in batch]
    img_filenames = [item[4] for item in batch]
    img_filepaths = [item[5] for item in batch]

    return [ims, bboxes, v0s, f_pixels_yannicks, img_filenames, img_filepaths]



    # def __getitem__(self, k):
    #     # with open(self.data[k][:-4] + ".json", "r") as fhdl:
    #     #     data = json.load(fhdl)
    #     #     data = data[2]

    #     #im = np.asarray(imread(self.data[k].replace("_true_camera_calibration.json", ".jpg"))[:,:,:3])
    #     im = Image.open(self.data[k])
    #     #im = Image.open(self.data[k].replace("_true_camera_calibration.json", ".jpg"))

    #     #hl_left, hl_right = getHorizonLineFromAngles(pitch=data["pitch"], roll=data["roll"], FoV=data["vfov"], im_h=im.size[0], im_w=im.size[1])
    #     #slope = np.arctan(hl_right - hl_left)
    #     #midpoint = (hl_left + hl_right) / 2
    #     #offset = (midpoint - 0.5) / np.sqrt( 1 + (hl_right - hl_left)**2 )
    #     #offset = getOffset(data["pitch"], data["roll"], data["vfov"], im.size[0], im.size[1])

    #     #idx1 = midpointpitch2bin(, data["pitch"])
    #     assert im.size[0] == data["width"]
    #     idx1 = midpointpitch2bin(data["offset"] / data["height"], data["pitch"])
    #     idx2 = np.digitize(data["roll"], roll_bins)
    #     idx3 = np.digitize(data["vfov"], vfov_bins)
    #     idx4 = np.digitize(data["spherical_distortion"], distortion_bins)
    #     #print("{:.04f}".format(data["vfov"]), "{:.04f}".format(data["pitch"]), idx1)

    #     y1 = np.zeros((256,), dtype=np.float32)
    #     y2 = np.zeros((256,), dtype=np.float32)
    #     y3 = np.zeros((256,), dtype=np.float32)
    #     y4 = np.zeros((256,), dtype=np.float32)

    #     if idx2 > 255 or idx1 > 255:
    #         print(self.data[k], data["offset"] / im.size[0], data["pitch"], idx1, idx2, idx3, idx4)
    #     y1[idx1] = y2[idx2] = y3[idx3] = y4[idx4] = 1.

    #     #x = torch.from_numpy(im.transpose((2,0,1)))
        # x = self.transforms(im)
    #     y1, y2, y3, y4 = map(torch.from_numpy, (y1, y2, y3, y4))

    #     return x, y1, y2, y3, y4, data
        
    #     #{"angle units": "radians", "yaw": 0.0, "has_artifact": false, "has_day_sky": false, "source": "pano_aoijeqajukkoem", "pitch": 0.00492545270356224, "primary_top_content": "buildings or ceilings", "vfov": 0.9096217797805077, "roll": -0.01719714340875391}


if __name__ == '__main__':

    # this_bin = midpointpitch2bin(1.1, 0.0)
    # a = np.zeros((256,)); a[this_bin] = 1
    # print("bin:", this_bin, "recovered:", bin2midpointpitch(a))

    # sys.exit()
    train = COCO2017Scale(train=True)
    print(len(train))
    for a in range(len(train)):
        _ = train[a]
        #print("---")
        #import pdb; pdb.set_trace()

