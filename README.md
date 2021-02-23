# Single View Metrology in the Wild
Code and data for **Single View Metrology in the Wild, Zhu et al, ECCV 2020**

To be released. Stay tuned by watching (subscribing to) the repo from the button on the upper right corner.

# Installation
```bash
conda create -y -n scalenet python=3.6
conda activate scalenet
pip install -r requirements.txt
conda install nb_conda

cd maskrcnn-benchmark
conda install cudatoolkit
python setup.py build develop
cd ..

git clone https://github.com/NVIDIA/apex.git
cd apex
python setup.py install --cuda_ext --cpp_ext # if you see an error about commenting out an IF setence, do it
cd ..

python setup_maskrcnn_rui.py build develop

```
## with Jupyter notebook
Lanuch jupyter notebook. Kernel -> Change Kernel [to scalenet]

# Todolist
- [x] Inference demo for camera calibration on sample images;
- [x] Inference demo for scale estimation and visualization on COCOScale images;
- [ ] Training code for scale estimation and visualization on COCOScale images;
- [ ] Inference demo and data for KITTI and IMDB celebrity datasets.

# Notes
- Due to copyright issues with Adobe, SUN360 data for training the camera calibration network cannot be released. Instead a demo as well as checkpoint for inference has been released.
- The code release is still in progress, and the released codes will be cleaned up and properly commented or documented once the release is complete. As a result the current version of implementations of models, dataloaders etc. maybe cluttered.

# Camera Calibration Network

## Location
`./RELEASE_ScaleNet_minimal`

## Demo
`./RELEASE_ScaleNet_minimal/demo-evalCameraCalib-SUN360-RELEASE.ipynb`

## Description
This network is trained on SUN360 dataset with supervision of some camera parameters (e.g. roll, pitch, field of view (or equivalently focal length), which can be converted to horizon as well). The release model takes in a random image, and estimates:
- vfov (vertical field of view)
- pitch
- roll
- focal length

Note that geometric relationships exist between those items. Specifically:
- *f_pix = h / 2. / np.tan(vfov / 2.)*, where f_pix is the focal length in pixels, h is the image height in pixels
- *f_mm = f_pix / h * sensor_size*, which converts the f_pixel to focal length in 35mm equivalent frame (e.g. images taken by full-frame sensors)

# Scale Estimation Inference on COCOScale
## Preparation
- Download [checkpoint/20200222-162430_pod_backCompat_adam_wPerson05_720-540_REafterDeathV_afterFaster_bs16_fix3_nokpsLoss_personLoss3Layers_loss3layers](https://drive.google.com/drive/folders/111hCohH_X5TjOQKRx5P1w8Ow_7Od_P6Q?usp=sharing) to `checkpoint`. After that the folder should look like:
    - \- checkpoint/
        - \- 1109-0141-mm1_SUN360RCNN-HorizonPitchRollVfovNET_myDistNarrowerLarge1105_bs16on4_le1e-5_indeptClsHeads_synBNApex_valBS1_yannickTransformAug
        - \- 20200222-162430_pod_backCompat_adam_wPerson05_720-540_REafterDeathV_afterFaster_bs16_fix3_nokpsLoss_personLoss3Layers_loss3layers
- Download [all zip files for COCOScale](https://drive.google.com/drive/folders/1yew9ol6w_T83fLVMQ34AHCu6k5eLArWs?usp=sharing) and unzip to `data/results_coco`. After that the folder should look like:
    - \- data/results_coco/
        - \-  results_test_20200302_Car_noSmall-ratio1-35-mergeWith-results_with_kps_20200225_train2017_detOnly_filtered_2-8_moreThan2
        - \- results_with_kps_20200208_morethan2_2-8
        - \- results_with_kps_20200225_val2017_test_detOnly_filtered_2-8_moreThan2
- Download [COCO images train/val 2017](https://cocodataset.org/#download) to `/data/COCO` (or other path; can be configured in dataset_coco_pickle_eccv.py). After that the folder should look like:
    - \- /data/COCO/
        - \- train2017
        - \- val2017


## Location
`./RELEASE_ScaleNet_minimal`

## Demo
`./RELEASE_ScaleNet_minimal/demo-evalScaleNet-COCOScale-RELEASE.ipynb`

## Description
The demo loaded images from the COCOScla dataset and runs inference and visualization of the scale estimation task.