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

# Camera Calibration Network

## Location
`./RELEASE_SUN360_camPred_minimal`

## Demo
`./RELEASE_SUN360_camPred_minimal/1109-evalSUN350_RCNNOnly-RELEASE.ipynb`

## Description
This network is trained on SUN360 dataset with supervision of some camera parameters (e.g. roll, pitch, field of view (or equivalently focal length), which can be converted to horizon as well). The release model takes in a random image, and estimates:
- vfov (vertical field of view)
- pitch
- roll
- focal length

Note that geometric relationships exist between those items. Specifically:
- *f_pix = h / 2. / np.tan(vfov / 2.)*, where f_pix is the focal length in pixels, h is the image height in pixels
- *f_mm = f_pix / h * sensor_size*, which converts the f_pixel to focal length in 35mm equivalent frame (e.g. images taken by full-frame sensors)

#

