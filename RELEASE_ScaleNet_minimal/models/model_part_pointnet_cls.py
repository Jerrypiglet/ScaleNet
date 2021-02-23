import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, zeros_

# Import Shaper (install first: https://github.com/haosulab/shaper)
# If cannot find: export SHAPER_MODELS_PATH='/home/ruizhu/Documents/Projects/kitti_instance_RGBD_utils/deepSfm_ori/models/shaper/shaper/models/pointnet'

from pointnet_cls import Stem, SharedMLP, MLP, FC, xavier_uniform, set_bn

class CamHPointNet(nn.Module):
    """PointNet classification model

    Args:
        in_channels (int): the number of input channels
        out_channels (int): the number of output channels
        stem_channels (tuple of int): the numbers of channels in stem feature extractor
        local_channels (tuple of int): the numbers of channels in local mlp
        global_channels (tuple of int): the numbers of channels in global mlp
        dropout_prob (float): the probability to dropout
        with_transform (bool): whether to use TNet to transform features.

    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 stem_channels=(64, 64),
                 local_channels=(64, 128, 1024),
                 global_channels=(512, 256),
                 dropout_prob=0.3,
                 with_transform=True,
                 with_bn=True,
                 with_FC=True):
        super(CamHPointNet, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.with_FC = with_FC
        self.with_bn = with_bn
        self.with_transform = with_transform

        self.stem = Stem(in_channels, stem_channels, with_transform=with_transform, bn=with_bn)
        self.mlp_local = SharedMLP(stem_channels[-1], local_channels, bn=with_bn)
        self.mlp_global = MLP(local_channels[-1], global_channels, dropout_prob=dropout_prob, bn=with_bn)
        if self.with_FC:
            self.fc = FC(global_channels[-1], global_channels[-1], bn=with_bn)
        self.classifier = nn.Linear(global_channels[-1], out_channels, bias=True)
        # self.classifier = nn.Sequential(
        #     FC(global_channels[-1], global_channels[-1]),
        #     nn.Linear(global_channels[-1], out_channels, bias=True))

        self.init_weights()

    def forward(self, data_batch):
        x = data_batch['points']

        # stem
        x, end_points = self.stem(x)
        # mlp for local features
        x = self.mlp_local(x)
        # max pool over points
        x, max_indices = torch.max(x, 2)
        end_points['key_point_indices'] = max_indices
        # mlp for global features
        x = self.mlp_global(x)

        if self.with_FC:
            x = self.fc(x)

        x = self.classifier(x)
        preds = {
            'cls_logit': x
        }
        preds.update(end_points)

        return preds

    # def init_weights(self):
    #     # default initialization in original implementation
    #     self.mlp_local.init_weights(xavier_uniform)
    #     self.mlp_global.init_weights(xavier_uniform)
    #     xavier_uniform(self.classifier)
    #     # set batch normalization to 0.01 as default
    #     set_bn(self, momentum=0.01)

    def init_weights(self):
        # Default initialization in original implementation
        self.mlp_local.init_weights(xavier_uniform)
        self.mlp_global.init_weights(xavier_uniform)
        if self.with_FC:
            self.fc.init_weights(xavier_uniform)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
        # Set batch normalization to 0.01 as default
        if self.with_bn:
            set_bn(self, momentum=0.01)