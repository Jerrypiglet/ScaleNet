"""PointNet for part segmentation

References:
    @article{qi2016pointnet,
      title={PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation},
      author={Qi, Charles R and Su, Hao and Mo, Kaichun and Guibas, Leonidas J},
      journal={arXiv preprint arXiv:1612.00593},
      year={2016}
    }
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from nn import MLP, SharedMLP, Conv1d
from nn.init import xavier_uniform, set_bn
from models.pointnet.pointnet_cls import TNet


class Stem(nn.Module):
    """Stem (main body or stalk). Extract features from raw point clouds

    Structure: input (-> [TNet] -> transform_input) -> [MLP] -> features (-> [TNet] -> transform_feature)

    Attributes:
        with_transform: whether to use TNet

    """

    def __init__(self, in_channels,
                 stem_channels=(64, 128, 128),
                 with_transform=True,
                 bn=True):
        super(Stem, self).__init__()

        self.in_channels = in_channels
        self.out_channels = stem_channels[-1]
        self.with_transform = with_transform

        # feature stem
        self.mlp = SharedMLP(in_channels, stem_channels, bn=bn)
        self.mlp.init_weights(xavier_uniform)

        if self.with_transform:
            # input transform
            self.transform_input = TNet(in_channels, in_channels)
            # feature transform
            self.transform_feature = TNet(self.out_channels, self.out_channels)

    def forward(self, x):
        """PointNet Stem forward

        Args:
            x (torch.Tensor): (batch_size, in_channels, num_points)

        Returns:
            torch.Tensor: (batch_size, stem_channels[-1], num_points)
            dict (optional non-empty):
                trans_input: (batch_size, in_channels, in_channels)
                trans_feature: (batch_size, stem_channels[-1], stem_channels[-1])
                stem_features (list of torch.Tensor)

        """
        end_points = {}

        # input transform
        if self.with_transform:
            trans_input = self.transform_input(x)
            x = torch.bmm(trans_input, x)
            end_points["trans_input"] = trans_input

        # feature
        features = []
        for module in self.mlp:
            x = module(x)
            features.append(x)
        end_points["stem_features"] = features

        # feature transform
        if self.with_transform:
            trans_feature = self.transform_feature(x)
            x = torch.bmm(trans_feature, x)
            end_points["trans_feature"] = trans_feature

        return x, end_points


class PointNetPartSeg(nn.Module):
    """PointNet for part segmentation

    References:
        https://github.com/charlesq34/pointnet/blob/master/part_seg/pointnet_part_seg.py

    """

    def __init__(self,
                 in_channels,
                 num_classes,
                 num_seg_classes,
                 stem_channels=(64, 128, 128),
                 local_channels=(512, 2048),
                 cls_channels=(256, 256),
                 seg_channels=(256, 256, 128),
                 dropout_prob_cls=0.3,
                 dropout_prob_seg=0.2,
                 with_transform=True):
        """

        Args:
           in_channels (int): the number of input channels
           out_channels (int): the number of output channels
           stem_channels (tuple of int): the numbers of channels in stem feature extractor
           local_channels (tuple of int): the numbers of channels in local mlp
           cls_channels (tuple of int): the numbers of channels in classification mlp
           seg_channels (tuple of int): the numbers of channels in segmentation mlp
           dropout_prob_cls (float): the probability to dropout in classification mlp
           dropout_prob_seg (float): the probability to dropout in segmentation mlp
           with_transform (bool): whether to use TNet to transform features.

        """
        super(PointNetPartSeg, self).__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num_seg_classes = num_seg_classes

        # stem
        self.stem = Stem(in_channels, stem_channels, with_transform=with_transform)
        self.mlp_local = SharedMLP(stem_channels[-1], local_channels)

        # classification
        # Notice that we apply dropout to each classification mlp.
        self.mlp_cls = MLP(local_channels[-1], cls_channels, dropout=dropout_prob_cls)
        self.cls_logit = nn.Linear(cls_channels[-1], num_classes, bias=True)

        # part segmentation
        # Notice that the original repo concatenates global feature, one hot class embedding,
        # stem features and local features. However, the paper does not use last local feature.
        # Here, we follow the released repo.
        in_channels_seg = local_channels[-1] + num_classes + sum(stem_channels) + sum(local_channels)
        self.mlp_seg = SharedMLP(in_channels_seg, seg_channels[:-1], dropout=dropout_prob_seg)
        self.conv_seg = Conv1d(seg_channels[-2], seg_channels[-1], 1)
        self.seg_logit = nn.Conv1d(seg_channels[-1], num_seg_classes, 1, bias=True)

        self.init_weights()

    def forward(self, data_batch):
        x = data_batch["points"]
        cls_label = data_batch["cls_label"]
        num_points = x.shape[2]
        end_points = {}

        # stem
        stem_feature, end_points_stem = self.stem(x)
        end_points["trans_input"] = end_points_stem["trans_input"]
        end_points["trans_feature"] = end_points_stem["trans_feature"]
        stem_features = end_points_stem["stem_features"]

        # mlp for local features
        local_features = []
        x = stem_feature
        for ind, mlp in enumerate(self.mlp_local):
            x = mlp(x)
            local_features.append(x)

        # max pool over points
        global_feature, max_indices = torch.max(x, 2)  # (batch_size, local_channels[-1])
        end_points['key_point_inds'] = max_indices

        # classification
        x = global_feature
        x = self.mlp_cls(x)
        cls_logit = self.cls_logit(x)

        # segmentation
        global_feature_expand = global_feature.unsqueeze(2).expand(-1, -1, num_points)
        with torch.no_grad():
            I = torch.eye(self.num_classes, dtype=global_feature.dtype, device=global_feature.device)
            one_hot = I[cls_label]  # (batch_size, num_classes)
            one_hot_expand = one_hot.unsqueeze(2).expand(-1, -1, num_points)

        x = torch.cat(stem_features + local_features + [global_feature_expand, one_hot_expand], dim=1)
        x = self.mlp_seg(x)
        x = self.conv_seg(x)
        seg_logit = self.seg_logit(x)

        preds = {
            "cls_logit": cls_logit,
            "seg_logit": seg_logit
        }
        preds.update(end_points)

        return preds

    def init_weights(self):
        self.mlp_local.init_weights(xavier_uniform)
        self.mlp_cls.init_weights(xavier_uniform)
        self.mlp_seg.init_weights(xavier_uniform)
        self.conv_seg.init_weights(xavier_uniform)
        nn.init.xavier_uniform_(self.cls_logit.weight)
        nn.init.zeros_(self.cls_logit.bias)
        nn.init.xavier_uniform_(self.seg_logit.weight)
        nn.init.zeros_(self.seg_logit.bias)
        # Set batch normalization to 0.01 as default
        set_bn(self, momentum=0.01)


class PointNetPartSegLoss(nn.Module):
    """Pointnet part segmentation loss with optional regularization loss"""

    def __init__(self, reg_weight, cls_loss_weight, seg_loss_weight):
        super(PointNetPartSegLoss, self).__init__()
        self.reg_weight = reg_weight
        self.cls_loss_weight = cls_loss_weight
        self.seg_loss_weight = seg_loss_weight
        assert self.seg_loss_weight >= 0.0

    def forward(self, preds, labels):
        seg_logit = preds["seg_logit"]
        seg_label = labels["seg_label"]
        seg_loss = F.cross_entropy(seg_logit, seg_label)
        loss_dict = {
            "seg_loss": seg_loss * self.seg_loss_weight,
        }

        if self.cls_loss_weight > 0.0:
            cls_logit = preds["cls_logit"]
            cls_label = labels["cls_label"]
            cls_loss = F.cross_entropy(cls_logit, cls_label)
            loss_dict["cls_loss"] = cls_loss

        # regularization over transform matrix
        if self.reg_weight > 0.0:
            trans_feature = preds["trans_feature"]
            trans_norm = torch.bmm(trans_feature.transpose(2, 1), trans_feature)  # [in, in]
            I = torch.eye(trans_norm.size(2), dtype=trans_norm.dtype, device=trans_norm.device)
            reg_loss = F.mse_loss(trans_norm, I.unsqueeze(0).expand_as(trans_norm), reduction="sum")
            loss_dict["reg_loss"] = reg_loss * (0.5 * self.reg_weight / trans_norm.size(0))

        return loss_dict


if __name__ == '__main__':
    batch_size = 32
    in_channels = 3
    num_points = 1024
    num_classes = 16
    num_seg_classes = 50

    points = torch.rand(batch_size, in_channels, num_points)
    cls_label = torch.randint(num_classes, (batch_size,))
    transform = TNet()
    out = transform(points)
    print('TNet', out.shape)

    pointnet = PointNetPartSeg(in_channels, num_classes, num_seg_classes)
    out_dict = pointnet({"points": points, "cls_label": cls_label})
    for k, v in out_dict.items():
        print('PointNet:', k, v.shape)
