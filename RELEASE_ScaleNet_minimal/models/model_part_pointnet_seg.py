import torch
import torch.nn as nn

from pointnet_part_seg import *

class CamHPersonHPointNet(nn.Module):
    """PointNet for part segmentation

    References:
        https://github.com/charlesq34/pointnet/blob/master/part_seg/pointnet_part_seg.py

    """

    def __init__(self,
                 opt,
                 in_channels,
                 num_classes_camH,
                 num_classes_v0,
                 num_classes_fmm,
                 num_seg_classes,
                 stem_channels=(64, 128, 128),
                 local_channels=(512, 2048),
                 cls_channels=(256, 256),
                 seg_channels=(256, 256, 128),
                 dropout_prob_cls=0.3,
                 dropout_prob_seg=0.2,
                 with_transform=True,
                 with_bn=True,
                 if_cls=True):
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
        super(CamHPersonHPointNet, self).__init__()

        self.opt = opt

        self.in_channels = in_channels
        self.num_classes_camH = num_classes_camH
        self.num_classes_v0 = num_classes_v0
        self.num_classes_fmm = num_classes_fmm
        self.num_seg_classes = num_seg_classes
        self.with_bn = with_bn
        self.with_transform = with_transform
        self.if_cls = if_cls


        # stem
        self.stem = Stem(in_channels, stem_channels, with_transform=with_transform, bn=with_bn)
        self.mlp_local = SharedMLP(stem_channels[-1], local_channels, bn=with_bn)

        if self.if_cls:
            # classification
            # Notice that we apply dropout to each classification mlp.
            # -- pointnet_camH_refine
            self.mlp_cls = MLP(local_channels[-1], cls_channels, dropout_prob=dropout_prob_cls, bn=with_bn)
            self.cls_logit = nn.Linear(cls_channels[-1], num_classes_camH, bias=True)

            # -- pointnet_fmm_refine
            if self.opt.pointnet_fmm_refine:
                self.mlp_cls_fmm = MLP(local_channels[-1], cls_channels, dropout_prob=dropout_prob_cls, bn=with_bn)
                self.cls_logit_fmm = nn.Linear(cls_channels[-1], num_classes_fmm, bias=True)

            # -- pointnet_v0_refine
            if self.opt.pointnet_v0_refine:
                self.mlp_cls_v0 = MLP(local_channels[-1], cls_channels, dropout_prob=dropout_prob_cls, bn=with_bn)
                self.cls_logit_v0 = nn.Linear(cls_channels[-1], num_classes_v0, bias=True)

        # part segmentation
        # Notice that the original repo concatenates global feature, one hot class embedding,
        # stem features and local features. However, the paper does not use last local feature.
        # Here, we follow the released repo.
        in_channels_seg = local_channels[-1] + sum(stem_channels) + sum(local_channels)
        self.mlp_seg = SharedMLP(in_channels_seg, seg_channels[:-1], dropout_prob=dropout_prob_seg, bn=with_bn)
        self.conv_seg = Conv1d(seg_channels[-2], seg_channels[-1], 1)
        self.seg_logit = nn.Conv1d(seg_channels[-1], num_seg_classes, 1, bias=True)

        self.init_weights()

    def forward(self, data_batch):
        preds = {}

        x = data_batch["points"]
        # cls_label = data_batch["cls_label"]
        num_points = x.shape[2]
        end_points = {}

        # stem
        stem_feature, end_points_stem = self.stem(x)
        if self.with_transform:
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

        if self.if_cls:
            # classification
            x = global_feature
            x = self.mlp_cls(x)
            cls_logit = self.cls_logit(x)
            preds.update({
                    "cls_logit": cls_logit
                })

            if self.opt.pointnet_fmm_refine:
                x_fmm = self.mlp_cls_fmm(global_feature)
                cls_logit_fmm = self.cls_logit_fmm(x_fmm)
                preds.update({
                    "cls_logit_fmm": cls_logit_fmm
                })

            if self.opt.pointnet_v0_refine:
                x_v0 = self.mlp_cls_v0(global_feature)
                cls_logit_v0 = self.cls_logit_v0(x_v0)
                preds.update({
                    "cls_logit_v0": cls_logit_v0
                })

        # segmentation
        global_feature_expand = global_feature.unsqueeze(2).expand(-1, -1, num_points)
        # with torch.no_grad():
        #     I = torch.eye(self.num_classes, dtype=global_feature.dtype, device=global_feature.device)
        #     one_hot = I[cls_label]  # (batch_size, num_classes)
        #     one_hot_expand = one_hot.unsqueeze(2).expand(-1, -1, num_points)

        x = torch.cat(stem_features + local_features + [global_feature_expand], dim=1)
        x = self.mlp_seg(x)
        x = self.conv_seg(x)
        seg_logit = self.seg_logit(x)

        preds.update({
            "seg_logit": seg_logit
        })
        preds.update(end_points)

        return preds

    def init_weights(self):
        self.mlp_local.init_weights(xavier_uniform)
        if self.if_cls:
            self.mlp_cls.init_weights(xavier_uniform)
        self.mlp_seg.init_weights(xavier_uniform)
        self.conv_seg.init_weights(xavier_uniform)
        if self.if_cls:
            nn.init.xavier_uniform_(self.cls_logit.weight)
            nn.init.zeros_(self.cls_logit.bias)
        nn.init.xavier_uniform_(self.seg_logit.weight)
        nn.init.zeros_(self.seg_logit.bias)

        if self.if_cls:
            if self.opt.pointnet_fmm_refine:
                self.mlp_cls_fmm.init_weights(xavier_uniform)
                nn.init.xavier_uniform_(self.cls_logit_fmm.weight)
                nn.init.zeros_(self.cls_logit_fmm.bias)

            if self.opt.pointnet_v0_refine:
                self.mlp_cls_v0.init_weights(xavier_uniform)
                nn.init.xavier_uniform_(self.cls_logit_v0.weight)
                nn.init.zeros_(self.cls_logit_v0.bias)

        if self.with_bn:
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
