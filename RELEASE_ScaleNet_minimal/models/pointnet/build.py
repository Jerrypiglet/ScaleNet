from .pointnet_cls import PointNetCls, PointNetClsLoss
from .pointnet_part_seg import PointNetPartSeg, PointNetPartSegLoss
from ..metric import ClsAccuracy, SegAccuracy, PartSegMetric, MetricList


def build_pointnet(cfg):
    if cfg.TASK == "classification":
        net = PointNetCls(
            in_channels=cfg.INPUT.IN_CHANNELS,
            out_channels=cfg.DATASET.NUM_CLASSES,
            stem_channels=cfg.MODEL.POINTNET.STEM_CHANNELS,
            local_channels=cfg.MODEL.POINTNET.LOCAL_CHANNELS,
            global_channels=cfg.MODEL.POINTNET.GLOBAL_CHANNELS,
            dropout_prob=cfg.MODEL.POINTNET.DROPOUT_PROB,
            with_transform=cfg.MODEL.POINTNET.WITH_TRANSFORM,
        )
        loss_fn = PointNetClsLoss(cfg.MODEL.POINTNET.REG_WEIGHT)
        metric_fn = ClsAccuracy()
    elif cfg.TASK == "part_segmentation":
        net = PointNetPartSeg(
            in_channels=cfg.INPUT.IN_CHANNELS,
            num_classes=cfg.DATASET.NUM_CLASSES,
            num_seg_classes=cfg.DATASET.NUM_SEG_CLASSES,
            stem_channels=cfg.MODEL.POINTNET.STEM_CHANNELS,
            local_channels=cfg.MODEL.POINTNET.LOCAL_CHANNELS,
            cls_channels=cfg.MODEL.POINTNET.CLS_CHANNELS,
            seg_channels=cfg.MODEL.POINTNET.SEG_CHANNELS,
            dropout_prob_cls=cfg.MODEL.POINTNET.DROPOUT_PROB_CLS,
            dropout_prob_seg=cfg.MODEL.POINTNET.DROPOUT_PROB_SEG,
            with_transform=cfg.MODEL.POINTNET.WITH_TRANSFORM,
        )
        loss_fn = PointNetPartSegLoss(cfg.MODEL.POINTNET.REG_WEIGHT,
                                      cfg.MODEL.POINTNET.CLS_LOSS_WEIGHT,
                                      cfg.MODEL.POINTNET.SEG_LOSS_WEIGHT)
        metric_fn = PartSegMetric(cfg.DATASET.NUM_SEG_CLASSES)
        if cfg.MODEL.POINTNET.CLS_LOSS_WEIGHT > 0.0:
            metric_fn = MetricList([metric_fn, ClsAccuracy()])
    else:
        raise NotImplementedError()

    return net, loss_fn, metric_fn
