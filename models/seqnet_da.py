import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator, RegionProposalNetwork, RPNHead
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.ops import MultiScaleRoIAlign

from models.resnet import build_resnet
from models.roi_head_da import SeqRoIHeadsDa
from models.da_head import DomainAdaptationModule
from models.box_head import BBoxRegressor
from apex import amp


class SeqNetDa(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.target_start_epoch = cfg.TARGET_REID_START

        backbone, box_head, reid_head = build_resnet(name="resnet50", pretrained=True)

        anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),))
        head = RPNHead(
            in_channels=backbone.out_channels,
            num_anchors=anchor_generator.num_anchors_per_location()[0],
        )
        pre_nms_top_n = dict(training=cfg.MODEL.RPN.PRE_NMS_TOPN_TRAIN, testing=cfg.MODEL.RPN.PRE_NMS_TOPN_TEST)
        post_nms_top_n = dict(training=cfg.MODEL.RPN.POST_NMS_TOPN_TRAIN, testing=cfg.MODEL.RPN.POST_NMS_TOPN_TEST)
        rpn = RegionProposalNetwork(
            anchor_generator=anchor_generator,
            head=head,
            fg_iou_thresh=cfg.MODEL.RPN.POS_THRESH_TRAIN,
            bg_iou_thresh=cfg.MODEL.RPN.NEG_THRESH_TRAIN,
            batch_size_per_image=cfg.MODEL.RPN.BATCH_SIZE_TRAIN,
            positive_fraction=cfg.MODEL.RPN.POS_FRAC_TRAIN,
            pre_nms_top_n=pre_nms_top_n,
            post_nms_top_n=post_nms_top_n,
            nms_thresh=cfg.MODEL.RPN.NMS_THRESH,
        )

        faster_rcnn_predictor = FastRCNNPredictor(2048, 2)
        # reid_head = deepcopy(box_head)
        box_roi_pool = MultiScaleRoIAlign(featmap_names=["feat_res4"], output_size=14, sampling_ratio=2)
        box_predictor = BBoxRegressor(2048, num_classes=2, bn_neck=cfg.MODEL.ROI_HEAD.BN_NECK)
        roi_heads = SeqRoIHeadsDa(
            # SeqNet
            faster_rcnn_predictor=faster_rcnn_predictor,
            reid_head=reid_head,
            # parent class
            box_roi_pool=box_roi_pool,
            box_head=box_head,
            box_predictor=box_predictor,
            fg_iou_thresh=cfg.MODEL.ROI_HEAD.POS_THRESH_TRAIN,
            bg_iou_thresh=cfg.MODEL.ROI_HEAD.NEG_THRESH_TRAIN,
            batch_size_per_image=cfg.MODEL.ROI_HEAD.BATCH_SIZE_TRAIN,
            positive_fraction=cfg.MODEL.ROI_HEAD.POS_FRAC_TRAIN,
            bbox_reg_weights=None,
            score_thresh=cfg.MODEL.ROI_HEAD.SCORE_THRESH_TEST,
            nms_thresh=cfg.MODEL.ROI_HEAD.NMS_THRESH_TEST,
            detections_per_img=cfg.MODEL.ROI_HEAD.DETECTIONS_PER_IMAGE_TEST,
        )

        transform = GeneralizedRCNNTransform(
            min_size=cfg.INPUT.MIN_SIZE,
            max_size=cfg.INPUT.MAX_SIZE,
            image_mean=[0.485, 0.456, 0.406],
            image_std=[0.229, 0.224, 0.225],
        )

        self.backbone = backbone
        self.rpn = rpn
        self.roi_heads = roi_heads
        # here modified, for adapting to amp
        self.roi_heads.box_roi_pool.forward = amp.half_function(self.roi_heads.box_roi_pool.forward)
        self.transform = transform
        self.da_heads = DomainAdaptationModule(cfg.MODEL.DA_HEADS)

        # loss weights
        self.lw_rpn_reg = cfg.SOLVER.LW_RPN_REG
        self.lw_rpn_cls = cfg.SOLVER.LW_RPN_CLS
        self.lw_proposal_reg = cfg.SOLVER.LW_PROPOSAL_REG
        self.lw_proposal_cls = cfg.SOLVER.LW_PROPOSAL_CLS
        self.lw_box_reg = cfg.SOLVER.LW_BOX_REG
        self.lw_box_cls = cfg.SOLVER.LW_BOX_CLS
        self.lw_box_reid = cfg.SOLVER.LW_BOX_REID
        self.lw_box_reid_t = cfg.SOLVER.LW_BOX_REID_T

    # The is_source here should be switched when inferencing
    def inference(self, images, targets=None, query_img_as_gallery=False, is_source=False):
        original_image_sizes = [img.shape[-2:] for img in images]
        images, targets = self.transform(images, targets)
        features = self.backbone(images.tensors)

        if query_img_as_gallery:
            assert targets is not None

        if targets is not None and not query_img_as_gallery:
            # query
            boxes = [t["boxes"] for t in targets]

            box_features = self.roi_heads.box_roi_pool(features, boxes, images.image_sizes)
            box_features = self.roi_heads.reid_head(box_features, is_source)
            embeddings, _ = self.roi_heads.embedding_head(box_features)
            return embeddings.split(1, 0)
        else:
            # gallery
            proposals, _ = self.rpn(images, features, targets)
            detections, _ = self.roi_heads(
                features, proposals, images.image_sizes, targets, query_img_as_gallery, is_source
            )
            detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)
            return detections

    def forward(
        self,
        images_s,
        targets_s=None,
        images_t=None,
        targets_t=None,
        query_img_as_gallery=False,
        is_source=False,
        epoch=0,
    ):
        if not self.training:
            return self.inference(images_s, targets_s, query_img_as_gallery, is_source)

        images_s, targets_s = self.transform(images_s, targets_s)
        images_t, targets_t = self.transform(images_t, targets_t)

        losses = {}
        features_s = self.backbone(images_s.tensors)
        proposals_s, proposal_losses_s = self.rpn(images_s, features_s, targets_s)
        _, detector_losses_s = self.roi_heads(features_s, proposals_s, images_s.image_sizes, targets_s)

        da_ins_feas_s, da_ins_labels_s, da_ins_feas_s_before, da_ins_labels_s_before = self.roi_heads.extract_da(
            features_s, proposals_s, images_s.image_sizes, targets_s
        )
        da_ins_labels_s = torch.cat(da_ins_labels_s)
        da_ins_labels_s_before = torch.cat(da_ins_labels_s_before)

        # rename rpn losses to be consistent with detection losses
        proposal_losses_s["loss_rpn_reg"] = proposal_losses_s.pop("loss_rpn_box_reg")
        proposal_losses_s["loss_rpn_cls"] = proposal_losses_s.pop("loss_objectness")

        features_t = self.backbone(images_t.tensors)
        proposals_t, proposal_losses_t = self.rpn(images_t, features_t, targets_t)

        if epoch >= self.target_start_epoch:
            _, reid_losses_t = self.roi_heads(
                features_t, proposals_t, images_t.image_sizes, targets_t, query_img_as_gallery=False, is_source=False
            )

            # rename target domain losses
            proposal_losses_t["loss_rpn_reg_t"] = proposal_losses_t.pop("loss_rpn_box_reg")
            proposal_losses_t["loss_rpn_cls_t"] = proposal_losses_t.pop("loss_objectness")
            reid_losses_t["loss_box_reg_t"] = reid_losses_t.pop("loss_box_reg")
            reid_losses_t["loss_box_cls_t"] = reid_losses_t.pop("loss_box_cls")
            reid_losses_t["loss_proposal_reg_t"] = reid_losses_t.pop("loss_proposal_reg")
            reid_losses_t["loss_proposal_cls_t"] = reid_losses_t.pop("loss_proposal_cls")
            losses.update(reid_losses_t)
            losses.update(proposal_losses_t)
            losses["loss_rpn_reg_t"] *= 0.1 * self.lw_rpn_reg
            losses["loss_rpn_cls_t"] *= 0.1 * self.lw_rpn_cls
            losses["loss_proposal_reg_t"] *= 0.1 * self.lw_proposal_reg
            losses["loss_proposal_cls_t"] *= 0.1 * self.lw_proposal_cls
            losses["loss_box_reg_t"] *= 0.1 * self.lw_box_reg
            losses["loss_box_cls_t"] *= 0.1 * self.lw_box_cls
            losses["loss_box_reid_t"] *= self.lw_box_reid_t

        da_ins_feas_t, da_ins_labels_t, da_ins_feas_t_before, da_ins_labels_t_before = self.roi_heads.extract_da(
            features_t, proposals_t, images_t.image_sizes, targets_t
        )
        da_ins_labels_t = torch.cat(da_ins_labels_t)
        da_ins_labels_t_before = torch.cat(da_ins_labels_t_before)
        if self.da_heads:
            da_losses_s = self.da_heads(
                [features_s["feat_res4"]],
                da_ins_feas_s,
                da_ins_labels_s,
                da_ins_feas_s_before,
                da_ins_labels_s_before,
                targets_s,
            )
            da_losses_t = self.da_heads(
                [features_t["feat_res4"]],
                da_ins_feas_t,
                da_ins_labels_t,
                da_ins_feas_t_before,
                da_ins_labels_t_before,
                targets_t,
            )

        losses.update(detector_losses_s)
        losses.update(proposal_losses_s)
        da_losses_t["loss_da_image_t"] = da_losses_t.pop("loss_da_image")
        da_losses_t["loss_da_instance_t"] = da_losses_t.pop("loss_da_instance")
        da_losses_t["loss_da_consistency_t"] = da_losses_t.pop("loss_da_consistency")
        losses.update(da_losses_s)
        losses.update(da_losses_t)

        # apply loss weights
        losses["loss_rpn_reg"] *= self.lw_rpn_reg
        losses["loss_rpn_cls"] *= self.lw_rpn_cls
        losses["loss_proposal_reg"] *= self.lw_proposal_reg
        losses["loss_proposal_cls"] *= self.lw_proposal_cls
        losses["loss_box_reg"] *= self.lw_box_reg
        losses["loss_box_cls"] *= self.lw_box_cls
        losses["loss_box_reid_s"] *= self.lw_box_reid

        return losses
