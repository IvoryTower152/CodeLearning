import torch
from torch import nn
from utils.config import opt
from utils import array_tool as at
import numpy as np
from model.utils.bbox_tools import bbox2loc, bbox_iou, loc2bbox
from torch.nn import functional as F
from collections import namedtuple

LossTuple = namedtuple('LossTuple',
                       ['rpn_loc_loss',
                        'rpn_cls_loss',
                        'roi_loc_loss',
                        'roi_cls_loss',
                        'total_loss'])


class FasterRCNNTrainer(nn.Module):
    # 初始化输入为网络模型
    def __init__(self, faster_rcnn):
        super(FasterRCNNTrainer, self).__init__()
        self.faster_rcnn = faster_rcnn
        self.rpn_sigma = opt.rpn_sigma
        self.roi_sigma = opt.roi_sigma

        self.anchor_target_creator = AnchorTargetCreator()
        self.proposal_target_creator = ProposalTargetCreator()

        self.loc_normalize_mean = faster_rcnn.loc_normalize_mean
        self.loc_normalize_std = faster_rcnn.loc_normalize_std

        self.optimizer = self.faster_rcnn.get_optimizer()

    def forward(self, imgs, bboxes, labels, scale):
        n = bboxes.shape[0]
        _, _, H, W = imgs.shape
        img_size = (H, W)

        # 共享特征提取器提取公共特征
        # [3, 600, 600]的图像得到[512, 37, 37]的特征图
        # 实际就是进行了16倍下采样, 与代码中feat_stride=16相对应
        features = self.faster_rcnn.extractor(imgs)

        # RPN组件 (RegionProposalNetwork)
        rpn_locs, rpn_scores, rois, roi_indices, anchor = \
            self.faster_rcnn.rpn(features, img_size, scale)

        bbox = bboxes[0]
        label = labels[0]
        rpn_score = rpn_scores[0]
        rpn_locs = rpn_locs[0]
        roi = rois

        # 对上一步生成的RoI进一步筛选
        # 最终产生128个RoI，为正样本和负样本的组合，并配有对应标签
        sample_roi, gt_roi_loc, gt_roi_label = self.proposal_target_creator(
            roi, at.tonumpy(bbox), at.tonumpy(label),
            self.loc_normalize_mean, self.loc_normalize_std)
        sample_roi_index = torch.zeros(len(sample_roi))

        # RoI head组件 (VGG16RoIHead)
        roi_cls_loc, roi_score = self.faster_rcnn.head(features, sample_roi, sample_roi_index)

        # 获得所有anchor的label和loc (H*W*9个，H与W为公共特征图size)
        gt_rpn_loc, gt_rpn_label = self.anchor_target_creator(
            at.tonumpy(bbox), anchor, img_size)
        gt_rpn_label = at.totensor(gt_rpn_label).long()
        gt_rpn_loc = at.totensor(gt_rpn_loc)

        # 计算RPN回归分支的损失
        # rpn_locs是RPN为每个anchor预测的偏移量
        # gt_rpn_loc是用anchor与bbox按照一定规则计算的真实偏移量，两者可以计算loss来优化RPN
        rpn_loc_loss = _fast_rcnn_loc_loss(
            rpn_locs,
            gt_rpn_loc,
            gt_rpn_label.data,
            self.rpn_sigma)

        # 计算RPN分类分支的损失
        rpn_cls_loss = F.cross_entropy(rpn_scores, gt_rpn_label.cuda(), ignore_index=-1)

        n_sample = roi_cls_loc.shape[0]
        roi_cls_loc = roi_cls_loc.view(n_sample, -1, 4)
        roi_loc = roi_cls_loc[torch.arange(0, n_sample).long().cuda(),
                              at.totensor(gt_roi_label).long()]
        gt_roi_label = at.totensor(gt_roi_label).long()
        gt_roi_loc = at.totensor(gt_roi_loc)

        # 计算Fast R-CNN回归分支损失
        roi_loc_loss = _fast_rcnn_loc_loss(
            roi_loc.contiguous(),
            gt_roi_loc,
            gt_roi_label.data,
            self.roi_sigma)

        # 计算Fast R-CNN分类分支损失
        roi_cls_loss = nn.CrossEntropyLoss()(roi_score, gt_roi_label.cuda())

        losses = [rpn_loc_loss, rpn_cls_loss, roi_loc_loss, roi_cls_loss]
        losses = losses + [sum(losses)]

        return LossTuple(*losses)

    def train_step(self, imgs, bboxes, labels, scale):
        self.optimizer.zero_grad()
        losses = self.forward(imgs, bboxes, labels, scale)
        losses.total_loss.backward()
        self.optimizer.step()
        return losses


class AnchorTargetCreator(object):
    def __init__(self, n_sample=256, pos_iou_thresh=0.7, neg_iou_thresh=0.3, pos_ration=0.5):
        self.n_sample = n_sample
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh = neg_iou_thresh
        self.pos_ratio = pos_ration

    def __call__(self, bbox, anchor, img_size):
        img_H, img_W = img_size

        n_anchor = len(anchor)
        inside_index = _get_inside_index(anchor, img_H, img_W)
        anchor = anchor[inside_index]
        argmax_ious, label = self._create_label(inside_index, anchor, bbox)
        loc = bbox2loc(anchor, bbox[argmax_ious])
        # 其他所有anchor的label以及loc补全，只是它们均是无用的(分别用-1和0填充)
        label = _unmap(label, n_anchor, inside_index, fill=-1)
        loc = _unmap(loc, n_anchor, inside_index, fill=0)
        return loc, label

    def _create_label(self, inside_index, anchor, bbox):
        label = np.empty((len(inside_index),), dtype=np.int32)
        label.fill(-1)
        argmax_ious, max_ious, gt_argmax_ious = self._calc_ious(anchor, bbox, inside_index)
        # IoU小于设定的阈值的样本就是负样本
        label[max_ious <  self.neg_iou_thresh] = 0
        # IoU最高的一定是正样本
        label[gt_argmax_ious] = 1
        # IoU大于设定的阈值的样本就是正样本
        label[max_ious >= self.pos_iou_thresh] = 1

        # 和RPN的时候一个情况筛选样本，这次n_sample为256
        n_pos = int(self.pos_ratio * self.n_sample)
        pos_index = np.where(label == 1)[0]
        if len(pos_index) > n_pos:
            disable_index = np.random.choice(
                pos_index, size=(len(pos_index) - n_pos), replace=False)
            label[disable_index] = -1

        n_neg = self.n_sample - np.sum(label == 1)
        neg_index = np.where(label == 0)[0]
        if len(neg_index) > n_neg:
            disable_index = np.random.choice(
                neg_index, size=(len(neg_index) - n_neg), replace=False)
            label[disable_index] = -1

        return argmax_ious, label

    def _calc_ious(self, anchor, bbox, inside_index):
        ious = bbox_iou(anchor, bbox)
        argmax_ious = ious.argmax(axis=1)
        max_ious = ious[np.arange(len(inside_index)), argmax_ious]
        gt_argmax_ious = ious.argmax(axis=0)
        gt_max_ious = ious[gt_argmax_ious, np.arange(ious.shape[1])]
        gt_argmax_ious = np.where(ious == gt_max_ious)[0]

        return argmax_ious, max_ious, gt_argmax_ious


def _get_inside_index(anchor, H, W):
    index_inside = np.where(
        (anchor[:, 0] >= 0) &
        (anchor[:, 1] >= 0) &
        (anchor[:, 2] <= H) &
        (anchor[:, 3] <= W)
    )[0]
    return index_inside


def _unmap(data, count, index, fill=0):
    if len(data.shape) == 1:
        ret = np.empty((count,), dtype=data.dtype)
        ret.fill(fill)
        ret[index] = data
    else:
        ret = np.empty((count,) + data.shape[1:], dtype=data.dtype)
        ret.fill(fill)
        ret[index, :] = data
    return ret


class ProposalTargetCreator(object):
    def __init__(self, n_sample=128, pos_ratio=0.25, pos_iou_thresh=0.5,
                 neg_iou_thresh_hi=0.5, neg_iou_thresh_lo=0.0):
        self.n_sample = n_sample
        self.pos_ratio = pos_ratio
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh_hi = neg_iou_thresh_hi
        self.neg_iou_thresh_lo = neg_iou_thresh_lo

    def __call__(self, roi, bbox, label,
                 loc_normalize_mean=(0.0, 0.0, 0.0, 0.0),
                 loc_normalize_std=(0.1, 0.1, 0.2, 0.2)):
        # 模拟的数据中只有1个边界框
        n_bbox, _ = bbox.shape
        # 真实框和之前产生的RoI框整合
        roi = np.concatenate((roi, bbox), axis=0)
        # 决定正样本数量
        pos_roi_per_image = np.round(self.n_sample * self.pos_ratio)
        # 计算每个roi框体与真实框bbox之间对应的IoU
        iou = bbox_iou(roi, bbox)
        # 为目前所有roi中(包括bbox)的框体分配一个标签，标签为0表示它为背景
        gt_assignment = iou.argmax(axis=1)
        max_iou = iou.max(axis=1)
        gt_roi_label = label[gt_assignment] + 1
        # 根据正样本iou阈值来筛选roi中的正样本，此前有设置过正样本数量，因此还需要对正样本筛选
        # 与论文描述的策略相同
        pos_index = np.where(max_iou >= self.pos_iou_thresh)[0]
        pos_roi_per_this_image = int(min(pos_roi_per_image, pos_index.size))
        if pos_index.size > 0:
            # 筛选的方式为随机抽，到数量就结束
            pos_index = np.random.choice(pos_index, size=pos_roi_per_this_image, replace=False)
        # 将一定iou阈值内的样本作为负样本
        neg_index = np.where(
            (max_iou < self.neg_iou_thresh_hi) &
            (max_iou >= self.neg_iou_thresh_lo)
        )[0]
        # n_sample是128，也就是框体总数最大不超过128，分别有正样本和负样本
        neg_roi_per_this_image = self.n_sample - pos_roi_per_this_image
        neg_roi_per_this_image = int(min(neg_roi_per_this_image, neg_index.size))
        if neg_index.size > 0:
            neg_index = np.random.choice(neg_index, size=neg_roi_per_this_image, replace=False)

        # 最后要保留的 roi box
        keep_index = np.append(pos_index, neg_index)
        # 标签部分
        gt_roi_label = gt_roi_label[keep_index]
        # 负样本要将标签置为0，表示背景 (目标检测多增的一类)
        gt_roi_label[pos_roi_per_this_image:] = 0
        # box部分
        sample_roi = roi[keep_index]

        # 换算一下这些roi和真实bbox之间的偏移
        gt_roi_loc = bbox2loc(sample_roi, bbox[gt_assignment[keep_index]])
        gt_roi_loc = ((gt_roi_loc - np.array(loc_normalize_mean, np.float32))
                      / np.array(loc_normalize_std, np.float32))

        # 最终返回这三个变量
        return sample_roi, gt_roi_loc, gt_roi_label


def _smooth_l1_loss(x, t, in_weight, sigma):
    sigma2 = sigma ** 2
    diff = in_weight * (x - t)
    abs_diff = diff.abs()
    flag = (abs_diff.data < (1.0 / sigma2)).float()
    y = (flag * (sigma2 / 2.0) * (diff ** 2) + (1 - flag) * (abs_diff - 0.5 / sigma2))
    return y.sum()


def _fast_rcnn_loc_loss(pred_loc, gt_loc, gt_label, sigma):
    in_weight = torch.zeros(gt_loc.shape).cuda()
    in_weight[(gt_label > 0).view(-1, 1).expend_as(in_weight).cuda()] = 1
    loc_loss = _smooth_l1_loss(pred_loc, gt_loc, in_weight.detach(), sigma)
    loc_loss /= ((gt_label >= 0).sum().float())
    return loc_loss


if __name__ == '__main__':
    print('Clone Dolly Tests.')