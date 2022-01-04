import numpy as np
from game1 import FasterRCNN
from torchvision.models import vgg16
from torch import nn
from utils.config import opt
from torchvision.ops import nms
from model.utils.bbox_tools import bbox2loc, bbox_iou, loc2bbox
from torchvision.ops import RoIPool
from torch.nn import functional as F
import torch
from utils import array_tool as at


''' For VGG16 Backbone'''
def decom_vgg16():
    model = vgg16(False)
    features = list(model.features)[:30]
    classifier = model.classifier

    classifier = list(classifier)
    # 分类器最后一层需要符合数据集类别数量
    del classifier[6]
    # 视情况还需要删除VGG16分类器的其他层，注意要倒序删除
    if not opt.use_drop:
        del classifier[5]
        del classifier[2]
    # 拆解已经成为list的classifier并重组
    classifier = nn.Sequential(*classifier)

    # 需要冻结features部分的前四个卷积层
    for layer in features[:10]:
        for p in layer.parameters():
            p.requires_grad = False
    return nn.Sequential(*features), classifier


'''For Faster R-CNN'''
class FasterRCNNVGG16(FasterRCNN):
    feat_stride = 16

    def __init__(self, n_fg_class=20, ratios=[0.5, 1, 2], anchor_scales=[8, 16, 32]):
        # 构建特征提取器与分类器网络
        extractor, classifier = decom_vgg16()
        # 构建RPN部分
        rpn = RegionProposalNetwork(
            512, 512,
            ratios=ratios,
            anchor_scales=anchor_scales,
            feat_stride=self.feat_stride
        )

        head = VGG16RoIHead(
            n_class=n_fg_class + 1,
            roi_size=7,
            sptial_scale=(1.0 / self.feat_stride),
            classifier=classifier
        )

        # 网络组合, 利用父类FasterRCNN提供的结构
        super(FasterRCNNVGG16, self).__init__(extractor, rpn, head)


''' For RPN component '''
class RegionProposalNetwork(nn.Module):
    def __init__(
            self, in_channels=512, mid_channel=512, ratios=[0.5, 1, 2],
            anchor_scales=[8, 16, 32], feat_stride=16,
            proposal_creator_params=dict()
        ):
        super(RegionProposalNetwork, self).__init__()
        # 生成基础anchor
        self.anchor_base = generate_anchor_base(anchor_scales=anchor_scales, ratios=ratios)
        self.feat_stride = feat_stride
        # proposal_creator_params可以进行一些参数的设置
        self.proposal_layer = ProposalCreator(self, **proposal_creator_params)
        n_anchor = self.anchor_base.shape[0]
        # RPN骨干网络, kernel size:3, stride: 1, padding: 1
        self.conv1 = nn.Conv2d(in_channels, mid_channel, 3, 1, 1)
        # RPN前景背景预测分支, kernel size: 1, stride: 1, padding: 0
        # 每个anchor有2个得分，前景得分与背景得分
        self.score = nn.Conv2d(mid_channel, n_anchor * 2, 1, 1, 0)
        # RPN回归预测分支, 每个anchor有4个值
        self.loc = nn.Conv2d(mid_channel, n_anchor * 4, 1, 1, 0)
        normal_init(self.conv1, 0, 0.01)
        normal_init(self.score, 0, 0.01)
        normal_init(self.loc, 0, 0.01)

    def forward(self, x, img_size, scale=1.0):
        n, _, hh, ww = x.shape
        # 产生特征图上全部anchor，并得到这些在原图中位置，[y_rt, x_rt, y_lb, x_lb]
        anchor = _enumerate_shifted_anchor(np.array(self.anchor_base), self.feat_stride, hh, ww)
        # n_anchor := 9
        n_anchor = anchor.shape[0] // (hh * ww)

        # 对公共特征图进行卷积，后接ReLU
        h = F.relu(self.conv1(x))

        # rpn的回归分支预测偏移量
        # 按模拟图像得到 [1, 36, 37, 37]
        # [37, 37] 指每个像素，36得到的是4*9，就是每个像素9个anchor，每个anchor预测值为4个
        # 由于是全卷积，因此需要再对数据进行处理
        rpn_locs = self.loc(h)
        # 处理后得到 [1, 12321, 4]，全特征图共计12321个anchor，每个预测4个值
        rpn_locs = rpn_locs.permute(0, 2, 3, 1).contiguous().view(n, -1, 4)

        # rpn的前景背景分类分支预测是否为目标
        # [1, 18, 37, 37]
        rpn_scores = self.score(h)
        # [1, 37, 37, 18]
        rpn_scores = rpn_scores.permute(0, 2, 3, 1).contiguous()
        # [1, 37, 37, 9, 2]
        rpn_softmax_scores = F.softmax(rpn_scores.view(n, hh, ww, n_anchor, 2), dim=4)
        # [1, 37, 37, 9]
        rpn_fg_scores = rpn_softmax_scores[:, :, :, :, 1].contiguous()
        # [1, 12321]
        rpn_fg_scores = rpn_fg_scores.view(n, -1)
        # [1, 12321, 2]
        rpn_scores = rpn_scores.view(n, -1, 2)

        rois = list()
        roi_indices = list()

        for i in range(n):
            roi = self.proposal_layer(
                rpn_locs[i].cpu().data.numpy(),
                rpn_fg_scores[i].cpu().data.numpy(),
                anchor, img_size, scale=scale)
            batch_index = i * np.ones((len(roi),), dtype=np.int32)
            rois.append(roi)
            roi_indices.append(batch_index)

        rois = np.concatenate(rois, axis=0)
        roi_indices = np.concatenate(roi_indices, axis=0)
        return rpn_locs, rpn_scores, rois, roi_indices, anchor


def generate_anchor_base(base_size=16, ratios=[0.5, 1, 2], anchor_scales=[8, 16, 32]):
    # 生成基础的anchor
    py = base_size / 2.0
    px = base_size / 2.0
    # 1个像素点9个anchor
    anchor_base = np.zeros((len(ratios) * len(anchor_scales), 4),
                           dtype=np.float32)
    for i in range(len(ratios)):
        for j in range(len(anchor_scales)):
            h = base_size * anchor_scales[j] * np.sqrt(ratios[i])
            w = base_size * anchor_scales[j] * np.sqrt(1.0 / ratios[i])

            index = i * len(anchor_scales) + j
            # [y_rt, x_rt, y_lb, x_lb]
            anchor_base[index, 0] = py - h / 2
            anchor_base[index, 1] = px - w / 2
            anchor_base[index, 2] = py + h / 2
            anchor_base[index, 3] = px + w / 2
    return anchor_base


def _enumerate_shifted_anchor(anchor_base, feat_stride, height, width):
    # 特征图的每个像素点均会产生9个anchor
    # 同时需要将特征图上的anchor映射回原始图像中，分别沿x与y轴制作出shift_x以及shift_y
    # np.arange生成采样点，第一个参数为起点，第二个参数为终点(不计入)，第三个参数为步长(此处为下采样步长)
    shift_y = np.arange(0, height * feat_stride, feat_stride)
    shift_x = np.arange(0, width * feat_stride, feat_stride)
    # 合并为网格点，表现为矩阵
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    # 计算得到原图上每个anchor的偏移量，用于将anchor_base进行换算
    shift = np.stack((shift_y.ravel(), shift_x.ravel(),
                      shift_y.ravel(), shift_x.ravel()), axis=1)
    A = anchor_base.shape[0]
    K = shift.shape[0]
    # 一个像素点9个anchor，H*W的特征图共计产生9*H*W个anchor
    # 对anchor_base加以变化得到全部anchor在原图中的位置
    anchor = anchor_base.reshape((1, A, 4)) + shift.reshape((1, K, 4)).transpose((1, 0, 2))
    anchor = anchor.reshape((K * A, 4)).astype(np.float32)
    return anchor


def normal_init(m, mean, stddev, truncated=False):
    # 卷积层初始化方案
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)
    else:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()


class ProposalCreator:
    def __init__(self,
                 parent_model,
                 nms_thresh=0.7,
                 n_train_pre_nms=12000,
                 n_train_post_nms=2000,
                 n_test_pre_nms=6000,
                 n_test_post_nms=300,
                 min_size=16):
        self.parent_model = parent_model
        self.nms_thresh = nms_thresh
        self.n_train_pre_nms = n_train_pre_nms
        self.n_train_post_nms = n_train_post_nms
        self.n_test_pre_nms = n_test_pre_nms
        self.n_test_post_nms = n_test_post_nms
        self.min_size = min_size

    def __call__(self, loc, score, anchor, img_size, scale=1.0):
        if self.parent_model.training:
            n_pre_nms = self.n_train_pre_nms
            n_post_nms = self.n_train_post_nms
        else:
            n_pre_nms = self.n_test_pre_nms
            n_post_nms = self.n_test_post_nms

        # loc2bbox的作用是将rpn预测的偏移量与对应的anchor相结合，相当于得到预测的anchor
        roi = loc2bbox(anchor, loc)
        # 对anchor进行处理，超过图像边界的部分要截取
        roi[:, slice(0, 4, 2)] = np.clip(roi[:, slice(0, 4, 2)], 0, img_size[0])
        roi[:, slice(1, 4, 2)] = np.clip(roi[:, slice(1, 4, 2)], 0, img_size[1])

        # 筛选掉过小的anchor
        min_size = self.min_size * scale
        hs = roi[:, 2] - roi[:, 0]
        ws = roi[:, 3] - roi[:, 1]
        keep = np.where((hs >= min_size) & (ws >= min_size))[0]
        roi = roi[keep, :]
        score = score[keep]

        # 按照rpn预测的score由大到小筛选出n_pre_nms个anchor
        # 同时得到对应的score
        order = score.ravel().argsort()[::-1]
        if n_pre_nms > 0:
            order = order[:n_pre_nms]
        roi = roi[order, :]
        score = score[order]

        # 将这些anchor通过nms进行进一步的筛选
        keep = nms(
            torch.from_numpy(roi).cuda(),
            torch.from_numpy(score).cuda(),
            self.nms_thresh)

        # n_post_nms为最终roi保留anchor的数量，它与nms的结果取一个最小值
        # 得到最终预测的anchor
        if n_post_nms > 0:
            keep = keep[:n_post_nms]
        roi = roi[keep.cpu().numpy()]

        return roi


''' For RoI head '''
class VGG16RoIHead(nn.Module):
    def __init__(self, n_class, roi_size, sptial_scale, classifier):
        super(VGG16RoIHead, self).__init__()
        # VGG的分类器部分(已截取)
        self.classifier = classifier
        # 子网络分支1：用于进行边界框回归
        self.cls_loc = nn.Linear(4096, n_class * 4)
        # 子网络分支2：用于进行目标类别检测(算上背景类)
        self.score = nn.Linear(4096, n_class)

        normal_init(self.cls_loc, 0, 0.001)
        normal_init(self.score, 0, 0.01)

        self.n_class = n_class
        self.roi_size = roi_size
        self.spatial_scale = sptial_scale
        self.roi = RoIPool((self.roi_size, self.roi_size), self.spatial_scale)

    def forward(self, x, rois, roi_indices):
        # x: 公共特征
        roi_indices = at.totensor(roi_indices).float()
        rois = at.totensor(rois).float()
        indices_and_rois = torch.cat([roi_indices[:, None], rois], dim=1)
        # 将原来的yx记录顺序变换为xy
        xy_indices_and_rois = indices_and_rois[:, [0, 2, 1, 4, 3]]
        indices_and_rois = xy_indices_and_rois.contiguous()

        # 正式进行RoI pooling
        # [128, 512, 7, 7]， RoI pooling 可以完成特征区域规范化
        pool = self.roi(x, indices_and_rois)
        pool = pool.view(pool.size(0), -1)
        # RoI 对应的特征图输入 classifier
        fc7 = self.classifier(pool)
        # 边界框回归分支与目标分类分支
        # [128, 84] 84即21*4，算上背景共有21种目标种类
        roi_cls_locs = self.cls_loc(fc7)
        # [128, 21] 21即21*1，分类概率
        roi_scores = self.score(fc7)
        return roi_cls_locs, roi_scores


if __name__ == '__main__':
    print('Clone Dolly Tests.')