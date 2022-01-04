# 将下一版本的特性提前导入到当前python版本中
from __future__ import absolute_import
from __future__ import division

import torch as t
import numpy as np
from utils import array_tool as at
from model.utils.bbox_tools import loc2bbox
from torchvision.ops import nms

from torch import nn
from data.dataset import preprocess
from torch.nn import functional as F
from utils.config import opt


# 函数装饰器，为代码增加额外的功能
def nogard(f):
    def new_f(*args, **kwargs):
        with t.no_grad():
            return f(*args, **kwargs)
    return new_f


# Faster R-CNN本体
class FasterRCNN(nn.Module):
    # 原始版本初始化时需要提供特征提取器、RPN模块以及RoIhead模块
    def __init__(self, extractor, rpn, head,
                 loc_normalize_mean = (0.0, 0.0, 0.0, 0.0),
                 loc_normalize_std = (0.1, 0.1, 0.2, 0.2)
    ):
        super(FasterRCNN, self).__init__()
        self.extractor = extractor
        self.rpn = rpn
        self.head = head

        self.loc_normalize_mean = loc_normalize_mean
        self.loc_normalize_std = loc_normalize_std
        # 预设置载入
        self.use_preset('evaluate')

    # 将函数设为只读属性(private)，防止被修改
    @property
    def n_class(self):
        # 获取目标的类别数(算上背景类)
        return self.head.n_class

    def forward(self, x, scale=1.0):
        img_size = x.shape[2:]
        h = self.extractor(x)
        rpn_locs, rpn_scores, rois, roi_indices, anchor = self.rpn(h, img_size, scale)
        roi_cls_locs, roi_scores = self.head(h, rois, roi_indices)
        return roi_cls_locs, roi_scores, rois, roi_indices

    # 一些预设的条件
    def use_preset(self, preset):
        if preset == 'visualize':
            self.nms_thresh = 0.3
            self.score_thresh = 0.7
        elif preset == 'evaluate':
            self.nms_thresh = 0.3
            self.score_thresh = 0.05
        else:
            raise ValueError('preset must be visualize or evaluate')

    def _suppress(self):
        pass

    # 使用声明好的装饰器
    @nogard
    def predict(self, imgs, sizes=None, visualize=False):
        pass

    def get_optimizer(self):
        lr = opt.lr
        params = []
        for key, value in dict(self.named_parameters()).items():
            if value.requires_grad:
                if 'bias' in key:
                    params += [{'params': [value], 'lr': lr * 2, 'weight_decay': 0}]
                else:
                    params += [{'params': [value], 'lr': lr, 'weight_decay': opt.weight_decay}]
        if opt.use_adam:
            self.optimizer = t.optim.Adam(params)
        else:
            self.optimizer = t.optim.SGD(params, momentum=0.9)
        return self.optimizer


if __name__ == '__main__':
    print('Clone Dolly Tests.')