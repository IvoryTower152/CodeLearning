import cv2
import numpy as np
import random
import torch

from game2 import FasterRCNNVGG16
from game3 import FasterRCNNTrainer
from skimage import transform as sktsf
from torchvision import transforms as tvtsf
from data import util
from utils import array_tool as at


# 伪样本生成
def generate_pseudo_sample():
    img = np.ones((600, 600), dtype=np.uint8)
    # 3通道
    bgr_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    # bgr_img[:, :, 0] = 0
    # bgr_img[:, :, 1] = 0
    # bgr_img[:, :, 2] = 255
    for i in range(len(bgr_img)):
        for j in range(len(bgr_img[i])):
            # 随机采样3个[0, 255]的整数用于填色
            bgr_img[i][j] = random.sample(range(0, 255), 3)
    # [x_min, y_min, x_max, y_max]
    # box = [100, 200, 300, 400]
    # bounding box 组成为 [[y_min, x_min, y_max, x_max], ...]
    box = [[200, 100, 400, 300]]
    box = np.stack(box).astype(np.float32)
    # cv2.imshow('Test', bgr_img)
    # cv2.waitKey(0)
    return bgr_img, box


def preprocess(img, box, min_size=600, max_size=1000):
    C, H, W = img.shape
    scale1 = min_size / min(H, W)
    scale2 = max_size / max(H, W)
    scale = min(scale1, scale2)
    img = img / 255.0
    img = sktsf.resize(img, (C, H*scale, W*scale), mode='reflect', anti_aliasing=False)
    normalize = tvtsf.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
    img = normalize(torch.from_numpy(img))
    img = img.numpy()

    C, o_H, o_W = img.shape
    scale = o_H / H
    bbox = util.resize_bbox(box, (H, W), (o_H, o_W))

    # 伪造标签流程
    label = []
    label.append(0)
    label = np.stack(label).astype(np.int32)

    return img.copy(), bbox.copy(), label.copy(), scale



def train():
    # 模型与训练器
    faster_rcnn = FasterRCNNVGG16()
    trainer = FasterRCNNTrainer(faster_rcnn).cuda()

    # 伪造数据
    img, box = generate_pseudo_sample()
    # 以下代码可以将opencv读取的图像(HWC, BGR)转为(CHW, RGB)
    img = img[:, :, ::-1].transpose((2, 0, 1))
    img, bbox, label, scale = preprocess(img, box)

    # 伪造的数据需要转换为tensor, 单一样本也需要封装成mini-batch形式
    # [1, 3, 160, 160]
    img = torch.from_numpy(np.array([img]))
    # [1, 1, 4]
    bbox = torch.from_numpy(np.array([bbox]))
    # [1, 1]
    label = torch.from_numpy(np.array([label]))

    # 模拟样本，这步不需要
    # scale = at.scalar(scale)

    img, bbox, label = img.cuda().float(), bbox.cuda(), label.cuda()
    trainer.train_step(img, bbox, label, scale)


if __name__ == '__main__':
    print('Clone Dolly Tests.')
    # img, box = generate_pseudo_sample()
    # cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
    # cv2.imshow('test', img)
    # cv2.waitKey(0)
    train()