# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Utilities for bounding box manipulation and GIoU.
"""
import torch
from torchvision.ops.boxes import box_area


def box_cxcywh_to_xyxy(x: torch.tensor):
    """ convert bounding boxes from [cx,cy,w,h] to [x1,y1,x2,y2]

    :param x: batched bounding boxes [N,4] in [cx,cy,w,h]
    :return: batched bounding boxes [N,4] in [x1,y1,x2,y2]
    """
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x: torch.tensor):
    """ convert bounding boxes from [x1,y1,x2,y2] to [cx,cy,w,h]

    :param x: batched bounding boxes [N,4] in [x1,y1,x2,y2]
    :return: batched bounding boxes [N,4] in [cx,cy,w,h]
    """
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


# modified from torchvision to also return the union
def box_iou(boxes1: torch.tensor, boxes2: torch.tensor):
    """ calculate iou between all possible pairs between boxes1 and boxes2

    :param boxes1: batched bounding boxes [N,4] in [x1,y1,x2,y2]
    :param boxes2: batched bounding boxes [M,4] in [x1,y1,x2,y2]
    :return: matrix containing the iou [N,M], matrix containing the union [N,M]
    """
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1: torch.tensor, boxes2: torch.tensor):
    """ generalized IoU from https://giou.stanford.edu/

    :param boxes1: batched bounding boxes [N,4] in [x1,y1,x2,y2]
    :param boxes2: batched bounding boxes [M,4] in [x1,y1,x2,y2]
    :return: a [N, M] pairwise matrix
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area


def normalize_box(box: torch.tensor, image_size: tuple):
    """ normalize box coordinates such that the top-left corner is [0,0] and the bottom-right is [1,1]

    :param box: batched bounding boxes [N,4]/[N,8] in [x1,y1,x2,y2]
    :param image_size: original image sizes in [w,h]
    :return: batched normalized bounding boxes [N,4]/[N,8] in [x1,y1,x2,y2]
    """
    box[..., 0] = box[..., 0] / image_size[0]
    box[..., 1] = box[..., 1] / image_size[1]
    box[..., 2] = box[..., 2] / image_size[0]
    box[..., 3] = box[..., 3] / image_size[1]

    if box.shape[-1] == 8:
        box[..., 4] = box[..., 4] / image_size[0]
        box[..., 5] = box[..., 5] / image_size[1]
        box[..., 6] = box[..., 6] / image_size[0]
        box[..., 7] = box[..., 7] / image_size[1]
    return box


def unnormalize_box(box, image_size):
    """ reverts normalized bounding boxes to their original values

        :param box: batched bounding boxes [N,4]/[N,8] in [x1,y1,x2,y2]
        :param image_size: original image sizes in [w,h]
        :return: batched bounding boxes [N,4]/[N,8] in [x1,y1,x2,y2]
        """
    box[..., 0] = box[..., 0] * image_size[0]
    box[..., 1] = box[..., 1] * image_size[1]
    box[..., 2] = box[..., 2] * image_size[0]
    box[..., 3] = box[..., 3] * image_size[1]

    if box.shape[-1] == 8:
        box[..., 4] = box[..., 4] * image_size[0]
        box[..., 5] = box[..., 5] * image_size[1]
        box[..., 6] = box[..., 6] * image_size[0]
        box[..., 7] = box[..., 7] * image_size[1]
    return box
