import torch
from utils.box_ops import generalized_box_iou, box_cxcywh_to_xyxy
import torch.nn.functional as F

class Criterion(torch.nn.Module):
    def __init__(self, losses: list, loss_coefs: list):
        self.losses = losses
        self.coefs = loss_coefs

    def forward(self):
        loss_l1 = 0.
        loss_giou = 0.
        curr_gt_step = self.interpolation(x_inital, curr_gt)
        all_pred_bboxes = torch.stack(inter_pred_bboxes)
        for n, pred in enumerate(all_pred_bboxes):
            loss_l1 += F.smooth_l1_loss(pred.view(-1, p_dim), curr_gt_step[n].view(-1, p_dim), reduction='none')
            loss_giou += 1 - torch.diag(
                generalized_box_iou(
                    box_cxcywh_to_xyxy(pred.view(-1, p_dim)),
                    box_cxcywh_to_xyxy(curr_gt_step[n].view(-1, p_dim))
                )
            )

        loss = loss_l1.mean() + loss_giou.mean()
        return loss