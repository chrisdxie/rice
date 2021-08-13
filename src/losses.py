import torch
import torch.nn as nn

from . import constants
from . import evaluation
from .util import utilities as util_


def compute_graph_score(pred_mask, gt_mask, 
                        obj_F_weight=0.8, obj_det_weight=0.2):
    """Compute score of graph compared with GT mask.
    
    Score with Overlap F measure and F@.75 score.

    Args:
        pred_mask: a [H x W] numpy.ndarray OR torch.Tensor with values in {0, 2, N+1}
                    Note: can also be mask of shape [N x H x W] with values in {0,1}
        gt_mask: a [H x W] numpy.ndarray OR torch.Tensor with values in {0, 2, ...}
                   Note: can also be mask of shape [N x H x W] with values in {0,1}

    Returns:
        a float.
    """

    if pred_mask.ndim == 3:
        pred_mask = util_.convert_mask_NHW_to_HW(pred_mask, start_label=constants.OBJECTS_LABEL)
    if gt_mask.ndim == 3:
        gt_mask = util_.convert_mask_NHW_to_HW(gt_mask, start_label=constants.OBJECTS_LABEL)
    if type(pred_mask) == torch.Tensor:
        pred_mask = pred_mask.cpu().numpy()
    if type(gt_mask) == torch.Tensor:
        gt_mask = gt_mask.cpu().numpy()

    score_metrics = evaluation.multilabel_metrics(pred_mask, gt_mask, 1, 1,
                                                  compute_boundary_stuff=False,
                                                  verbose=False)
    obj_F = score_metrics['Objects F-measure']
    obj_det = score_metrics['obj_detected_075_percentage']

    return obj_F_weight * obj_F + obj_det_weight * obj_det


class WeightedLoss(nn.Module):

    def __init__(self):
        super(WeightedLoss, self).__init__()
        self.weighted = False

    def generate_weight_mask(self, mask, to_ignore=None):
        """Generates a weight mask.

        Pixel weights are inversely proportional to how many pixels are in the class.

        Args:
            mask: a [N, ...] torch.FloatTensor with values in {0, 1, 2, ..., K+1},
                    where K is number of objects. {0,1} are background/table.
            to_ignore: a list of classes (integers) to ignore when creating mask.

        Returns:
            a [N, ...] torch.FloatTensor with values in [0,1]. Same shape as 'mask'.
                Note: this is not normalized to sum to 1.
        """
        N = mask.shape[0]

        if self.weighted:
            weight_mask = torch.zeros_like(mask).float()  # [N, ...]
            for i in range(N):
                unique_object_labels = torch.unique(mask[i])
                for obj in unique_object_labels:  # e.g. [0, 1, 2, 5, 9, 10]. bg, table, 4 objects
                    if to_ignore is not None and obj in to_ignore:
                        continue
                    num_pixels = torch.sum(mask[i] == obj, dtype=torch.float)
                    weight_mask[i, mask[i] == obj] = 1 / num_pixels  # inversely proportional to number of pixels

        else:  # mean over observed pixels
            weight_mask = torch.ones_like(mask)
            if to_ignore is not None:
                for obj in to_ignore:
                    weight_mask[mask == obj] = 0

        return weight_mask


class BCEWithLogitsLossWeighted(WeightedLoss):
    """Compute weighted BCE loss with logits."""

    def __init__(self, weighted=False):
        super(BCEWithLogitsLossWeighted, self).__init__()
        self.BCEWithLogitsLoss = nn.BCEWithLogitsLoss(reduction='none')
        self.weighted = weighted

    def forward(self, x, target):
        """Compute masked cosine similarity loss.

        Args:
            x: a [N x ...] torch.FloatTensor of foreground logits
            target: a [N x ...] torch.FloatTensor of values in [0, 1]

        Returns:
            a torch.Tensor scalar.
        """
        temp = self.BCEWithLogitsLoss(x, target) # Shape: [N x H x W]. values are in [0, 1]
        weight_mask = self.generate_weight_mask(target)
        loss = torch.sum(temp * weight_mask) / torch.sum(weight_mask) 

        return loss






