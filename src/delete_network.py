
import itertools
from collections import OrderedDict
import numpy as np
import cv2

import torch
import torch.nn as nn
from torch.nn import Sequential as Seq, Linear, ReLU
import torch.nn.functional as F
from torch_geometric.data import Data, Batch

from . import base_networks
from . import graph_construction as gc
from . import constants
from .util import utilities as util_


class DeleteNet(nn.Module):

    def __init__(self, config):
        super(DeleteNet, self).__init__()
        self.node_encoder = base_networks.NodeEncoder(config['node_encoder_config'])
        self.bg_fusion_module = base_networks.LinearEncoder(config['bg_fusion_module_config'])

    def forward(self, graph):
        """DeleteNet forward pass.

        Note: Assume that the graph contains the background node as the first node.

        Args:
            graph: a torch_geometric.Data instance with attributes:
                - rgb: a [N, 256, h, w] torch.FloatTensor of ResnNet50+FPN rgb image features
                - depth: a [N, 3, h, w] torch.FloatTensor. XYZ image
                - mask: a [N, h, w] torch.FloatTensor of values in {0, 1}
                - orig_masks: a [N, H, W] torch.FloatTensor of values in {0, 1}. Original image size.
                - crop_indices: a [N, 4] torch.LongTensor. xmin, ymin, xmax, ymax.
        
        Returns:
            a [N] torch.FloatTensor of delete score logits. The first logit (background) is always low,
                so BG is never deleted.
        """
        encodings = self.node_encoder(graph)  # dictionary
        concat_features = torch.cat([encodings[key] for key in encodings], dim=1)  # [N, \sum_i d_i]

        bg_feature = concat_features[0:1]  # [1, \sum_i d_i]
        node_features = concat_features[1:]  # [N-1, \sum_i d_i]
        node_minus_bg_features = node_features - bg_feature  # [N-1, \sum_i d_i]

        node_delete_logits = self.bg_fusion_module(node_minus_bg_features)  # [N-1, 1]
        delete_logits = torch.cat([torch.ones((1, 1), device=constants.DEVICE) * -100,
                                   node_delete_logits], dim=0)
        return delete_logits[:,0]


class DeleteNetWrapper(base_networks.NetworkWrapper):

    def setup(self):

        if 'deletenet_model' in self.config:
            self.model = self.config['deletenet_model']
        else:
            self.model = DeleteNet(self.config)
        self.model.to(self.device)


    def get_new_potential_masks(self, masks, fg_mask):
        """Compute new potential masks.

        See if any connected components of fg_mask _setminus_ mask can be 
            considered as a new mask. Concatenate them to masks.

        Args:
            masks: a [N, H, W] torch.Tensor with values in {0, 1}. 
            fg_mask: a [H, W] torch.Tensor with values in {0, 1}.

        Returns:
            a [N + delta, H, W] np.ndarray of new masks. delta = #new_masks.
        """
        occupied_mask = masks.sum(dim=0) > 0.5

        fg_mask = fg_mask.cpu().numpy().astype(np.uint8)
        fg_mask[occupied_mask.cpu().numpy()] = 0 
        fg_mask = cv2.erode(fg_mask, np.ones((3,3)), iterations=1)
        
        nc, components = cv2.connectedComponents(fg_mask, connectivity=8)
        components = torch.from_numpy(components).float().to(constants.DEVICE)
        for j in range(1, nc):
            mask = components == j
            component_size = mask.sum().float()
            if component_size > self.config['min_pixels_thresh']:
                masks = torch.cat([masks, mask[None].float()], dim=0)

        return masks


    def delete_scores(self, graph):
        """Compute delete scores for each node in the graph.

        Args:
            graph: a torch_geometric.Data instance

        Returns:
            a [N] torch.Tensor with values in [0, 1]
        """
        return torch.sigmoid(self.model(graph))

