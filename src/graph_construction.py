import sys, os
import numpy as np
import cv2

import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
import torchvision.transforms as transforms

from . import constants
from .util import utilities as util_


def get_resnet50_fpn_model(pretrained=True, trainable_layer_names=[]):
    """Load ResNet50 + FPN model, pre-trained on COCO 2017."""

    import torchvision.models.detection.backbone_utils as backbone_utils
    from torch.utils.model_zoo import load_url as load_url

    pretrained_backbone=False
    rn50_fpn = backbone_utils.resnet_fpn_backbone('resnet50', pretrained_backbone)
    # This is an instance of BackboneWithFPN: https://github.com/pytorch/vision/blob/master/torchvision/models/detection/backbone_utils.py#L11

    if pretrained:
        model_urls = {
            'maskrcnn_resnet50_fpn_coco':
                'https://download.pytorch.org/models/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth',
        }
        pretrained_state_dict = load_url(model_urls['maskrcnn_resnet50_fpn_coco'],
                                         progress=True)

        # Hack to load only the backbone weights to the model, instead of all of MaskRCNN
        rn50_fpn_dict = rn50_fpn.state_dict()
        pretrained_dict = {k : pretrained_state_dict['backbone.' + k] for k in rn50_fpn_dict.keys()}
        rn50_fpn_dict.update(pretrained_dict)
        rn50_fpn.load_state_dict(rn50_fpn_dict)

    rn50_fpn = rn50_fpn.to(constants.DEVICE)

    # Freeze layers unless specified
    for name, parameter in rn50_fpn.named_parameters():
        parameter.requires_grad_(False)
        for layer_name in trainable_layer_names:
            if layer_name in name:
                parameter.requires_grad_(True)

    return rn50_fpn


def extract_rgb_img_features(model, img):
    """Run model (COCO2017 pre-trained ResNet50+FPN) on image.

    Args:
        model: output from get_resnet50_fpn_model()
        img: a [3 x H x W] torch.FloatTensor. Should have been standardized already

    Returns:
        an OrderedDict of torch.FloatTensors of shape [1, 256, H, W].
    """
    H,W = img.shape[1:]
    features = model(img.unsqueeze(0).to(constants.DEVICE))

    for key in features.keys():
        if key == 'pool':
            del features[key]
            continue
        features[key] = F.interpolate(features[key], size=(H,W), mode='bilinear')
    return features


def FPN_feature_key(mask):
    """Compute which FPN layer to use.

    Args:
        mask: a [H x W] torch tensor with values in {0,1}

    Returns:
        a string
    """
    x_min, y_min, x_max, y_max = util_.mask_to_tight_box(mask)
    roi_w = x_max-x_min+1; roi_h = y_max-y_min+1;
    roi_w = roi_w.float(); roi_h = roi_h.float()
    k = torch.floor(4 + torch.log2(torch.sqrt(roi_w*roi_h)/224)) # Taken from FPN paper
    k = min(max(int(k), 2), 5) 
    features_key = str(k-2) # P2 -> '0', P3 -> '1', P4 -> '2', P5 -> '3'
    return features_key


def crop_tensor_to_nchw(tensor,
                        x_min, y_min, x_max, y_max,
                        img_size=(64,64),
                        mode='bilinear'):
    """Crop a tensor and reshape.

    Args:
        tensor: a torch.Tensor of shape [H x W], [C x H x W], or [N x C x H x W]
        x_min: int
        y_min: int
        x_max: int
        y_max: int
        x_axis: int
        y_axis: int
        img_size: tuple of (H, W)

    Returns:
        a torch.Tensor of shape [N x C x img_size[0] x img_size[1]]
    """
    y_axis = tensor.ndim - 2
    x_axis = tensor.ndim - 1

    crop = torch.narrow(tensor, y_axis, y_min, y_max - y_min + 1)
    crop = torch.narrow(crop, x_axis, x_min, x_max - x_min + 1)
    
    while crop.ndim < 4:  # NCHW
        crop.unsqueeze_(0)
    
    crop = F.interpolate(crop, img_size, mode=mode)
    
    return crop


def construct_segmentation_graph(rgb_img_features,
                                 xyz_img,
                                 masks, 
                                 create_edge_indices=True,
                                 compute_bg_node=True,
                                 neighbor_dist=10,
                                 padding_config=None,
                                 device=None):
    """Construct Graph from img + masks.

    Args:
        rgb_img_features: an OrderedDict of image features. Output of extract_rgb_img_features()
        xyz_img: a [3 x H x W] torch.FloatTensor. 3D point cloud from camera frame of reference
        masks: a [H x W] torch.FloatTensor of masks in {0, 1, ..., K-1}. HW 
                    OR
               a [N x H x W] torch.FloatTensor of masks in {0,1}. NHW
        compute_bg_node: bool.
        create_edge_indices: bool.
        neighbor_dist: int. Used to create edge indices.
        padding_config: a Python dictionary with padding parameters.

    Returns:
        graph: a torch_geometric.data.Data instance with keys:
                 - rgb: a [N, 256, h, w] torch.FloatTensor of ResnNet50+FPN rgb image features
                 - depth: a [N, 3, h, w] torch.FloatTensor. XYZ image
                 - mask: a [N, h, w] torch.FloatTensor of values in {0, 1}
                 - orig_masks: a [N, H, W] torch.FloatTensor of values in {0, 1}. Original image size.
                 - crop_indices: a [N, 4] torch.LongTensor. xmin, ymin, xmax, ymax.
    """
    if device is None:
        device = constants.DEVICE

    H, W = xyz_img.shape[1:]
    if padding_config is None:
        padding_config = {
            'inference' : True, 
            'padding_percentage' : 0.25,
            'new_H' : 64,
            'new_W' : 64,
        }
    new_H = padding_config['new_H']
    new_W = padding_config['new_W']

    # Get relevant masks
    if masks.ndim == 2:
        orig_masks = util_.convert_mask_HW_to_NHW(masks, to_ignore=range(0,constants.OBJECTS_LABEL))  # [N x H x W]
    elif masks.ndim == 3:
        orig_masks = masks
        masks = util_.convert_mask_NHW_to_HW(orig_masks, start_label=constants.OBJECTS_LABEL)
    else:
        raise Exception(f"<masks> MUST be in HW or NHW format. Got shape: {masks.shape}...")
    N = orig_masks.shape[0]  # Number of objects, and nodes in graph    

    # Crop/Resize Masks/Depth
    rgb_channels_dim = 256  # hard-coded based on ResNet50+FPN output
    rgb_cr = torch.zeros((N, rgb_channels_dim, new_H, new_W), dtype=torch.float32, device=device)  # + 1 for background
    depth_cr = torch.zeros((N, 3, new_H, new_W), dtype=torch.float32, device=device)
    mask_cr = torch.zeros((N, 1, new_H, new_W), dtype=torch.float32, device=device)
    crop_indices = torch.zeros((N, 4), dtype=torch.long, device=device)

    for i, mask in enumerate(orig_masks):
        x_min, y_min, x_max, y_max = util_.crop_indices_with_padding(mask, padding_config, inference=padding_config['inference'])
        crop_indices[i] = torch.stack([x_min, y_min, x_max, y_max])

        features_key = FPN_feature_key(mask)
        layer_features = rgb_img_features[features_key] # Shape: [1 x C x h x w]. C = 256
        rgb_cr[i] = crop_tensor_to_nchw(layer_features, x_min, y_min, x_max, y_max,
                                        img_size=(new_H, new_W))[0]

        depth_cr[i] = crop_tensor_to_nchw(xyz_img, x_min, y_min, x_max, y_max,
                                          img_size=(new_H, new_W), mode='nearest')[0]
        mask_cr[i] = crop_tensor_to_nchw(mask, x_min, y_min, x_max, y_max,
                                         img_size=(new_H, new_W), mode='nearest')[0]

    # Background node
    if compute_bg_node:
        crop_indices = torch.cat([torch.LongTensor([[0, 0, W-1, H-1]]).to(device),
                                  crop_indices], axis=0)
        rgb_cr = torch.cat([crop_tensor_to_nchw(rgb_img_features['3'], *crop_indices[0]),  # deepest layer. Semantic features
                            rgb_cr], axis=0)
        depth_cr = torch.cat([crop_tensor_to_nchw(xyz_img, *crop_indices[0]).to(device),
                              depth_cr], axis=0)
        bg_orig_mask = (masks == 0).float().unsqueeze(0)  # [1, H, W]
        orig_masks = torch.cat([bg_orig_mask,
                                orig_masks], axis=0)
        mask_cr = torch.cat([crop_tensor_to_nchw(bg_orig_mask, *crop_indices[0]),
                             mask_cr], axis=0)
        N += 1

    # Check to make sure no masks are 0
    valid_indices = []
    for i in range(N):
        if torch.sum(mask_cr[i]) > 0:
            valid_indices.append(i)
    valid_indices = np.array(valid_indices)
    N = len(valid_indices)

    graph = Data(rgb=rgb_cr[valid_indices], 
                 depth=depth_cr[valid_indices],
                 mask=mask_cr[valid_indices],
                 orig_masks=orig_masks[valid_indices],
                 crop_indices=crop_indices[valid_indices],
                )
    if create_edge_indices:
        build_edge_index(graph, neighbor_dist=neighbor_dist)
    graph = graph.to(device)

    return graph


def build_edge_index(graph, neighbor_dist):
    edge_index = util_.neighboring_mask_indices(graph.orig_masks, reduction_factor=1,
                                                neighbor_dist=neighbor_dist)
    edge_index = torch.cat([edge_index, edge_index.flip([1])], dim=0).T # Shape: [2 x E]
    graph.edge_index = edge_index.to(graph.mask.device)


def remove_bg_node(data_list):
    """Return a list of new graphs with background node removed.

    Note: the RGB/Depth/Mask is not copied over, but assigned. Thus, losses can be applied to
          the new graphs (w/out BG nodes) and gradients will still flow through the old graphs.

    Args:
        graph: Can be a torch_geometric.Data instance, torch_geometric.Batch instance,
            or a List of torch_geometric.Data instances.

    Returns:
        Same data type as input. A copy of graphs, but without background nodes and update edge_indices.
    """
    if isinstance(data_list, Data):
        input_type = 'Data'
        data_list = [data_list]
    elif isinstance(data_list, Batch):
        data_list = Batch.to_data_list(data_list)
        input_type = 'Batch'
    elif isinstance(data_list, list):
        input_type = 'list'
    else:
        raise NotImplementedError()
    # Note: data_list is now of type list

    new_data_list = []    
    for graph in data_list:

        # Double check to make sure background node hasn't already been removed
        if 'background_removed' in graph:
            raise Exception("Cannot remove background node if it has already been removed...")

        new_graph = Data()
        new_graph.rgb = graph.rgb[1:]
        new_graph.depth = graph.depth[1:]
        new_graph.mask = graph.mask[1:]
        new_graph.orig_masks = graph.orig_masks[1:]
        new_graph.crop_indices = graph.crop_indices[1:]
        new_graph.background_removed = True

        # Special cases
        if 'edge_index' in graph:
            edge_mask = torch.all(graph.edge_index != 0, dim=0)  # [E]
            new_graph.edge_index = graph.edge_index[:, edge_mask] - 1  # -1 since we removed background
        if 'paths' in graph and 'split' in graph:  # Splitting is stored
            new_graph.paths = {k - 1: graph.paths[k] for k in graph.paths.keys()}
            new_graph.split = graph.split[1:]

        new_data_list.append(new_graph)

    if input_type == 'Data':
        return new_data_list[0]
    elif input_type == 'Batch':
        return convert_list_to_batch(new_data_list)
    elif input_type == 'list':
        return new_data_list


def convert_list_to_batch(graph_list, external_key='crop_indices'):
    """Convert list of graphs into a Batch(Data) instance.

    Args:
        graph_list: a Python list of torch_geometric.data.Data instances

    Returns:
        a torch_geometric.data.Batch instance
    """
    for graph in graph_list:
        if 'x' not in graph.keys: # Batch.from_data_list needs 'x' to run correctly (to compute graph.num_nodes)
            graph.x = graph[external_key]
    return Batch.from_data_list(graph_list)


def convert_batch_to_list(batch_graph):
    """Convert Batch(Data) instance into a list of Data instances.
       
    Undoes the convert_list_to_batch() function.

    Args:
        batch_graph: a torch_geometric.Batch instance

    Returns:
        a Python list of torch_geometric.data.Data instances.
    """
    return Batch.to_data_list(batch_graph)


def get_edge_graph(graph, rgb_img_features, xyz_img, padding_config=None):
    """Compute graph where each node is an edge of original graph.

    Creates a new graph such that each node in the new graph corresponds
        to an edge in the original graph. The new graph is constructed
        in the same way, but the crop_indices cover the union of the
        masks. This graph has no edges.

    Args:
        graph: a torch_geometric.data.Data instance
        rgb_img_features: an OrderedDict of image features. Output of extract_rgb_img_features()
        xyz_img: a [3 x H x W] torch.FloatTensor. 3D point cloud from camera frame of reference
        padding_config: a Python dictionary.

    Returns:
        a torch_geometric.Data instance
    """
    union_orig_masks = torch.clamp(graph.orig_masks[graph.edge_index[0]] + \
                                   graph.orig_masks[graph.edge_index[1]], max=1) # Shape: [E x H x W]
    return construct_segmentation_graph(
                rgb_img_features,
                xyz_img,
                union_orig_masks, 
                compute_bg_node=False,
                create_edge_indices=False,
                padding_config=padding_config
            )


def add_zero_channel_to_masks(graph):
    """Add an empty channel of 0's to graph.mask."""
    graph.mask = torch.cat([graph.mask, torch.zeros_like(graph.mask)], dim=1)

