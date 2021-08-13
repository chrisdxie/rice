
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear, ReLU

from abc import ABC, abstractmethod


def maxpool2x2(input, ksize=2, stride=2):
    """2x2 max pooling"""
    return nn.MaxPool2d(ksize, stride=stride)(input)
def flatten(input):
    return nn.Flatten()(input)
def relu(input):
    return nn.ReLU()(input)


class Conv2d_GN_ReLU(nn.Module):
    """
    Implements a module that performs conv2d + groupnorm + ReLU.
    Assumes kernel size is odd.
    """

    def __init__(self, in_channels, out_channels, num_groups, ksize=3, stride=1):
        super(Conv2d_GN_ReLU, self).__init__()
        padding = 0 if ksize < 2 else ksize//2
        self.conv1 = nn.Conv2d(in_channels, out_channels, 
                               kernel_size=ksize, stride=stride, 
                               padding=padding, bias=False)
        self.gn1 = nn.GroupNorm(num_groups, out_channels)
        self.relu1 = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.gn1(out)
        out = self.relu1(out)

        return out

class Conv2d_GN_ReLUx2(nn.Module):
    """ 
    Implements a module that performs 
        conv2d + groupnorm + ReLU + 
        conv2d + groupnorm + ReLU
        (and a possible downsampling operation)
    Assumes kernel size is odd.
    """

    def __init__(self, in_channels, out_channels, num_groups, ksize=3, stride=1):
        super(Conv2d_GN_ReLUx2, self).__init__()
        self.layer1 = Conv2d_GN_ReLU(in_channels, out_channels, 
                                     num_groups, ksize=ksize, stride=stride)
        self.layer2 = Conv2d_GN_ReLU(out_channels, out_channels, 
                                     num_groups, ksize=ksize, stride=stride)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)

        return out

class Upsample_Concat_Conv2d_GN_ReLU_Multi_Branch(nn.Module):
    """ 
    Implements a module that performs
        Upsample (reduction: conv2d + groupnorm + ReLU + bilinear_sampling) +
        concat + conv2d + groupnorm + ReLU 
    for the U-Net decoding architecture with an arbitrary number of encoders

    The Upsample operation consists of a Conv2d_GN_ReLU that reduces the channels by 2,
        followed by bilinear sampling

    Note: in_channels is number of channels of ONE of the inputs to the concatenation
    """
    def __init__(self, in_channels, out_channels, num_groups, skip_channels, ksize=3, stride=1):
        super(Upsample_Concat_Conv2d_GN_ReLU_Multi_Branch, self).__init__()
        self.channel_reduction_layer = Conv2d_GN_ReLU(in_channels, in_channels//2, num_groups)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv_gn_relu = Conv2d_GN_ReLU(int(in_channels//2 + skip_channels), out_channels, num_groups)

    def forward(self, x, skips):
        """Forward module.

        Args:
            x: torch tensor.
            skips: a list of intermediate skip-layer torch tensors from each encoder
        """
        x = self.channel_reduction_layer(x)
        x = self.upsample(x)
        out = torch.cat([x] + skips, dim=1) # Concat on channels dimension
        out = self.conv_gn_relu(out)

        return out




################## Network Definitions ##################

class LinearEncoder(nn.Module):
    """
    Just 2 linear layers.
    """

    def __init__(self, config):
        super(LinearEncoder, self).__init__()
        self.layers = [Linear(config['input_dim'], config['hidden'][0]), ReLU()]  # first layer
        for l in range(1, len(config['hidden'])):  # hidden layers
            self.layers.append(Linear(config['hidden'][l-1], config['hidden'][l]))
            self.layers.append(ReLU())
        self.layers.append(Linear(config['hidden'][-1], config['output_dim']))  # last layer
        if config['final_relu']:
            self.layers.append(ReLU())
        self.layers = Seq(*self.layers)

    def forward(self, x):
        if x.ndim != 2:
            x = flatten(x)
        return self.layers(x)

class CNNEncoder(nn.Module):
    """Based on Y-Net architecture from (Xie et al. CVPR 2019)"""

    def __init__(self, config):
        super(CNNEncoder, self).__init__()
        self.ic = config['input_channels']
        self.oc = config['output_channels']
        self.build_network()
        
    def build_network(self):
        """Build encoder network, using a U-Net-like architecture."""

        ### Encoder ###
        self.layer1 = Conv2d_GN_ReLUx2(self.ic, self.oc//8, self.oc//16) # groups of 2
        self.layer2 = Conv2d_GN_ReLUx2(self.oc//8, self.oc//4, self.oc//16) # groups of 4
        self.layer3 = Conv2d_GN_ReLUx2(self.oc//4, self.oc//2, self.oc//8) # groups of 4
        self.last_layer = Conv2d_GN_ReLU(self.oc//2, self.oc, self.oc//4) # groups of 4


    def forward(self, images):

        x1 = self.layer1(images)
        mp_x1 = maxpool2x2(x1)
        x2 = self.layer2(mp_x1)
        mp_x2 = maxpool2x2(x2)
        x3 = self.layer3(mp_x2)
        mp_x3 = maxpool2x2(x3)
        x4 = self.last_layer(mp_x3)

        return [x1, x2, x3, x4]

class NodeEncoder(nn.Module):

    def __init__(self, config):
        super(NodeEncoder, self).__init__()

        self.encoders = nn.ModuleDict()  # Linear OR CNN
        self.additional_linear_encoders = nn.ModuleDict()  # linear on top of CNN
        self.ap_encoders = OrderedDict()  # Average Pooling
        self.mask_average = {}  # Useful for linear encoders

        for key in config.keys():
            if 'encoder_config' not in key:
                continue
            
            encoder_key = key.replace('_encoder_config', '')

            if config[key]['type'] == 'cnn':
                self.encoders[encoder_key] = CNNEncoder(config[key])
                if 'linear_encoder_config' in config[key]:
                    ap_ksize = config[key]['avg_pool_kernel_size']
                    self.ap_encoders[encoder_key] = nn.AvgPool2d((ap_ksize, ap_ksize))
                    self.additional_linear_encoders[encoder_key] = LinearEncoder(config[key]['linear_encoder_config'])

            elif config[key]['type'] == 'linear':
                self.encoders[encoder_key] = LinearEncoder(config[key])

            if 'mask_average' in config[key] and config[key]['mask_average']:
                self.mask_average[encoder_key] = True
            else:
                self.mask_average[encoder_key] = False

        self.fusion_module = None
        if 'fusion_module_config' in config:
            self.fusion_module = LinearEncoder(config['fusion_module_config'])

    def forward(self, graph):
        """Forward pass of node encoder.
        
        Note: N = #nodes

        Args…
            graph: a torch_geometric.Data instance with attributes:
                        - rgb: a [N x 256 x H' x W'] torch.FloatTensor of rgb features
                        - depth: a [N x 3 x H' x W'] torch.FloatTensor
                        - mask: a [N x 1 x H' x W'] torch.FloatTensor
                        - orig_masks: a [N x H x W] torch.FloatTensor of original masks

        Returns:
            an OrderedDict of CNNEncoder/LinearEncoder outputs
                for each CNNEncoder key, the value is a list [x1, x2, x3, x4]
                    of 4 torch.FloatTensors of shape: [N x C x H' x W']
                for each LinearEncoder key, the value is a [N x C] torch tensor
        """
        encodings = OrderedDict()

        # First, run the encoders
        for key in self.encoders.keys():

            encoder_input = graph[key]
            if self.mask_average[key]:
                encoder_input = torch.sum(encoder_input * graph.mask, dim=[2,3], keepdim=True) / torch.sum(graph.mask, dim=[2,3], keepdim=True)
            encodings[key] = self.encoders[key](encoder_input)

        # Next, run linear encoders on top of CNN outputs if they exist
        for key in self.additional_linear_encoders.keys():
            x = self.ap_encoders[key](encodings[key][3])  # index 3 is the final output of the encoder
            encodings[key] = self.additional_linear_encoders[key](x) 

        # Fusion module if it exists
        if self.fusion_module:
            concat_features = torch.cat([encodings[key] for key in encodings], dim=1) # Shape: [N x \sum_i d_i]
            fused_features = self.fusion_module(concat_features)
            return fused_features     
        else:
            return encodings


class SplitNetDecoder(nn.Module):
    """Based on Y-Net decoder from Xie et. al (CVPR 2019)."""

    def __init__(self, config):
        super(SplitNetDecoder, self).__init__()
        self.oc = config['output_channels']
        self.eoc = config['encoder_output_channels']  # This is an OrderedDict of number of channels of each CNN encoder
        self.eod = config['encoder_output_dims']  # This is an OrderedDict of number of dimensions of each Linear encoder. These are only used for split scores
        self.img_size = np.array(config['img_size'])  # np.array: [h,w]
        self.build_network()

    def build_network(self):
        """Build a decoder network using a U-Net-like architecture."""
        reduction_factor = 8

        # Fusion layer
        self.fuse_layer = Conv2d_GN_ReLU(sum(self.eoc.values()), self.oc*8, self.oc, ksize=1)

        # Split score branch (after fusion layer)
        ap_ksize = list((self.img_size / reduction_factor // 4).astype(np.int)) # apply AvgPool2d to get [N x C x 4 x 4]        
        self.split_score_avg_pool = nn.AvgPool2d(ap_ksize)
        self.split_score_1 = nn.Linear(self.oc*8 * 16 + sum(self.eod.values()), 1024)
        self.split_score_2 = nn.Linear(1024, 256)
        self.split_score_3 = nn.Linear(256, 1)

        # Decoding branch (after fusion layer)
        dl1_skip_channels = sum([x//2 for x in self.eoc.values()])
        self.decoder_layer1 = Upsample_Concat_Conv2d_GN_ReLU_Multi_Branch(self.oc*8, self.oc*4, self.oc, dl1_skip_channels)
        dl2_skip_channels = sum([x//4 for x in self.eoc.values()])
        self.decoder_layer2 = Upsample_Concat_Conv2d_GN_ReLU_Multi_Branch(self.oc*4, self.oc*2, self.oc//2, dl2_skip_channels)
        dl3_skip_channels = sum([x//8 for x in self.eoc.values()])
        self.decoder_layer3 = Upsample_Concat_Conv2d_GN_ReLU_Multi_Branch(self.oc*2, self.oc, self.oc//4, dl3_skip_channels)

        # Final layer
        self.decoder_layer4 = Conv2d_GN_ReLU(self.oc, self.oc, self.oc//4)

        # This puts features everywhere, not just nonnegative orthant
        self.decoder_last_conv = nn.Conv2d(self.oc, 1, kernel_size=3,
                                   stride=1, padding=1, bias=True)


    def forward(self, encoder_outputs):
        """Forward function.

        Args:
            encoder_outputs: an OrderedDict of lists. 
                               - each list has: 4 torch tensors of intermediate outputs
            mask_boundary: a [N x 1 x H x W] torch.FloatTensor

        Returns:
            a [N x 1] torch.FloatTensor of split score logits (NOTE: this is not used...)
            a [N x 1 x H x W] torch.FloatTensor of boundary score logits
        """
        cnn_encoder_outputs = OrderedDict({k:v for (k,v) in encoder_outputs.items() if type(v) == list})

        # Apply fusion layer to the concatenation of encoder outputs
        fuse_out = torch.cat([cnn_encoder_outputs[key][3] for key in cnn_encoder_outputs], dim=1)
        fuse_out = self.fuse_layer(fuse_out)

        # Apply split score branch
        ss_out = flatten(self.split_score_avg_pool(fuse_out))
        ss_out = relu(self.split_score_1(ss_out))
        ss_out = relu(self.split_score_2(ss_out)) # Shape: [N x 256]
        ss_out = self.split_score_3(ss_out)

        # Apply decoder branch
        d_out = self.decoder_layer1(fuse_out, [cnn_encoder_outputs[key][2] for key in cnn_encoder_outputs])
        d_out = self.decoder_layer2(d_out, [cnn_encoder_outputs[key][1] for key in cnn_encoder_outputs])
        d_out = self.decoder_layer3(d_out, [cnn_encoder_outputs[key][0] for key in cnn_encoder_outputs])
        d_out = self.decoder_layer4(d_out)
        d_out = self.decoder_last_conv(d_out)

        return ss_out, d_out




################## Network Wrapper ##################

# dictionary keys returned by dataloaders that you don't want to send to GPU
dont_send_to_device = [
    'scene_dir', 
    'view_num', 
    'subset', 
    'supporting_plane', 
    'label_abs_path',
]

class NetworkWrapper(ABC):

    def __init__(self, config):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.config = config.copy()

        # Build network and losses
        self.setup()

    @abstractmethod
    def setup(self):
        pass

    def train_mode(self):
        """Put all modules into train mode."""
        self.model.train()

    def eval_mode(self):
        """Put all modules into eval mode."""
        self.model.eval()

    def send_batch_to_device(self, batch):
        for key in batch.keys():
            if key in dont_send_to_device:
                continue
            if len(batch[key]) == 0: # can happen if a modality (e.g. RGB) is not loaded
                continue
            batch[key] = batch[key].to(self.device)

    def save(self, filename):
        """Save the model as a checkpoint."""
        checkpoint = {'model' : self.model.state_dict()}
        torch.save(checkpoint, filename)

    def load(self, filename):
        """Load the model checkpoint."""
        checkpoint = torch.load(filename)
        self.model.load_state_dict(checkpoint['model'])
        print(f"Loaded {self.__class__.__name__} model from: {filename}")

