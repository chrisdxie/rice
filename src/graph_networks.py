from collections import OrderedDict
import itertools

import numpy as np

import torch
import torch.nn as nn
from torch.nn import Sequential as Seq, Linear, ReLU
from torch_scatter import scatter_mean
from torch_geometric.data import Data

from . import graph_construction as gc
from . import base_networks
from . import constants
from .util import utilities as util_


def groupnorm_option(num_channels, use_gn):
    if use_gn:
        return nn.GroupNorm(8, num_channels)  # hard-coded 8 channels
    else:
        return nn.Identity()


class EdgeModel(torch.nn.Module):

    def __init__(self, config):
        super(EdgeModel, self).__init__()

        input_channels = config['edge_input_channels'] + 2*config['node_input_channels']
        hidden_size_1 = config['edge_model_hidden'][0]
        hidden_size_2 = config['edge_model_hidden'][1]

        self.edge_mlp = Seq(Linear(input_channels, hidden_size_1),
                            groupnorm_option(hidden_size_1, config['use_groupnorm']),
                            ReLU(),
                            Linear(hidden_size_1, hidden_size_2),
                            groupnorm_option(hidden_size_2, config['use_groupnorm']),
                            ReLU(),
                            Linear(hidden_size_2, config['edge_output_channels']))

    def forward(self, src, dest, edge_attr):
        """Edge Model of Graph Net layer.

        Args:
            src: [E x node_input_channels], where E is the number of edges.
                 Treat E as the batch size.
            dest: [E x node_input_channels], where E is the number of edges.
            edge_attr: [E x edge_input_channels]

        Returns:
            a [E x edge_output_channels] torch.FloatTensor
        """

        src_dest_edge = torch.cat([src, dest, edge_attr], 1)  # [E x (2*node_input_channels + edge_input_channels)]
        return self.edge_mlp(src_dest_edge)  # [E x edge_output_channels]


class NodeModel(torch.nn.Module):

    def __init__(self, config):
        super(NodeModel, self).__init__()

        mlp1_input_channels = config['node_input_channels'] + config['edge_output_channels']
        mlp1_hidden_size_1 = config['node_model_mlp1_hidden'][0]
        mlp1_hidden_size_2 = config['node_model_mlp1_hidden'][1]

        mlp2_input_channels = config['node_input_channels'] + mlp1_hidden_size_2
        mlp2_hidden_size_1 = config['node_model_mlp2_hidden'][0]

        self.node_mlp_1 = Seq(Linear(mlp1_input_channels, mlp1_hidden_size_1),
                              groupnorm_option(mlp1_hidden_size_1, config['use_groupnorm']),
                              ReLU(),
                              Linear(mlp1_hidden_size_1, mlp1_hidden_size_2),
                              groupnorm_option(mlp1_hidden_size_2, config['use_groupnorm']),
                              ReLU(),
                              )
        self.node_mlp_2 = Seq(Linear(mlp2_input_channels, mlp2_hidden_size_1),
                              groupnorm_option(mlp2_hidden_size_1, config['use_groupnorm']),
                              ReLU(),
                              Linear(mlp2_hidden_size_1, config['node_output_channels']))

    def forward(self, x, edge_index, edge_attr):
        """Node Model of Graph Net layer.

        Args:
            x: [N x node_input_channels], where N is the number of nodes.
            edge_index: [2 x E] with max entry N - 1, where E is number of edges.
            edge_attr: [E x edge_input_channels]

        Returns:
            a [N x node_output_channels] torch tensor
        """
        row, col = edge_index

        srcnode_edge = torch.cat([x[row], edge_attr], dim=1)  # [E x (node_input_channels + edge_output_channels)]
        srcnode_edge = self.node_mlp_1(srcnode_edge)  # [E x hidden_size_2]
        per_node_aggs = scatter_mean(srcnode_edge, col, dim=0, dim_size=x.size(0))  # Mean-aggregation for every dest node. [N x hidden_size_2]

        node_agg_u = torch.cat([x, per_node_aggs], dim=1)
        out = self.node_mlp_2(node_agg_u)  # [N x node_output_channels]

        return out


class ResidualMetaLayer(torch.nn.Module):
    """Adapted from https://pytorch-geometric.readthedocs.io/en/1.3.2/_modules/torch_geometric/nn/meta.html.
        - removes global model
        - residual addition to input
    """

    def __init__(self, config):
        super(ResidualMetaLayer, self).__init__()
        self.edge_model = EdgeModel(config)
        self.edge_channel_reducer = None
        if config['edge_input_channels'] != config['edge_output_channels']:
            self.edge_channel_reducer = Linear(config['edge_input_channels'], config['edge_output_channels'])

        self.node_model = NodeModel(config)
        self.node_channel_reducer = None
        if config['node_input_channels'] != config['node_output_channels']:
            self.node_channel_reducer = Linear(config['node_input_channels'], config['node_output_channels'])

        self.reset_parameters()

    def reset_parameters(self):
        for item in [self.node_model, self.edge_model]:
            if hasattr(item, 'reset_parameters'):
                item.reset_parameters()


    def forward(self, x, edge_index, edge_attr):
        row, col = edge_index

        edge_attr_res = self.edge_model(x[row], x[col], edge_attr)
        if self.edge_channel_reducer is not None:
            edge_attr = self.edge_channel_reducer(edge_attr)
        edge_attr = base_networks.relu(edge_attr + edge_attr_res)

        x_res = self.node_model(x, edge_index, edge_attr)
        if self.node_channel_reducer is not None:
            x = self.node_channel_reducer(x)
        x = base_networks.relu(x + x_res)

        return x, edge_attr

    def __repr__(self):
        return ('{}(\n'
                '    edge_model={},\n'
                '    node_model={},\n'
                ')').format(self.__class__.__name__, self.edge_model,
                            self.node_model)


class NodeEdgeAggregationModel(torch.nn.Module):
    """A layer that aggregates node features and edge features, and outputs a vector."""

    def __init__(self, config):
        super(NodeEdgeAggregationModel, self).__init__()

        input_channels = config['input_dim']
        hidden_size_1 = config['hidden'][0]
        hidden_size_2 = config['hidden'][1]
        output_dim = config['output_dim']

        layers = [
            Linear(config['input_dim'], hidden_size_1),
            groupnorm_option(hidden_size_1, config['use_groupnorm']),
            ReLU(), 
            Linear(hidden_size_1, hidden_size_2),
            groupnorm_option(hidden_size_2, config['use_groupnorm']),
            ReLU(),
            Linear(hidden_size_2, output_dim),
        ]
        if config['final_relu']:
            layers.append(ReLU())
        self.global_mlp = Seq(*layers)


    def forward(self, x, edge_index, edge_attr, batch):
        """Forward function.

        Args:
            x: [N, node_output_channels], where N is the #nodes.
            edge_index: [2, E] with max entry N - 1, where E is #edges.
            edge_attr: [E, edge_output_channels]
            batch: [N] with max entry B - 1, where B = #graphs.

        Returns:
            a [B, output_dim] torch tensor
        """
        B = batch.max()+1

        row, col = edge_index
        edge_batch = batch[row]  # [E]. edge_batch is same as batch in EdgeModel.forward()

        per_batch_edge_aggregations = scatter_mean(edge_attr, edge_batch, dim=0, dim_size=B)  # [B, edge_output_channels]
        per_batch_node_aggregations = scatter_mean(x, batch, dim=0, dim_size=B)  # [B, node_output_channels]

        out = torch.cat([per_batch_node_aggregations, per_batch_edge_aggregations], dim=1)  # [B, (node_output_channels + edge_output_channels)]
        return self.global_mlp(out)  # [B, output_dim]


class SGSNet(torch.nn.Module):

    def __init__(self, config):
        """Initialization of SGS-Net.

        Args:
            layer_config: a list of dictionaries specifying GraphNet layers
        """
        super(SGSNet, self).__init__()
        self.config = config
        self.build_network()

    def build_network(self):

        ### Node Encoder + fusion ###
        self.node_encoder = base_networks.NodeEncoder(self.config['node_encoder_config'])

        ### GraphNet Layers ###
        self.gn_layers = nn.ModuleDict()
        for i, layer_config in enumerate(self.config['layer_config']):
            self.gn_layers[f'gn_layer_{i}'] = ResidualMetaLayer(layer_config)
        self.gn_output_layer = NodeEdgeAggregationModel(self.config['gn_output_layer'])

    def forward(self, graph, other_args):
        """Forward pass of SGSNet.

        Args:
            graph: a torch_geometric.data.Batch instance with attributes:
                     - rgb: a [N x C_app] torch.FloatTensor of rgb features
                     - depth: a [N x 3 x H' x W'] torch.FloatTensor
                     - mask: a [N x 1 x H' x W'] torch.FloatTensor
                     - orig_masks: a [N x H x W] torch.FloatTensor of original masks
                     - crop_indices: a [N, 4] torch.LongTensor. xmin, ymin, xmax, ymax.
            other_args: a Python dictionary with the following keys:
                            - rgb_img_features : output of gc.extract_rgb_img_features
                            - xyz_img : a [3 x H x W] torch.FloatTensor
                            - padding_config (optional): a Python dictionary

        Returns:
            a [B] torch.Tensor of logits.
        """
        B = graph.batch.max()+1

        # Run Node encoder + fusion module to compute initial node/edge features
        node_features = self.node_encoder(graph) # Shape: [N x C_node]

        if graph.edge_index.shape[1] > 0:
            padding_config = other_args['padding_config'] if 'padding_config' in other_args else None
            edge_graph = gc.get_edge_graph(graph, other_args['rgb_img_features'],
                                           other_args['xyz_img'],
                                           padding_config=padding_config)
            edge_features = self.node_encoder(edge_graph)
        else:
            edge_input_channels = self.config['node_encoder_config']['fusion_module_config']['output_dim']
            edge_features = torch.zeros((0, edge_input_channels), dtype=torch.float32, device=constants.DEVICE)

        # import ipdb; ipdb.set_trace()

        # Run SGSNet layers
        for key in self.gn_layers:
            node_features, edge_features = self.gn_layers[key](node_features, graph.edge_index, edge_features)
        outputs = self.gn_output_layer(node_features, graph.edge_index, edge_features, graph.batch)

        return outputs[:,0]


class SGSNetWrapper(base_networks.NetworkWrapper):

    def setup(self):
        """Setup model, losses, optimizers, misc."""
        self.model = SGSNet(self.config)
        self.model.to(self.device)
        
    def run_on_graphs(self, graph_list, other_args):
        """Run algorithm on list of graphs in batches.
        
        Args:
            graph_list: a list of torch_geometric.Data instances. Length B.
            other_args: a Python dictionary with the following keys:
                            - rgb_img_features : output of gc.extract_rgb_img_features
                            - xyz_img : a [3 x H x W] torch.FloatTensor

        Returns:
            a [B] torch.FloatTensor with values in [0, 1].
        """
        def fit_batchsize_end_index(cumsum, start_ind):
            cumsum_lessthan_batchsize = np.where(cumsum < self.config['inference_edge_batch_size'])[0]
            if np.any(cumsum_lessthan_batchsize):
                end_ind = start_ind + cumsum_lessthan_batchsize[-1] + 1
            else:
                end_ind = start_ind + 1  # At least 1 graph will be evaluated...
            return end_ind

        self.eval_mode()
        with torch.no_grad():
            
            scores = torch.zeros(len(graph_list), dtype=torch.float, device=constants.DEVICE)
            
            i = 0
            while i < len(graph_list):

                # Get batch based on number of edges
                cumsum = np.cumsum([x.edge_index.shape[1] for x in graph_list[i:]])
                next_i = fit_batchsize_end_index(cumsum, i)
                batch_graph = gc.convert_list_to_batch(graph_list[i:next_i]).to(constants.DEVICE)

                # Run SGSNet
                out = self.model(batch_graph, other_args=other_args)  # [B]
                scores[i : next_i] = torch.sigmoid(out)
            
                i = next_i

        return scores

    def compare_graphs(self, graph_list_1, graph_list_2, other_args):
        """Run algorithm on list of graphs in batches.

        Args:
            graph_list_1: a list of torch_geometric.Data instances. Length num_rows.
            graph_list_2: a list of torch_geometric.Data instances. Length num_cols.
            other_args: a Python dictionary with the following keys:
                            - rgb_img_features : output of gc.extract_rgb_img_features
                            - xyz_img : a [3 x H x W] torch.FloatTensor

        Returns:
            a [num_rows, num_cols] np.ndarray of booleans of whether graph_2 is better than graph_1.
        """
        self.eval_mode()

        num_rows = len(graph_list_1)
        all_graphs = graph_list_1 + graph_list_2
        all_scores = self.run_on_graphs(all_graphs, other_args).cpu().numpy()
        graph_list_1_scores = all_scores[:num_rows]
        graph_list_2_scores = all_scores[num_rows:]

        # Hack: store the scores
        for i in range(len(graph_list_1)):
            graph_list_1[i].sgs_net_score = graph_list_1_scores[i]
        for i in range(len(graph_list_2)):
            graph_list_2[i].sgs_net_score = graph_list_2_scores[i]

        return graph_list_2_scores[None, :] > graph_list_1_scores[:, None]



