import torch
import torch.nn as nn
import torch_geometric
from torch_geometric.nn import CGConv
from torch_geometric.utils import to_dense_batch
from torch_geometric.data import Batch
from torch_geometric.nn import radius_graph

from typing import List

from .utils import obtain_indices, MSG_DIRECTIONS, MSG_DIRECTION_TYPE, DEFAULT_MSG_DIRECTION, DEFAULT_VNODE_Z

class CGCNN(nn.Module):
    """
    Graph Embedding with automatic periodic boundary condition handling
    Accepts List[Data] as input
    """
    def __init__(
        self, 
        atom_fea_len, 
        nbr_fea_len, 
        n_conv, 
        radius,
        max_num_neighbors,
        gdf_var,
        VNODE_Z: int = DEFAULT_VNODE_Z,
        msg_direction: MSG_DIRECTION_TYPE = DEFAULT_MSG_DIRECTION
    ):
        super().__init__()
        self.atom_fea_len = atom_fea_len
        self.nbr_fea_len = nbr_fea_len
        self.radius = radius
        self.max_num_neighbors = max_num_neighbors

        self.VNODE_Z = VNODE_Z
        self.msg_direction = msg_direction
        assert self.msg_direction in MSG_DIRECTIONS, f"Unsupported MSG DIRECTION {self.msg_direction}"
        
        # Embedding layer
        self.embedding = nn.Embedding(125, atom_fea_len)
        
        # Convolutional layers
        self.convs = nn.ModuleList([
            CGConv(
                channels=atom_fea_len,
                dim=nbr_fea_len,
                batch_norm=True,
                aggr='add'
            ) for _ in range(n_conv)
        ])
        
        # Gaussian expansion
        self.register_buffer('gaussian_filters', 
            torch.linspace(0, radius, nbr_fea_len)
        )
        self.var = gdf_var

    def _get_gaussian_distance(self, distances):
        """Convert distances to Gaussian-expanded features"""
        return torch.exp(
            -((distances.unsqueeze(-1) - self.gaussian_filters) ** 2 / self.var ** 2)
        )

    def forward(
        self,
        data: torch_geometric.data.Data,
        h: torch.Tensor
    ):
        # Get atom features
        atom_num = data.atomic_numbers.long()
        atom_fea = self.embedding(atom_num)

        if h is not None:
            atom_fea = h
        
        # Generate edges with PBC handling
        edge_index, edge_dist = self._get_edges(data)
        edge_attr = self._get_gaussian_distance(edge_dist)
        
        # Apply convolutions
        for conv in self.convs:
            atom_fea = conv(atom_fea, edge_index, edge_attr)

        z = data.atomic_numbers
        batch = data.batch

        vn_indices = z == self.VNODE_Z
        pn_indices = z != self.VNODE_Z

        return {
            "feat": atom_fea,
            "batch": batch,
            # "vn_pred": vn_pred,
            "vn_feat": atom_fea[vn_indices],
            "vn_batch": batch[vn_indices],
            # "pn_pred": pn_pred,
            "pn_feat": atom_fea[pn_indices],
            "pn_batch": batch[pn_indices],
        }

    def _get_edges(self, data):
        # Generate edges and distances
        edge_index = radius_graph(
            data.pos,
            r=self.radius,
            batch=data.batch,
            max_num_neighbors=self.max_num_neighbors
        )
        
        # Calculate distances
        row, col = edge_index
        edge_dist = torch.norm(data.pos[row] - data.pos[col], dim=1)

        indices = obtain_indices(data, edge_index, self.VNODE_Z, self.msg_direction)
        edge_index = edge_index[:, indices]
        edge_dist = edge_dist[indices]

        return edge_index, edge_dist
