import torch
from torch_geometric.data import Data
from torch import Tensor
from typing import Literal

DEFAULT_VNODE_Z = 120
MSG_DIRECTIONS = ['p2p', 'p2v', 'p2pv', 'v2v']
DEFAULT_MSG_DIRECTION = 'p2p'
MSG_DIRECTION_TYPE = Literal['p2p', 'p2v', 'p2pv', 'v2v']

def obtain_indices(data: Data, edge_index: Tensor, VNODE_Z: int, msg_direction: MSG_DIRECTION_TYPE):

    x = data.atomic_numbers

    if msg_direction == "p2p":
        indices = torch.where(
            torch.logical_and(
                x[edge_index[0]] != VNODE_Z, x[edge_index[1]] != VNODE_Z
            )
        )[0]
    elif msg_direction == "p2v":
        indices = torch.where(
            torch.logical_and(
                x[edge_index[0]] != VNODE_Z, x[edge_index[1]] == VNODE_Z
            )
        )[0]
    elif msg_direction == "v2v":
        indices = torch.where(
            torch.logical_and(
                x[edge_index[0]] == VNODE_Z, x[edge_index[1]] == VNODE_Z
            )
        )[0]
    elif msg_direction == "p2pv":
        indices = torch.where(x[edge_index[0]] != VNODE_Z)[0]

    return indices