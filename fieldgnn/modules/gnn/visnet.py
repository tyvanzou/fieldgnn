from torch_geometric.nn.models.visnet import (
    # EquivariantScalar,
    GatedEquivariantBlock,
    Atomref,
    ViSNetBlock as ViSNetBlockO,
)
import torch
from typing import Optional, Tuple, Literal
from torch import Tensor
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.utils import scatter

from torch.autograd import grad

from .utils import obtain_indices, MSG_DIRECTIONS, MSG_DIRECTION_TYPE, DEFAULT_VNODE_Z, DEFAULT_MSG_DIRECTION


# NOTE we modify these module to output feature
class EquivariantScalar(torch.nn.Module):
    r"""Computes final scalar outputs based on node features and vector
    features.

    Args:
        hidden_channels (int): The number of hidden channels in the node
            embeddings.
    """

    def __init__(self, hidden_channels: int, out_channels: int = None) -> None:
        super().__init__()

        if out_channels is None:
            out_channels = hidden_channels

        self.output_network = torch.nn.ModuleList(
            [
                GatedEquivariantBlock(
                    hidden_channels,
                    (hidden_channels + out_channels) // 2,
                    scalar_activation=True,
                ),
                GatedEquivariantBlock(
                    (hidden_channels + out_channels) // 2,
                    out_channels,
                    # hidden_channels // 2,
                    # 1,
                    scalar_activation=False,
                ),
            ]
        )

        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets the parameters of the module."""
        for layer in self.output_network:
            layer.reset_parameters()

    def pre_reduce(self, x: Tensor, v: Tensor) -> Tensor:
        r"""Computes the final scalar outputs.

        Args:
            x (torch.Tensor): The scalar features of the nodes.
            v (torch.Tensor): The vector features of the nodes.

        Returns:
            out (torch.Tensor): The final scalar outputs of the nodes.
        """
        for layer in self.output_network:
            x, v = layer(x, v)

        return x + v.sum() * 0


class ViSNetBlock(ViSNetBlockO):
    def __init__(
        self,
        lmax: int = 1,
        vecnorm_type: Optional[str] = None,
        trainable_vecnorm: bool = False,
        num_heads: int = 8,
        num_layers: int = 6,
        hidden_channels: int = 128,
        num_rbf: int = 32,
        trainable_rbf: bool = False,
        max_z: int = 100,
        cutoff: float = 5.0,
        max_num_neighbors: int = 32,
        vertex: bool = False,
        VNODE_Z: int = DEFAULT_VNODE_Z,
        msg_direction: MSG_DIRECTION_TYPE = DEFAULT_MSG_DIRECTION,
    ) -> None:
        self.VNODE_Z = VNODE_Z
        self.msg_direction = msg_direction
        assert self.msg_direction in MSG_DIRECTIONS, f"Unsupported MSG DIRECTION {self.msg_direction}"

        super().__init__(
            lmax=lmax,
            vecnorm_type=vecnorm_type,
            trainable_vecnorm=trainable_vecnorm,
            num_heads=num_heads,
            num_layers=num_layers,
            hidden_channels=hidden_channels,
            num_rbf=num_rbf,
            trainable_rbf=trainable_rbf,
            max_z=max_z,
            cutoff=cutoff,
            max_num_neighbors=max_num_neighbors,
            vertex=vertex,
        )

    def forward(
        self, data: torch_geometric.data.Data, h: Tensor
    ) -> Tuple[Tensor, Tensor]:
        r"""Computes the scalar and vector features of the nodes.

        Args:
            z (torch.Tensor): The atomic numbers.
            pos (torch.Tensor): The coordinates of the atoms.
            batch (torch.Tensor): A batch vector, which assigns each node to a
                specific example.

        Returns:
            x (torch.Tensor): The scalar features of the nodes.
            vec (torch.Tensor): The vector features of the nodes.
        """
        z = data.atomic_numbers
        pos = data.pos
        batch = data.batch

        # x = self.embedding(z)
        if h is None:
            x = self.embedding(z)
        else:
            x = h

        edge_index, edge_weight, edge_vec = self.distance(pos, batch)

        # NOTE: unidirectional message passing from actual node to virtual node
        indices = obtain_indices(data, edge_index, self.VNODE_Z, self.msg_direction)

        edge_index = edge_index[:, indices]
        edge_weight = edge_weight[indices]
        edge_vec = edge_vec[indices]

        # repulsion_indices = torch.where(edge_weight > 1)[0]
        # edge_index = edge_index[:, repulsion_indices]
        # edge_weight = edge_weight[repulsion_indices]
        # edge_vec = edge_vec[repulsion_indices]


        edge_attr = self.distance_expansion(edge_weight)
        mask = edge_index[0] != edge_index[1]
        edge_vec[mask] = edge_vec[mask] / torch.norm(edge_vec[mask], dim=1).unsqueeze(1)
        edge_vec = self.sphere(edge_vec)
        x = self.neighbor_embedding(z, x, edge_index, edge_weight, edge_attr)
        vec = torch.zeros(
            x.size(0),
            ((self.lmax + 1) ** 2) - 1,
            x.size(1),
            dtype=x.dtype,
            device=x.device,
        )
        edge_attr = self.edge_embedding(edge_index, edge_attr, x)

        for attn_idx, attn in enumerate(self.vis_mp_layers[:-1]):
            dx, dvec, dedge_attr = attn(
                x, vec, edge_index, edge_weight, edge_attr, edge_vec
            )
            x = x + dx
            vec = vec + dvec
            edge_attr = edge_attr + dedge_attr
                # print(ovec[first_real_node_idx])
                # print(
                #     self.msg_direction,
                #     attn_idx,
                #     x[z != self.VNODE_Z][:5, 0],
                #     dx[z != self.VNODE_Z][:5, 0],
                #     vec[z != self.VNODE_Z][:5, 0, 0],
                #     dvec[z != self.VNODE_Z][:5, 0, 0],
                # )
            # if self.msg_direction == "p2v":
            # print(
            #     attn_idx,
            #     x[z != self.VNODE_Z][:10, :3],
            #     dx[z != self.VNODE_Z][:10, :3],
            # )
            # if self.msg_direction == "p2v":
            #     print(
            #         self.msg_direction,
            #         attn_idx,
            #         x[z == self.VNODE_Z][:5, 0],
            #         dx[z == self.VNODE_Z][:5, 0],
            #         vec[z == self.VNODE_Z][:5, 0, 0],
            #         dvec[z == self.VNODE_Z][:5, 0, 0],
            #     )

        dx, dvec, _ = self.vis_mp_layers[-1](
            x, vec, edge_index, edge_weight, edge_attr, edge_vec
        )
        x = x + dx
        vec = vec + dvec

        x = self.out_norm(x)
        vec = self.vec_out_norm(vec)

        return x, vec


class ViSNet(torch.nn.Module):
    def __init__(
        self,
        lmax: int = 1,
        vecnorm_type: Optional[str] = None,
        trainable_vecnorm: bool = False,
        num_heads: int = 8,
        num_layers: int = 6,
        hidden_channels: int = 128,
        num_rbf: int = 32,
        trainable_rbf: bool = False,
        max_z: int = 100,
        cutoff: float = 5.0,
        max_num_neighbors: int = 32,
        vertex: bool = False,
        atomref: Optional[Tensor] = None,
        reduce_op: str = "sum",
        mean: float = 0.0,
        std: float = 1.0,
        derivative: bool = False,
        VNODE_Z: int = 119,
        msg_direction: Literal["p2p", "p2v", "p2pv", "v2v"] = "p2pv",
    ) -> None:
        super().__init__()

        self.representation_model = ViSNetBlock(
            lmax=lmax,
            vecnorm_type=vecnorm_type,
            trainable_vecnorm=trainable_vecnorm,
            num_heads=num_heads,
            num_layers=num_layers,
            hidden_channels=hidden_channels,
            num_rbf=num_rbf,
            trainable_rbf=trainable_rbf,
            max_z=max_z,
            cutoff=cutoff,
            max_num_neighbors=max_num_neighbors,
            vertex=vertex,
            VNODE_Z=VNODE_Z,
            msg_direction=msg_direction,
        )

        self.output_model = EquivariantScalar(hidden_channels=hidden_channels)
        self.prior_model = Atomref(atomref=atomref, max_z=max_z)
        self.reduce_op = reduce_op
        self.derivative = derivative

        self.VNODE_Z = VNODE_Z

        self.register_buffer("mean", torch.tensor(mean))
        self.register_buffer("std", torch.tensor(std))

        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets the parameters of the module."""
        self.representation_model.reset_parameters()
        self.output_model.reset_parameters()
        if self.prior_model is not None:
            self.prior_model.reset_parameters()

    def forward(
        self,
        data: Data,
        # z: Tensor,
        # pos: Tensor,
        # batch: Tensor,
        return_force: bool = False,
        h: Tensor = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        r"""Computes the energies or properties (forces) for a batch of
        molecules.

        Args:
            z (torch.Tensor): The atomic numbers.
            pos (torch.Tensor): The coordinates of the atoms.
            batch (torch.Tensor): A batch vector,
                which assigns each node to a specific example.

        Returns:
            # y (torch.Tensor): The energies or properties for each molecule.
            # dy (torch.Tensor, optional): The negative derivative of energies.
            x (torch.Tensor): scalar feature
            v (torch.Tensor): vector feature
        """
        z = data.atomic_numbers
        pos = data.pos
        batch = data.batch

        if return_force:
            pos.requires_grad_(True)

        x, v = self.representation_model(data, h)

        vn_indices = z == self.VNODE_Z
        pn_indices = z != self.VNODE_Z

        # x = self.output_model.pre_reduce(x, v)
        # x = x * self.std

        # if self.prior_model is not None:
        #     x = self.prior_model(x, z)

        # y = scatter(x, batch, dim=0, reduce=self.reduce_op)
        # y = y + self.mean

        # if self.derivative:
        #     grad_outputs = [torch.ones_like(y)]
        #     dy = grad(
        #         [y],
        #         [pos],
        #         grad_outputs=grad_outputs,
        #         create_graph=True,
        #         retain_graph=True,
        #     )[0]
        #     if dy is None:
        #         raise RuntimeError("Autograd returned None for the force prediction.")
        #     return y, -dy

        # return y, None

        vn_pred = scatter(
            x[vn_indices], batch[vn_indices], dim=0, reduce=self.reduce_op
        )
        pn_pred = scatter(
            x[pn_indices], batch[pn_indices], dim=0, reduce=self.reduce_op
        )

        ret = {
            "feat": x,
            "vn_pred": vn_pred,
            "vn_feat": x[vn_indices],
            "vn_batch": batch[vn_indices],
            "pn_pred": pn_pred,
            "pn_feat": x[pn_indices],
            "pn_batch": batch[pn_indices],
        }

        if return_force:
            # vn_force = -grad(
            #     outputs=vn_pred,
            #     inputs=pos[vn_indices],
            #     grad_outputs=torch.ones_like(vn_pred),
            #     create_graph=True,
            #     retain_graph=True,
            # )[0]
            # ret["vn_force"] = vn_force
            pn_force = -grad(
                outputs=pn_pred,
                inputs=pos,
                grad_outputs=torch.ones_like(pn_pred),
                create_graph=True,
                retain_graph=True,
            )[0][pn_indices]
            ret["pn_force"] = pn_force

        return ret
