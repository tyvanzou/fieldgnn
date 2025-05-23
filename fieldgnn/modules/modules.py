from pathlib import Path
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch_geometric
from torch_geometric.data import Data, Batch
from torch_geometric.data.data import BaseData
from torch_geometric.data.collate import collate
from torch_scatter import scatter_mean
from einops import rearrange

from pymatgen.core.lattice import Lattice

# common modules
from fieldgnn.modules import objectives, heads

# GNNs
# GemNet
from fieldgnn.modules.jmp.models.gemnet.backbone import GemNetOCBackbone
from fieldgnn.modules.jmp.models.gemnet.config import BackboneConfig
from fieldgnn.modules.jmp.utils.goc_graph import (
    Cutoffs,
    Graph,
    MaxNeighbors,
    generate_graph,
    subselect_graph,
    tag_mask,
)

# visnet
from fieldgnn.modules.gnn import (
    ViSNet as ViSNetO,
    SchNet as SchNetO,
    DimeNetPlusPlus as DimeNetO,
    CGCNN as CGCNNO,
)

from fieldgnn.utils import Log
from fieldgnn.config import get_config, get_model_config, get_data_config, get_train_config

from abc import ABC, abstractmethod
from typing import Literal, Any, Dict, Union, List
from typing_extensions import override

import warnings

from torch.autograd import grad


MSG_DIRECTION_TYPE = Literal["p2p", "p2v", "v2v", "p2pv"]
MSG_DIRECTIONS = ["p2p", "p2v", "v2v", "p2pv"]


class BaseModule(torch.nn.Module, ABC, Log):
    def __init__(self):
        torch.nn.Module.__init__(self)
        ABC.__init__(self)
        Log.__init__(self)

        self.config = get_config()
        self.model_config = get_model_config()
        self.data_config = get_data_config()
        self.train_config = get_train_config()
        self._validate_config()
        self.num_grids = self.data_config["num_grids"]
        self.repulsion_distance = self.data_config['repulsion_distance']
        self.msg_routes = self.train_config["msg_routes"]
        self.VNODE_Z = self.train_config["VNODE_Z"]

    # we validate each part in specifc module
    def _validate_config(self):
        pass

    # process single data
    @abstractmethod
    def data_transform(self, data: BaseData) -> Any:
        contain_vn = False
        for route in self.msg_routes:
            if 'v' in route:
                contain_vn = True
                break

        if contain_vn:
            data = self.add_virtual_nodes(data)

        # if not torch.is_tensor(data.y):
        #     data.y = torch.tensor(data.y)
        # data.y = data.y.float().view(-1)
        data.atomic_numbers = data.atomic_numbers.long()
        data.natoms = len(data.atomic_numbers)
        data.pos = data.pos.float()
        data.cell = data.cell.reshape(1, 3, 3)

        return data

    # collate transformed data
    @abstractmethod
    def collate_fn(self, data_list: List[Any]) -> Any:
        # normally data_list should be list of torch_geometric.data.Data
        # we directly use torch_geometric.data.collate.collate
        return collate(data_list[0].__class__, data_list)[0]

    # process collated data to target
    @abstractmethod
    def process_batch(self, batch_data: Any):
        # forward process
        pass

    def forward(self, batch_data: Dict, **kwargs) -> torch.Tensor:
        # by default we use graph data
        graph_data = batch_data["graph"]
        batch_graph = self.process_data_list(graph_data)
        # NOTE: in-place modification!
        batch_data["collated_graph"] = batch_graph
        result = self.process_batch(batch_graph, **kwargs)
        return result

    def get_grid_poses(self, lat: Lattice) -> torch.Tensor:
        num_grids = self.num_grids
        grid_poses = []
        for i in range(num_grids[0]):
            for j in range(num_grids[1]):
                for k in range(num_grids[2]):
                    grid_pos = lat.get_cartesian_coords(
                        [i / num_grids[0], j / num_grids[1], k / num_grids[2]]
                    )
                    grid_poses.append(torch.from_numpy(grid_pos).float())
        grid_poses = torch.stack(grid_poses)  # [N, 3]
        return grid_poses

    def add_virtual_nodes(self, data: BaseData) -> BaseData:
        num_grids = self.num_grids
        VNODE_Z = self.VNODE_Z
        repulsion_distance = self.repulsion_distance

        cell = data["cell"].tolist()
        pos = data["pos"]
        X = data["atomic_numbers"]

        lattice_obj = Lattice(cell)
        grid_poses = self.get_grid_poses(lattice_obj).to(X.device)

        dist_matrix = torch.cdist(grid_poses, pos)  # [N_vnodes, N_real_nodes]
        min_distances = dist_matrix.min(dim=1).values  # [N_vnodes]
        
        mask = min_distances >= repulsion_distance
        filtered_grid_poses = grid_poses[mask]
        N_filtered_vnodes = len(filtered_grid_poses)

        pos_added = torch.concat([pos, filtered_grid_poses], dim=0)
        X_added = torch.concat(
            [
                X,
                torch.ones(
                    N_filtered_vnodes,
                    dtype=torch.long,
                    device=X.device,
                )
                * VNODE_Z,
            ],
            dim=0,
        )

        data["atomic_numbers"] = X_added
        data["pos"] = pos_added.float()
        data["cell"] = torch.tensor(cell, dtype=torch.float, device=X.device)

        return data


    # def add_virtual_nodes(self, data: BaseData) -> BaseData:
    #     num_grids = self.num_grids
    #     VNODE_Z = self.VNODE_Z

    #     cell = data["cell"].tolist()
    #     pos = data["pos"]
    #     X = data["atomic_numbers"]

    #     lattice_obj = Lattice(cell)
    #     grid_poses = self.get_grid_poses(lattice_obj).to(X.device)

    #     N_VNODES = num_grids[0] * num_grids[1] * num_grids[2]

    #     pos_added = torch.concat([pos, grid_poses], dim=0)
    #     X_added = torch.concat(
    #         [
    #             X,
    #             torch.ones(
    #                 N_VNODES,
    #                 dtype=torch.long,
    #                 device=X.device,
    #             )
    #             * VNODE_Z,
    #         ],
    #         dim=0,
    #     )

    #     data["atomic_numbers"] = X_added
    #     data["pos"] = pos_added.float()
    #     data["cell"] = torch.tensor(cell, dtype=torch.float, device=X.device)

    #     return data

    def process_data_list(self, data_list: List[BaseData]) -> Any:
        data_list = [self.data_transform(data) for data in data_list]
        batch_data = self.collate_fn(data_list)
        return batch_data


class SchNet(BaseModule):
    def __init__(self):
        super().__init__()

        self.model_config = self.model_config["schnet"]
        self.schnets = nn.ModuleDict(
            {
                msg_direction: SchNetO(
                    **self.model_config,
                    VNODE_Z=self.VNODE_Z,
                    msg_direction=msg_direction,
                )
                for msg_direction in self.msg_routes
            }
        )

    def data_transform(self, data: Data) -> Data:
        return super().data_transform(data)

    def collate_fn(self, data_list: List[Data]) -> Data:
        return super().collate_fn(data_list)

    def process_batch(self, batch_data) -> BaseData:
        h = None
        for msg_direction in self.msg_routes:
            out = self.schnets[msg_direction](
                batch_data,
                h=h,
            )
            h = out["feat"]
        return out


class CGCNN(BaseModule):
    def __init__(self):
        super().__init__()

        self.model_config = self.model_config["cgcnn"]
        self.cgcnns = nn.ModuleDict(
            {
                msg_direction: CGCNNO(
                    **self.model_config,
                    VNODE_Z=self.VNODE_Z,
                    msg_direction=msg_direction,
                )
                for msg_direction in self.msg_routes
            }
        )

    def data_transform(self, data: Data) -> Data:
        return super().data_transform(data)

    def collate_fn(self, data_list: List[Data]) -> Data:
        return super().collate_fn(data_list)

    def process_batch(self, batch_data) -> BaseData:
        h = None
        for msg_direction in self.msg_routes:
            out = self.cgcnns[msg_direction](
                batch_data,
                h=h,
            )
            h = out["feat"]
        return out


class ViSNet(BaseModule):
    def __init__(self):
        super().__init__()
        self.model_config = self.model_config["visnet"]
        # self.visnet = ViSNetO(**self.model_config, VNODE_Z=self.VNODE_Z)
        self.visnets = nn.ModuleDict(
            {
                msg_direction: ViSNetO(
                    **self.model_config,
                    VNODE_Z=self.VNODE_Z,
                    msg_direction=msg_direction,
                )
                for msg_direction in self.msg_routes
            }
        )

    def data_transform(self, data: Data) -> Data:
        return super().data_transform(data)

    def collate_fn(self, data_list: List[Data]) -> Data:
        return super().collate_fn(data_list)

    def process_batch(self, batch_data) -> BaseData:
        h = None
        for msg_direction in self.msg_routes:
            out = self.visnets[msg_direction](
                batch_data,
                h=h,
            )
            h = out["feat"]
        return out


# TODO
class DimeNet(BaseModule):
    def __init__(self):
        super().__init__()

        self.model_config = self.model_config["dimenet"]
        self.dimenets = nn.ModuleDict(
            {
                msg_direction: DimeNetO(
                    **self.model_config,
                    VNODE_Z=self.VNODE_Z,
                    msg_direction=msg_direction,
                )
                for msg_direction in self.msg_routes
            }
        )

    def data_transform(self, data: Data) -> Data:
        return super().data_transform(data)

    def collate_fn(self, data_list: List[Data]) -> Data:
        return super().collate_fn(data_list)

    def process_batch(self, batch_data) -> BaseData:
        h = None
        for msg_direction in self.msg_routes:
            out = self.dimenets[msg_direction](
                batch_data,
                h=h,
            )
            h = out["feat"]
        return out


class JMP(BaseModule):
    def __init__(self):
        super().__init__()

        assert len(self.msg_routes) == 1, f"JMP only support single stage message passing"
        assert self.msg_routes[0] in ['p2p', 'p2pv'], f'JMP only support single stage message passing of p2p or p2pv'

        self.model_config = self.model_config["jmp"]
        self.hid_dim = self.model_config["hid_dim"]

        base_config = BackboneConfig.base()
        # Plus one for Virtual Node
        self.atom_embedding = nn.Embedding(125, self.hid_dim)
        self.atom_embedding.apply(objectives.init_weights)
        # self.gemnet = nn.ModuleDict(
        #     {
        #         msg_direction: GemNetOCBackbone(base_config, **base_config)
        #         for msg_direction in self.msg_routes
        #     }
        # )
        self.gemnet = GemNetOCBackbone(base_config, **base_config)
        if self.model_config.get("ckpt") is not None:
            self.load_backbone_state_dict(self.model_config["ckpt"])

    def load_backbone_state_dict(self, ckpt_path):
        def filter_state_dict(state_dict: dict[str, torch.Tensor], prefix: str):
            return {
                k[len(prefix) :]: v
                for k, v in state_dict.items()
                if k.startswith(prefix)
            }

        # Due to that pre-trained JMP-S weight is used, the ckpt path is not configurable
        ckpt = torch.load(ckpt_path)
        state_dict = ckpt["state_dict"]
        backbone_state_dict = filter_state_dict(state_dict, "backbone.")
        # due to that we remove symmetric mp, there are some unexpected keys to ignore
        load_ret = self.gemnet.load_state_dict(backbone_state_dict, strict=False)
        print("Load Backbone Statedict Return: ", load_ret)
        # for msg_direction in self.msg_routes:
        #     load_ret = self.gemnet[msg_direction].load_state_dict(
        #         backbone_state_dict, strict=False
        #     )
        #     self.log(
        #         f"Load JMP Backbone Statedict for {msg_direction} Return: {str(load_ret)}",
        #         "info",
        #     )

    def generate_graphs(
        self,
        data: BaseData,
        cutoffs: Cutoffs,
        max_neighbors: MaxNeighbors,
        pbc: bool,
        msg_direction: MSG_DIRECTION_TYPE,
    ) -> Data:
        assert msg_direction in MSG_DIRECTIONS
        aint_graph = generate_graph(
            data,
            cutoff=cutoffs.aint,
            max_neighbors=max_neighbors.aint,
            pbc=pbc,
            VNODE_Z=self.VNODE_Z,
            msg_direction=msg_direction,
        )
        subselect = partial(
            subselect_graph,
            data,
            aint_graph,
            cutoff_orig=cutoffs.aint,
            max_neighbors_orig=max_neighbors.aint,
        )
        main_graph = subselect(cutoffs.main, max_neighbors.main)
        aeaint_graph = subselect(cutoffs.aeaint, max_neighbors.aeaint)
        qint_graph = subselect(cutoffs.qint, max_neighbors.qint)

        # We can't do this at the data level: This is because the batch collate_fn doesn't know
        # that it needs to increment the "id_swap" indices as it collates the data.
        # So we do this at the graph level (which is done in the GemNetOC `get_graphs_and_indices` method).
        # main_graph = symmetrize_edges(main_graph, num_atoms=data.pos.shape[0])
        qint_graph = tag_mask(data, qint_graph, tags=[1, 2])

        graphs = {
            "main": main_graph,
            "a2a": aint_graph,
            "a2ee2a": aeaint_graph,
            "qint": qint_graph,
        }

        for graph_type, graph in graphs.items():
            for key, value in graph.items():
                setattr(data, f"{graph_type}_{key}", value)

        return data

    # override
    def data_transform(self, data: Data) -> Data:
        # add virtual node and add some attributes
        data = super().data_transform(data)
        device = data["atomic_numbers"].device

        data.tags = 2 * torch.ones(data.natoms).to(device)
        data.tags = data.tags.long().to(device)

        data.fixed = torch.zeros(data.natoms, dtype=torch.bool).to(device)

        # # NOTE: This config is for Porous Material
        # cutoff = 8
        # max_neighbors = 10
        # if self.train_config['enable_vn']:
        #     cutoff = 19
        #     n_atoms = (
        #         data.natoms - self.num_grids[0] * self.num_grids[1] * self.num_grids[2]
        #     )
        # else:
        #     cutoff = 19
        #     n_atoms = data.natoms

        # cutoff = 19
        # n_atoms = data.natoms

        # if n_atoms > 300:
        #     max_neighbors = 5
        # elif n_atoms > 200:
        #     max_neighbors = 10
        # else:
        #     max_neighbors = 30
        cutoff = 12
        max_neighbors = 8
        # cutoff = 6
        # max_neighbors = 32
        # n_atoms = data.natoms
        # if n_atoms > 1212:
        #     max_neighbors = 5
        
        data = self.generate_graphs(
            data,
            cutoffs=Cutoffs.from_constant(cutoff),
            max_neighbors=MaxNeighbors.from_goc_base_proportions(max_neighbors),
            # cutoffs=Cutoffs.from_constant(12.0),
            # max_neighbors=MaxNeighbors.from_goc_base_proportions(30),
            # TODO: Here should be use pbc=True since crystal
            pbc=True,
            msg_direction=self.msg_routes[0]
        )

        return data

        # datas = {
        #     msg_direction: self.generate_graphs(
        #         data,
        #         cutoffs=Cutoffs.from_constant(cutoff),
        #         max_neighbors=MaxNeighbors.from_goc_base_proportions(max_neighbors),
        #         pbc=True,
        #         # pbc=False,
        #         msg_direction=msg_direction,
        #     )
        #     for msg_direction in self.msg_routes
        # }

        # return datas

    def collate_fn(self, data_list: List[Data]):
        # return {
        #     msg_direction: BaseModule.collate_fn(
        #         self, [data[msg_direction] for data in data_list]
        #     )
        #     for msg_direction in self.msg_routes
        # }
        return super().collate_fn(data_list)

    def process_batch(self, batch):
        # TODO: This can results in error when different msg_direction's batch is different.
        # x = list(batch.values())[0]["atomic_numbers"]
        # pos = list(batch.values())[0]["pos"]
        # atom_batch_idx = list(batch.values())[0]["batch"]

        x = batch["atomic_numbers"]
        pos = batch["pos"]
        atom_batch_idx = batch["batch"]

        h = self.atom_embedding(x)

        # struct
        # graph_embed = self.schnet(x, pos, atom_batch_idx)
        graph_output = self.gemnet(batch, h=h)
        # graph_embed = graph_output["h"]
        h = graph_output["energy"]

        # h = None
        # for msg_direction in self.msg_routes:
        #     if h is None:
        #         h = self.atom_embedding(x)

        #     graph_output = self.gemnet[msg_direction](batch[msg_direction], h=h)
        #     h = graph_output["energy"]

        vn_indices = x == self.VNODE_Z
        pn_indices = x != self.VNODE_Z

        # return None
        return {
            "feat": h,
            "batch": atom_batch_idx,
            "vn_feat": h[vn_indices],
            "pn_feat": h[pn_indices],
            "vn_batch": atom_batch_idx[vn_indices],
            "pn_batch": atom_batch_idx[pn_indices],
        }


TYPE_MODULE_MAP = {
    "schnet": SchNet,
    "visnet": ViSNet,
    "dimenet": DimeNet,
    "jmp": JMP,
    "cgcnn": CGCNN,
}
