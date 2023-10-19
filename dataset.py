import torch
from torch_geometric.data import InMemoryDataset, Data
from tqdm import tqdm
import numpy as np
from pathlib import Path
from scipy.sparse import coo_matrix
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T

from utility import sliding, load_pkl

SERIES_PATH = Path('/home/mk/Desktop/ma/data/23-09-26-17474581')

def _get_nodes(series, include_robots=True):

    humans = np.array([[*x['pos'], x['vel']] for x in series['humans']])

    if include_robots:
        robots = np.array([[*x['pos'], x['vel']] for x in series['robots']])
        return np.vstack([robots, humans])
    
    return humans

# def get_stepgraph(prev_nodes, curr_nodes, num_robots, num_humans):
#     ''' assumes nodes are ordered as robots first and then humans'''
#     assert num_robots+num_humans == len(prev_nodes) == len(curr_nodes)

#     spatial_adj = np.ones((len(prev_nodes), len(curr_nodes))) - np.eye(num_robots)
#     temporal_adj = block_diag(np.eye(num_robots), np.zeros((num_humans, num_humans)))

#     nodes = np.vstack([prev_nodes, curr_nodes])
#     edges = np.vstack([np.hstack([spatial_adj, temporal_adj]), np.hstack([temporal_adj, spatial_adj])])

#     return nodes, edges

def _get_pairgraph(prev_nodes, curr_nodes):
    assert len(prev_nodes) != len(curr_nodes)

    curr_eye = np.eye(len(curr_nodes))
    
    prev_adj = np.ones([len(prev_nodes)]*2) - np.eye(len(prev_nodes))
    curr_adj = np.ones([len(curr_nodes)]*2) - curr_eye
    prevcurr_adj = np.vstack([np.zeros([len(curr_nodes)]*2), curr_eye])

    nodes = np.vstack([prev_nodes, curr_nodes])
    adj = np.vstack([
        np.hstack([prev_adj, prevcurr_adj]),
        np.hstack([prevcurr_adj.T, curr_adj]),
    ])

    return nodes, adj


class SeriesDataset(InMemoryDataset):
    def __init__(self, root, *, chunk,
                 transform=None, pre_transform=None, pre_filter=None):
        self.chunk = chunk
        self.root = Path(root)

        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return [f'data_series_{self.chunk[0]}-{self.chunk[1]}.pt']

    def process_files(self):
        data_list = []
        
        start, end = self.chunk
        paths = sorted(self.root.iterdir())[start:end]
        for path in tqdm(paths, desc=f'Processing'):
            file_data = load_pkl(path)

            for prev, curr in sliding(file_data['timesteps'], 2):
                # get nodes for prev robots and humans and curr humans
                nodes, adj_mat = _get_pairgraph(_get_nodes(prev), _get_nodes(curr, include_robots=False))
                nodes = torch.tensor(nodes)
                
                # create coo representation for edges
                adj_mat = coo_matrix(adj_mat)
                edges = torch.tensor(np.vstack([adj_mat.row, adj_mat.col]))
                
                # get action angles and scale to [0, 1]
                # TODO include displacement magnitude in training
                actions = torch.tensor([x['action'][0] for x in prev['robots']])
                actions = (actions+np.pi)/(2*np.pi)
                
                data = Data(x=nodes, edge_index=edges, y=actions)
                data_list.append(data)

        return data_list

    def process(self):
        data_list = self.process_files()
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


def series_dloaders(chunks=((0, 8000), (8000, 9000), (9000, 10000)),
                    batch_sizes=(64, 64, 1),
                    shuffles=(True, True, False)):
    transform = T.Compose(
        # T.NormalizeFeatures(),
        T.ToDevice('cuda'),
    )

    datasets = [SeriesDataset(SERIES_PATH, chunk=c, transform=transform) for c in chunks]
    loaders = [DataLoader(ds, batch_size=bs, shuffle=s, num_workers=8) for ds, bs, s in zip(datasets, batch_sizes, shuffles)]

    return loaders