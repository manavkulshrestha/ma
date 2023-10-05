import torch
from torch_geometric.data import InMemoryDataset, Data
from tqdm import tqdm
import numpy as np
from pathlib import Path
from scipy.sparse import coo_matrix
from scipy.linalg import block_diag

from utility import sliding, load_pkl


def get_nodes(series):
    robots = np.array([[*x['pos'], x['vel']] for x in series['robots']])
    humans = np.array([[*x['pos'], x['vel']] for x in series['humans']])

    return np.vstack([robots, humans])

def get_stepgraph(prev_nodes, curr_nodes, num_robots, num_humans):
    ''' assumes nodes are ordered as robots first and then humans'''
    assert num_robots+num_humans == len(prev_nodes) == len(curr_nodes)

    spatial_adj = np.ones((len(prev_nodes), len(curr_nodes))) - np.eye(num_robots)
    temporal_adj = block_diag(np.eye(num_robots), np.zeros((num_humans, num_humans)))

    nodes = np.vstack([prev_nodes, curr_nodes])
    edges = np.vstack([np.hstack([spatial_adj, temporal_adj]), np.hstack([temporal_adj, spatial_adj])])

    return nodes, edges

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
            series = file_data['timesteps']
            num_robots = file_data['num_robots']
            num_humans = file_data['num_humans']

            for prev, curr in sliding(series, 2):

                nodes, adj_mat = get_stepgraph(get_nodes(prev), get_nodes(curr))
                nodes = torch.tensor(nodes)
                
                adj_mat = coo_matrix(adj_mat)
                edges = torch.tensor(np.vstack([adj_mat.row, adj_mat.col]))
                
                actions = torch.tensor([x['action'] for x in prev['robots']])
                
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