import time
import torch
from torch_geometric.data import InMemoryDataset, Data, HeteroData
from tqdm import tqdm
import numpy as np
from pathlib import Path
from scipy.sparse import coo_matrix
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T

from utility import sliding, load_pkl

SERIES_PATH = Path('/home/mk/Desktop/ma/data/23-09-26-17474581')

def get_nodes(timestep, key):
    return torch.tensor(np.array([[*x['pos'], *x['vel']] for x in timestep[key]])).float()

def fully_connected(num_nodes, self_loops=False):
    return fully_interconnected(num_nodes, num_nodes, self_loops=self_loops)

def fully_interconnected(num_nodes1, num_nodes2, self_loops=True):
    ret = np.ones([num_nodes1, num_nodes2])
    if not self_loops:
        ret -= np.eye(num_nodes1)

    ret = coo_matrix(ret)
    return torch.tensor(np.vstack([ret.row, ret.col])).long()

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
                # get action angles and scale to [0, 1]
                # TODO include displacement magnitude in training
                actions = torch.tensor([x['action'][0] for x in prev['robots']])
                actions = (actions+np.pi)/(2*np.pi)

                data = HeteroData()
                data['prev_robots'].y = actions
                
                # nodes for robots and humans
                data['prev_robots'].x = get_nodes(prev, 'robots')
                data['prev_humans'].x = get_nodes(prev, 'humans')
                data['curr_humans'].x = get_nodes(curr, 'humans')

                # connections in previous timestep
                data['prev_robots', 'herd', 'prev_humans'].edge_index = fully_interconnected(num_robots, num_humans)
                data['prev_robots', 'communicate', 'prev_robots'].edge_index = fully_connected(num_robots)
                data['prev_humans', 'interact', 'prev_humans'].edge_index = fully_connected(num_humans)

                # connections in current timestep
                data['curr_humans', 'interact', 'curr_humans'].edge_index = fully_connected(num_humans)

                # connections for humans across time
                data['prev_humans', 'move', 'curr_humans'].edge_index = torch.tensor(np.tile(np.arange(num_humans), [2, 1])).long()

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


def merge_dloaders(*dls):
    for dl in dls:
        for d in dl:
            yield d

def series_dloaders(chunks=((0, 2500), (2500, 5000), (5000, 7500), (7500, 10000)),
                    batch_sizes=(1, 1, 1, 1),
                    shuffles=(True, True, True, False),
                    return_metadata=False):
    # TODO add mask arg for merging
    # transform = T.Compose(
    #     # T.NormalizeFeatures(),
    #     # T.ToDevice('cuda')
    # )
    transform = None

    datasets = [SeriesDataset(SERIES_PATH, chunk=c, transform=transform) for c in chunks]
    loaders = [DataLoader(ds, batch_size=bs, shuffle=s, num_workers=8) for ds, bs, s in zip(datasets, batch_sizes, shuffles)]

    ret = merge_dloaders(*loaders[:3]), loaders[3]
    return (ret, datasets[0][0].metadata()) if return_metadata else ret