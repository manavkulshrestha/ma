from torch_geometric.data import InMemoryDataset, Data
import numpy as np
import torch, os

class ObjectPointCloudDataset(InMemoryDataset):

    def __init__(self, 
                 root: str | None = None,  
                 chunk = (1,1000), 
                 transform = None,  
                 pre_transform = None,  
                 pre_filter = None,  
                 sample_count: int = 1000, 
                 log: bool = True, 
                 output_name: str = None,
                 qt_file: str = None):
        
        self.chunk = chunk
        self.sample_count = sample_count
        self.output_name = output_name
        self.qt_file = qt_file

        super().__init__(root, transform, pre_transform, pre_filter, log)
        if qt_file is not None:
            self.data, self.slices = torch.load(qt_file)
        else:
            self.data, self.slices = torch.load(self.processed_paths[0])
        print('done loading')



    @property
    def processed_file_names(self) -> str:
        if self.output_name is not None:
            return [f'{self.output_name}.pt']
        return [f'cdata_{self.chunk[0]}-{self.chunk[1]}.pt']
    


    def process_files(self):        
        data_entries = []

        files = os.listdir(self.root)
        files.remove('processed')

        for i in range(self.chunk[0], self.chunk[1]):
            # Load point cloud
            pc = np.load(os.path.join(self.root, files[i]))

            pc = pc[np.random.choice(pc.shape[0], self.sample_count)]

            # Obtaining id from filename
            id = int(files[i][:3])

            # Create data entry
            data_entries.append(Data(x = torch.tensor(pc, dtype=torch.float), 
                                     y = torch.tensor(id, dtype=torch.LongTensor)))

        return data_entries



    def process(self):
        data_list = self.process_files()
        # print('datalist done')
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        # print('done with prefilter')

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        # print('done with pretransform')
        data, slices = self.collate(data_list)
        # print('done collating')
        torch.save((data, slices), self.processed_paths[0])
        print('done saving')



    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.out_channels})'
