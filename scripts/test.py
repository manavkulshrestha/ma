from _datasets import ObjectPointCloudDataset
import torch
from torch.utils.data import DataLoader, Dataset

if __name__ == '__main__':

    model_net_dataset = torch.load('../../data/ModelNet10/processed/training.pt')
    dataset_load = torch.load('train.pt')


    # Load dataset
    test_loader = ObjectPointCloudDataset(root = '.', 
                                      chunk = (87984, 109980), 
                                      sample_count = 512,
                                      output_name = 'test')
    
    train_loader = DataLoader(test_loader, batch_size=32, shuffle=True,
                              num_workers=6)

    print(model_net_dataset)
    
    for data in test_loader:
        data = data.to('cpu')
        print(data)
        break