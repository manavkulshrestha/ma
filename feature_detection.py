import os.path as osp

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.datasets import ModelNet
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MLP, PointNetConv, fps, global_max_pool, radius
from torch_geometric.typing import WITH_TORCH_CLUSTER
from _datasets import ObjectPointCloudDataset

if not WITH_TORCH_CLUSTER:
    print("This example requires 'torch-cluster'")


# Overall PointNet2 does the following
# For a multitude of set abstraction levels:
#   1. Sample a subset of points (Sampling Layer) by extracting centroid of local regions
#   2. For each point in the subset, find the k nearest neighbors (Grouping Layer)
#   3. For each point in the subset, extract features from the k nearest neighbors (PointNet Layer)


# 
class SAModule(torch.nn.Module):
    def __init__(self, ratio, r, nn):
        super().__init__()
        self.ratio = ratio
        self.r = r
        self.conv = PointNetConv(nn, add_self_loops=False)

    def forward(self, x, pos, batch):
        # Using iterative farthest point sampling to select a subset of points
        idx = fps(pos, batch, ratio=self.ratio)
        # For each point in the subset, find the k nearest neighbors to a distance r
        # Points found can be split into two groups: row and col
        row, col = radius(pos, pos[idx], self.r, batch, batch[idx],
                          max_num_neighbors=64)
        # Joining the groups together to form a graph
        edge_index = torch.stack([col, row], dim=0)
        
        x_dst = None if x is None else x[idx]

        # Extracting features from two previous layers to generate a sampled point cloud
        x = self.conv((x, x_dst), (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch



# 
class GlobalSAModule(torch.nn.Module):
    def __init__(self, nn):
        super().__init__()
        # Instantiates nn module (Which one?)
        self.nn = nn

    def forward(self, x, pos, batch):
        # Calling nn module on the concatenated features of the sampled point cloud
        x = self.nn(torch.cat([x, pos], dim=1))
        x = global_max_pool(x, batch)
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch



# Point net layer of the set abstraction?
# TODO: Check if this is correct
class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # Input channels account for both `pos` and node features.
        # First layer of set abstraction instantiates a SAModule
        self.sa1_module = SAModule(0.5, 0.2, MLP([3, 64, 64, 128]))

        # Second layer of set abstraction instantiates a SAModule
        self.sa2_module = SAModule(0.25, 0.4, MLP([128 + 3, 128, 128, 256]))

        # Third layer of set abstraction instantiates a GlobalSAModule
        self.sa3_module = GlobalSAModule(MLP([256 + 3, 256, 512, 1024]))

        # Instantiates MLP module to obtain one leayer before output
        # Changed last layer for our labels
        self.mlp = MLP([1024, 512, 256, 78], dropout=0.5, norm=None)

    def forward(self, data):
        sa0_out = (None, data.x, data.batch)
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)
        x, pos, batch = sa3_out

        result, obj_features = self.mlp(x, return_emb=True)

        return result.log_softmax(dim=-1), obj_features



# ############################################################

# THIS SEEMS FINE AS A STANDARD TRAINING AND TESTING FUNCTIONS

def train(epoch):
    # Set model to training mode
    model.train()

    # Iterate over the training batches
    for data in train_loader:
        # Allocate data to device
        data = data.to(device)
        # Zero out gradients
        optimizer.zero_grad()
        # Compute loss
        loss = F.nll_loss(model(data)[0], data.y.type(torch.LongTensor).to(device))
        # Compute gradients
        loss.backward()
        # Update parameters
        optimizer.step()



def test(loader):
    # Set model to evaluation mode
    model.eval()

    correct = 0
    # Iterate over the testing batches
    for data in loader:
        # Allocate data to device
        data = data.to(device)
        # Compute predictions
        with torch.no_grad():
            pred = model(data)[0].max(1)[1]
        # Compute accuracy
        correct += pred.eq(data.y).sum().item()

    # Return accuracy
    return correct / len(loader.dataset)



#  ########################### main ###########################
if __name__ == '__main__':
    # Getting the data off files (in our case npy files)
    # path = osp.join(osp.dirname(osp.realpath(__file__)), '..',
    #                 'data/ModelNet10')
    
    # ############################################################
    # THIS DOES NOT DIRECTLY APPLY TO US YET
    # ############################################################

    # # Preprocessing functions for transforming the data
    # pre_transform, transform = T.NormalizeScale(), T.SamplePoints(1024)

    # # Instantiating the dataset with the InMemoryDataset class
    # train_dataset = ModelNet(path, '10', True, transform, pre_transform)
    # test_dataset = ModelNet(path, '10', False, transform, pre_transform)

    # ############################################################
    # THIS DOES APPLY TO US
    # ############################################################

    train_dataset = ObjectPointCloudDataset(root = '../dataset/v4', 
                                      chunk = (0, 87984), 
                                      sample_count = 512,
                                      output_name = 'trainv9'
                                      )
    
    test_dataset = ObjectPointCloudDataset(root = '../dataset/v4', 
                                      chunk = (87984, 109980), 
                                      sample_count = 512,
                                      output_name = 'testv9'
                                      )

    # Create intances of dataloaders for training and testing
    # TODO: Check if batch size is correct or test with different ones
    # TODO: Check if num_workers is correct or test with different ones
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,
                              num_workers=8)
    
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False,
                             num_workers=8)

    # Use cuda if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    # Instantiate the model in the device
    model = Net().to(device)

    # Instantiate the optimizer (We can look for many @ https://pytorch.org/docs/stable/optim.html)
    # TODO: Check if lr is correct or test with different ones
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # # # Train the model
    # for epoch in range(1, 101):
        
    #     train(epoch)
    #     test_acc = test(test_loader)
    #     print(f'Epoch: {epoch:03d}, Test: {test_acc:.4f}')

    #     if epoch % 1 == 0:
    #         torch.save({
    #             'epoch': epoch,
    #             'model_state_dict': model.state_dict(),
    #             'optimizer_state_dict': optimizer.state_dict(),
    #             'loss': test_acc
    #         }, f'model_{epoch}_{test_acc}.pt')
