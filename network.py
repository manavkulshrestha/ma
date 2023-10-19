from torch_geometric.nn import GATv2Conv
from torch.nn import Module


class ActionNet(Module):
    def __init__(self, heads=32, concat=False):
        super().__init__()
        self.conv1 = GATv2Conv(4, 128, heads=heads, concat=False)
        self.conv2 = GATv2Conv(256, 512, heads=heads, concat=False)
        self.conv3 = GATv2Conv(512, 1024, heads=heads, concat=False)
        self.conv4 = GATv2Conv(1024, 512, heads=heads, concat=False)
        self.conv5 = GATv2Conv(512, 256, heads=heads, concat=False)
        self.conv6 = GATv2Conv(256, 128, heads=heads, concat=False)
        self.conv7 = GATv2Conv(128, 1, heads=heads, concat=False)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = self.conv3(x, edge_index).relu()
        x = self.conv4(x, edge_index).relu()
        x = self.conv5(x, edge_index).relu()
        x = self.conv6(x, edge_index).relu()
        x = self.conv7(x, edge_index).sigmoid()

        return x

# x = F.dropout(x, p=0.5, training=self.training)
