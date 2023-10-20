import torch
from sklearn.metrics import mean_squared_error as mse
from tqdm import tqdm
from torch.nn import MSELoss
import numpy as np

from dataset import series_dloaders
from network import ActionNet
from utility import ModelManager


def train_epoch(model, dloader, *, opt, epoch, loss_fn, progress=False):
    model.train()
    train_loss = 0
    total_examples = 0

    progress = tqdm if progress else lambda x, **kwargs: x
    for batch in progress(dloader, desc=f'[Epoch {epoch:03d}] training'):
        batch = batch.cuda()
        x, edge_idx, y = batch.x, batch.edge_index, batch.y 
        opt.zero_grad()

        out = model(x.float(), edge_idx)

        # get loss and update model
        batch_loss = loss_fn(out[len(y)], y)
        batch_loss.backward()
        opt.step()
        train_loss += batch_loss.item() * batch.num_graphs
        total_examples += batch.num_graphs

    return train_loss/total_examples

@torch.no_grad()
def test_epoch(model, dloader, *, epoch, progress=False):
    model.eval()
    scores = []

    progress = tqdm if progress else lambda x, **kwargs: x
    for batch in progress(dloader, desc=f'[Epoch {epoch:03d}] testing'):
        batch = batch.cuda()
        x, edge_idx, y = batch.x, batch.edge_index, batch.y 

        out = model(x.float(), edge_idx)

        score = mse(out[len(y)].cpu().numpy(), y.cpu().numpy())
        scores.append(score)

    return np.mean(scores)

def main():
    train_loader, test_loader = series_dloaders()
    model = ActionNet(heads=32, concat=False).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    modelm = ModelManager(ActionNet, 'action', save_every=10, save_best=True, initial_score=np.inf)

    for epoch in range(1, 10000):
        train_loss = train_epoch(model, train_loader, opt=optimizer, epoch=epoch, loss_fn=MSELoss())
        test_mse = test_epoch(model, test_loader, epoch=epoch)

        print(f'[Epoch {epoch:03d}] Train Loss: {train_loss:.4f}, Test MSE: {test_mse}')
        modelm.saves(model, epoch, test_mse)


if __name__ == '__main__':
    main()