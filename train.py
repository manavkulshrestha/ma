import torch
from sklearn.metrics import mean_squared_error as mse
from tqdm import tqdm
from torch.nn import MSELoss
import numpy as np

from dataset import series_dloaders
from network import ActionNet
from utility import save_model


def train_epoch(model, dloader, *, opt, epoch, loss_fn):
    model.train()
    train_loss = 0
    total_examples = 0

    progress = tqdm if progress else lambda x, **kwargs: x
    for batch in progress(dloader, desc=f'[Epoch {epoch:03d}] training'):
        batch = batch.cuda()
        x, edge_idx, y = batch.x, batch.edge_index, batch.y 
        opt.zero_grad()

        out = model(x, edge_idx)

        # get loss and update model
        batch_loss = loss_fn(out[len(y)], y)
        batch_loss.backward()
        opt.step()
        train_loss += batch_loss.item() * batch.num_graphs
        total_examples += batch.num_graphs

    return train_loss/total_examples

@torch.no_grad()
def test_epoch(model, dloader, *, epoch):
    model.eval()
    scores = []

    progress = tqdm if progress else lambda x, **kwargs: x
    for batch in progress(dloader, desc=f'[Epoch {epoch:03d}] testing'):
        batch = batch.cuda()
        x, edge_idx, y = batch.x, batch.edge_index, batch.y 

        out = model(x, edge_idx)

        score = mse(out[len(y)].cpu().numpy(), y.cpu().numpy())
        scores.append(score)

    return np.mean(scores)

def main():
    train_loader, test_loader, _ = series_dloaders()
    model = ActionNet(511, 256, 128, heads=32).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    best_val_acc = 0
    for epoch in range(1, 10000):
        train_loss = train_epoch(model, train_loader, opt=optimizer, epoch=epoch, loss=MSELoss())
        test_mse = test_epoch(model, test_loader, epoch=epoch)

        print(f'[Epoch {epoch:03d}] Train Loss: {train_loss:.4f}, Test MSE: {train_loss}')

        # save best model based on validation f1
        if best_val_acc < train_loss:
            best_val_acc = train_loss
            save_model(model, 'action_best_model.pt')

        # save a model every 20 epochs
        if epoch % 20 == 0:
            save_model(model, f'action_best_model-{epoch}.pt')


if __name__ == '__main__':
    main()