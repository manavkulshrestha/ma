import datetime
import numpy as np
import torch_geometric


def signed_rad(rad):
    rad = rad % (2*np.pi)

    if rad > np.pi:
        return rad - 2*np.pi
    
    return rad


def main():
    pass


if __name__ == '__main__':
    main()

    # pip install torch-scatter -f https://data.pyg.org/whl/torch-2.1.0+cu118.html

    # pip install torch-geometric torch-sparse torch-scatter torch-cluster -f https://pytorch-geometric.com/whl/torch-2.1.0+cu121.html
    # pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-2.1.0+cu121.html
    # pip uninstall torch-scatter torch-cluster torch-sparse torch-spline-conv