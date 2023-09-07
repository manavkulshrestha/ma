import numpy as np


def signed_rad(rad):
    rad = rad % (2*np.pi)

    if rad > np.pi:
        return rad - 2*np.pi
    
    return rad


def main():
    print(signed_rad(1.9*np.pi))


if __name__ == '__main__':
    main()