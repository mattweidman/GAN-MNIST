import struct
import numpy as np
import scipy.misc

def getLabels():
    with open('train-labels-idx1-ubyte', 'rb') as f:
        magic, num = struct.unpack(">II", f.read(8))
        labels = np.fromfile(f, dtype=np.uint8)
        return labels

def getImages():
    with open('train-images-idx3-ubyte', 'rb') as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        imgs = np.fromfile(f, dtype=np.uint8).reshape(num, rows, cols)
        return imgs

