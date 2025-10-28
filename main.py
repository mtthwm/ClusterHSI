from sklearn import cluster as cl
from tifffile import TiffFile
import numpy as np


# with TiffFile('./data/cb_raw_image_103_cubert_4.tif') as tiff:
#     X = tiff.asarray().reshape((16384, 106))
#     print(X)
    # clusters = cl.k_means(X, n_clusters=4)
    # print(clusters)

X = np.array([
    [[1, 2, 3],
     [4, 5, 6],
     [7, 8, 9]],
    [[1, 2, 3],
     [4, 5, 6],
     [7, 8, 9]],
    [[1, 2, 3],
     [4, 5, 6],
     [7, 8, 9]],
     ])

print(f"Original: {X.shape}\n{X}")

Xr1 = X.reshape((3, 9))
print(f"Reshaped (3,9):\n{Xr1}")

Xr2 = X.reshape((9, 3))
print(f"Reshaped (9,3):\n{Xr2}")

Xr1T = Xr1.transpose()
print(f"Xr1T Transposed:\n{Xr1T}")

# FINAL OPERATION:
X = X # Suppose X is the HSI image with shape (3, 3, 3)
pix = X.reshape(3, 9).transpose() # Three is the number of channels, while 9 is the number of spectral samples in each channel
print(pix)