from sklearn import cluster as cl
from tifffile import TiffFile


# with TiffFile('./data/cb_raw_image_103_cubert_4.tif') as tiff:
#     X = tiff.asarray().reshape((16384, 106))
#     print(X)
    # clusters = cl.k_means(X, n_clusters=4)
    # print(clusters)

X = [
    [[1, 2, 3],
     [4, 5, 6],
     [7, 8, 9]],
    [[1, 2, 3],
     [4, 5, 6],
     [7, 8, 9]],
     ]