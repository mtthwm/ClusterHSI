from sklearn import cluster as cl
from tifffile import TiffFile, imwrite, imread
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axis import Axis

with TiffFile('./data/cb_raw_image_103_cubert_4.tif') as tiff:
    im_data = tiff.asarray()
    n_chan = im_data.shape[0]
    im_size_x = im_data.shape[1]
    im_size_y = im_data.shape[2]

    X = im_data.reshape(n_chan, im_size_x*im_size_y).transpose()
    
    _, k_means_labels, _ = cl.k_means(X, n_clusters=4)
    # affinity_labels = cl.AffinityPropagation().fit(X).labels_
    
    fig, axs = plt.subplots(nrows=1, ncols=3)

    axs[0].imshow(im_data[50])
    axs[0].set_title("Raw Cubert Image")
    
    axs[1].imshow(k_means_labels.reshape(im_size_x, im_size_y))
    axs[1].set_title("K Means Segmentation")

    # axs[2].imshow(affinity_labels.reshape(im_size_x, im_size_y))
    # axs[2].set_title("Spectral Clustering")

    plt.show()
