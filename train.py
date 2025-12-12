import numpy as np
from tifffile import imread
import matplotlib.pyplot as plt
from matplotlib.image import imsave
from sklearn.svm import LinearSVC

SPECTRA_FILE = "spectra-MeatSegmentation-12112336.npy" 
LABEL_FILE = "labels-MeatSegmentation-12112336.npy"
TEST_FILE = "/home/matthew-morales/LabelStudioData/MeatSegmentation/unallocated/ground-truth/vdR18sWHhCqsJnk3.tif"
IMG_SIZE = 352

def main ():
    spectra = np.load(SPECTRA_FILE)
    labels = np.load(LABEL_FILE)

    print(spectra.shape)

    model = LinearSVC()
    estimator = model.fit(spectra, labels)
    
    test_arr = imread(TEST_FILE)
    label_image = np.zeros((IMG_SIZE, IMG_SIZE))

    for y in range(IMG_SIZE):
        for x in range(IMG_SIZE):
            spect = test_arr[:, y, x]
            label_guess = estimator.predict(spect)
            label_image[y,x] = label_guess

    imsave("./test.png", label_image/3)


main()