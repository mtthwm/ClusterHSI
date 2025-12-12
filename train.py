import numpy as np
from tifffile import imread
from matplotlib.image import imsave
from sklearn.svm import LinearSVC
from pickle import load, dump
from datetime import datetime

SPECTRA_FILE = "spectra-MeatSegmentation-12120029.npy" 
LABEL_FILE = "labels-MeatSegmentation-12120029.npy"
TEST_FILE = "/home/matthew-morales/LabelStudioData/MeatSegmentation/unallocated/ground-truth/JfZ980ume2N4DKWS.tif"
IMG_SIZE = 352

MODEL_PICKLE="./model-12120050.pkl"

def train ():
    spectra = np.load(SPECTRA_FILE)
    labels = np.load(LABEL_FILE)

    print(spectra.shape, labels.shape)

    model = LinearSVC()
    estimator = model.fit(spectra, labels)
    return estimator

def main ():
    if not MODEL_PICKLE:
        estimator = train()
        with open(f"model-{datetime.now().strftime("%m%d%H%M")}.pkl", "wb") as f:
            dump(estimator, f, protocol=5)
    else:
        with open(MODEL_PICKLE, "rb") as f:
            estimator = load(f)
    
    test_arr = imread(TEST_FILE)
    label_image = np.zeros((IMG_SIZE, IMG_SIZE))

    for y in range(IMG_SIZE):
        for x in range(IMG_SIZE):
            spect = test_arr[:, y, x]
            label_guess = estimator.predict(spect.reshape(1, -1))[0]
            label_image[y,x] = label_guess

    imsave("./test.png", label_image/3)


main()