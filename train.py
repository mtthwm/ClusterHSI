import numpy as np
import random
import matplotlib.pyplot as plt

SPECTRA_FILE = "spectra-MeatSegmentation-12112112.npy" 
LABEL_FILE = "labels-MeatSegmentation-12112112.npy"

def main ():
    spectra = np.load(SPECTRA_FILE)
    labels = np.load(LABEL_FILE)

    X = np.linspace(0, 105, 106)

    for _ in range(10):
        ridx = random.randint(0, len(spectra))
        spect = spectra[ridx]    
        plt.plot(X, spect)

    plt.show()
main()