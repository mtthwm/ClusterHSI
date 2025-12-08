from sys import argv
from RGB.HSI2RGB import HSI2RGB
import glob
from os import path
from matplotlib.image import imsave
from tifffile import imread
import numpy as np

def convert_to_rgb(hsi_image):
        """Convert hyperspectral data to RGB using HSI2RGB."""
        # Reshape to (height, width, bands) for compatibility
        hsi_image = hsi_image.transpose(1, 2, 0)        # Define wavelengths
        wl = np.linspace(450, 850, hsi_image.shape[-1])        # Reshape for HSI2RGB processing
        data = np.reshape(hsi_image, (-1, hsi_image.shape[-1]))        # Convert to RGB
        rgb_image = HSI2RGB(wl, data, hsi_image.shape[0], hsi_image.shape[1], 65, 0.002)
        return np.clip(rgb_image, 0, 1)

def main ():
    src_dir = argv[1]
    dest_dir = argv[2]
    files = glob.glob(path.join(src_dir, "*.tif"))
    for file in files:
        arr = imread(file)
        rgb_arr = convert_to_rgb(arr)
        dest_path = path.join(dest_dir, file.split("/")[-1].replace('.tif', '.png'))
        imsave(dest_path, rgb_arr)

main()