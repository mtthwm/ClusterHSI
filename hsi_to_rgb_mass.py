from sys import argv
from RGB.HSI2RGB import HSI2RGB
import glob
from os import path
from matplotlib.image import imsave
from tifffile import imread
import numpy as np

CROP_SIZE = 352

def convert_to_rgb(hsi_image):
        """Convert hyperspectral data to RGB using HSI2RGB."""
        # Reshape to (height, width, bands) for compatibility
        hsi_image = hsi_image.transpose(1, 2, 0)        # Define wavelengths
        wl = np.linspace(450, 850, hsi_image.shape[-1])        # Reshape for HSI2RGB processing
        data = np.reshape(hsi_image, (-1, hsi_image.shape[-1]))        # Convert to RGB
        rgb_image = HSI2RGB(wl, data, hsi_image.shape[0], hsi_image.shape[1], 65, 0.002)
        return np.clip(rgb_image, 0, 1)

def center_crop(img, target_h, target_w):
    """Crop image symmetrically to target size."""
    if img.ndim == 3:
        _, h, w = img.shape
        top = (h - target_h) // 2
        left = (w - target_w) // 2
        return img[:, top:top+target_h, left:left+target_w]
    return img

def main ():
    src_dir = argv[1]
    dest_dir = argv[2]
    do_crop = len(argv) > 3 and argv[3] == "crop"
    files = glob.glob(path.join(src_dir, "*.tif"))
    for file in files:
        arr = imread(file)
        if do_crop:
            arr = center_crop(arr, CROP_SIZE, CROP_SIZE)
            rgb_arr = convert_to_rgb(arr[0:106])
        else:
            rgb_arr = convert_to_rgb(arr)
        dest_path = path.join(dest_dir, file.split("/")[-1].replace('.tif', '.png'))
        imsave(dest_path, rgb_arr)

main()