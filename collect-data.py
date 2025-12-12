import numpy as np
import json
from collections import defaultdict
import glob
import os
import tifffile
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.image import imsave
import random

CROP_SIZE = 128
NUM_BANDS = 106
LABELS = {
    "Beef": 1,
    "Chicken": 2,
    "Turkey": 3,
    "Pork": 4
}
LABEL_STUDIO_ROOT = "/home/matthew-morales/LabelStudioData"
DATASET_NAME = "MeatSegmentation"
WAVELENGTHS = np.linspace(450, 850, NUM_BANDS)
CROP_OFFSET = (88, 179)
DOWNSAMPLE_FACTOR = 4
DEBUG = True
IMG_SIZE = CROP_SIZE // DOWNSAMPLE_FACTOR

class LSC:
    '''
    This is a static class containing methods pulled from https://github.com/HumanSignal/label-studio-converter/blob/master/label_studio_converter/brush.py#L70

    Genuinely not sure why it didn't work out of the box, but I seemed to be able to get it working.
    '''
    class InputStream:
        def __init__(self, data):
            self.data = data
            self.i = 0

        def read(self, size):
            out = self.data[self.i : self.i + size]
            self.i += size
            return int(out, 2)
    
    def access_bit(data, num):
        """from bytes array to bits by num position"""
        base = int(num // 8)
        shift = 7 - int(num % 8)
        return (data[base] & (1 << shift)) >> shift
        
    def bytes2bit(data):
        """get bit string from bytes data"""
        return ''.join([str(LSC.access_bit(data, i)) for i in range(len(data) * 8)])
        
    def decode_rle(rle, print_params: bool = False):
        """from LS RLE to numpy uint8 3d image [width, height, channel]

        Args:
            print_params (bool, optional): If true, a RLE parameters print statement is suppressed
        """
        input = LSC.InputStream(LSC.bytes2bit(rle))
        num = input.read(32)
        word_size = input.read(5) + 1
        rle_sizes = [input.read(4) + 1 for _ in range(4)]

        if print_params:
            print(
                'RLE params:', num, 'values', word_size, 'word_size', rle_sizes, 'rle_sizes'
            )

        i = 0
        out = np.zeros(num, dtype=np.uint8)
        while i < num:
            x = input.read(1)
            j = i + 1 + input.read(rle_sizes[input.read(2)])
            if x:
                val = input.read(word_size)
                out[i:j] = val
                i = j
            else:
                while i < j:
                    val = input.read(word_size)
                    out[i] = val
                    i += 1
        return out
    
    def decode_from_annotation(from_name, results):
        """from LS annotation to {"tag_name + label_name": [numpy uint8 image (width x height)]}"""
        layers = {}
        counters = defaultdict(int)
        for result in results:
            key = (
                "brushlabels"
                if result["type"].lower() == "brushlabels"
                else ("labels" if result["type"].lower() == "labels" else None)
            )
            if key is None or "value" not in result:
                continue


            rle = result["value"]["rle"]
            width = result["original_width"]
            height = result["original_height"]
            labels = result["value"][key]
            name = "".join(labels)

            # result count
            i = str(counters[name])
            counters[name] += 1
            name += "-" + i

            image = LSC.decode_rle(rle)
            layers[name] = np.reshape(image, [height, width, 4])[:, :, 3]
        return layers 

def load_image_pair (json_file: str, root_dir: str) -> tuple[np.ndarray, np.ndarray]:
    '''
    Loads a JSON representation of the images
    '''
    def get_tiff_ver (col_file_name):
        return col_file_name.replace(".png", ".tif").replace("colorized", "ground-truth")
    
    labels = np.zeros(IMG_SIZE*IMG_SIZE)
    pixels = np.zeros((IMG_SIZE*IMG_SIZE, 106))

    with open(json_file) as fp:
        task_json = json.load(fp)
        layers = LSC.decode_from_annotation("tag", task_json["result"])
        task = task_json["task"]
        col_file = task["data"]["image"].replace("/data/local-files/?d=", "")
        hsi_file = os.path.join(root_dir, get_tiff_ver(col_file))
        full_hsi_arr = tifffile.imread(hsi_file)[:,CROP_OFFSET[1]:CROP_OFFSET[1]+CROP_SIZE,CROP_OFFSET[0]:CROP_OFFSET[0]+CROP_SIZE]
        hsi_arr = full_hsi_arr[:, ::DOWNSAMPLE_FACTOR, ::DOWNSAMPLE_FACTOR]
        if hsi_arr.shape == (106, IMG_SIZE, IMG_SIZE):
            for y in range(IMG_SIZE):
                for x in range(IMG_SIZE):
                    pixels[y*IMG_SIZE + x, :] = hsi_arr[:, y, x]
            for layer_name, full_arr in layers.items():
                arr = full_arr[CROP_OFFSET[1]:CROP_OFFSET[1]+CROP_SIZE,CROP_OFFSET[0]:CROP_OFFSET[0]+CROP_SIZE]
                arr = arr[::DOWNSAMPLE_FACTOR, ::DOWNSAMPLE_FACTOR]
                for y in range(IMG_SIZE):
                    for x in range(IMG_SIZE):
                        if arr[y][x] != 0:
                            label_name = layer_name.split("-")[0]
                            labels[y*IMG_SIZE + x] = LABELS[label_name]
        

            return labels, pixels
        else:
            print("INCORRECT SIZE. SKIPPING")
            return None, None

def main ():
    files = glob.glob(os.path.join(LABEL_STUDIO_ROOT, DATASET_NAME, "train/masks/*"))
    pixel_num = IMG_SIZE*IMG_SIZE*len(files)
    all_pixels = np.zeros((pixel_num, 106), dtype=np.uint16)
    all_labels = np.zeros(pixel_num)
    for i, file in enumerate(files):
        labels, pixels = load_image_pair(file, LABEL_STUDIO_ROOT)
        if DEBUG:
            file_name = file.split("/")[-1]
            imsave(f"./test/{file_name}.png", labels.reshape(IMG_SIZE, IMG_SIZE))
        if labels is not None and np.any(labels):
            IMGSQ = IMG_SIZE*IMG_SIZE
            for j in range(0, IMGSQ):
                idx = i*IMGSQ+j
                all_pixels[idx] = pixels[j]
                all_labels[idx] = labels[j]

    id = datetime.now().strftime("%m%d%H%M")
    np.save(f"spectra-{DATASET_NAME}-{id}", all_pixels)
    np.save(f"labels-{DATASET_NAME}-{id}", all_labels)

main()