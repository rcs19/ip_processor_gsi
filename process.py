import skimage
import numpy as np

from pathlib import Path
from matplotlib import pyplot as plt

def process_scan(scan_path: Path, filenames: list = None):
    """
    The scanned data should be a single image (.tif). The scan consists of image plates that appear as rectangular sections in the scan. This function will manually crop the image by adding horizontal and vertical lines to indicate the cropping positions.
    1. Load the scanned image.
    2. Add horizontal and vertical lines by clicking on the horizontal and vertical sum plots.
    """

    if filenames is None:
        print("Filenames not provided")
        return
    n_images = len(filenames)

    scan = skimage.io.imread(scan_path)
    
    if scan_path.suffix.lower() in ['.jpg', '.png']:
        scan_image = 0.2125 * scan[:,:,0] + 0.7154 * scan[:,:,1] + 0.0721 * scan[:,:,2]

    xsize = scan_image.shape[1]
    ysize = scan_image.shape[0]
    print(f"Scan size: {xsize} x {ysize}")
    
    smooth = 5
    horizontal_sum =  np.convolve(scan_image.sum(axis=1), np.ones(smooth)/smooth, mode='same')[::-1]
    vertical_sum = np.convolve(scan_image.sum(axis=0), np.ones(smooth)/smooth, mode='same')
    
    # Adjust layout to remove whitespace between subplots
    fig, ax = plt.subplots(2, 2, figsize=(10, 6), gridspec_kw={'wspace': 0, 'hspace': 0})

    ax[0,0].imshow(scan_image, cmap='gray')
    ax[0,1].plot(horizontal_sum, range(len(horizontal_sum)))    
    ax[1,0].plot(vertical_sum)
    ax[1,1].set_axis_off()
    ax[0,1].invert_yaxis()  # Match orientation with the original image

    ax[1,0].set_xlim(0, scan_image.shape[1])
    ax[0,1].set_ylim(0, scan_image.shape[0])
    ax[1,0] = ax[0,0].twiny()
    ax[0,1] = ax[0,0].twinx()

    # Remove tick labels between subplots
    for axes in [ax[0,0], ax[0,1], ax[1,0]]:
        axes.set_aspect('auto', adjustable='box')
    
    plt.show()

if __name__ == "__main__":
    scan_file_path = Path("data/example.jpg")
    filenames = [
        "HOPG",
        "XCPI",
        "KE",
        "ESM1e",
        "ESM1p",
        "ESM2e",
        "ESM2p",
        "XPPHC1",
        "XRPHC2",
        "HXRD1",
        "HXRD2",
        ]
    process_scan(scan_file_path, filenames)