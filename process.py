import skimage
import numpy as np
import cv2

from pathlib import Path
from matplotlib import pyplot as plt

def process_scan(scan_path: Path, filenames: list = None, show_steps = False):
    """
    The scanned data should be a .tif file. The scan consists of image plates that appear as rectangular sections in the scan. This function will crop the scan to individual image plates based on OpenCV contour detection. The displayed image is downscaled and gamma adjusted for speed and visualization purposes.

    1. Load the scanned image.
    2. Take background value - this is done by averaging over the the 50th row of pixels (assumes there is blank space at the top of the scan).
    3. Create a binary mask where pixel values greater than the background are set to 1, else 0.
    4. Smooth the binary mask using a Butterworth low-pass filter to reduce noise.
    5. Create another mask that only takes values above a threshold (0.1).
    6. Use OpenCV on this second mask to find rectangles that correspond to image plates. The minimum width and height of the rectangles are defined based on expected image plate sizes.

    """

    if filenames is None:
        print("Filenames not provided")
        return
    n_images = len(filenames)

    scan = skimage.io.imread(scan_path)
    
    if scan_path.suffix.lower() in ['.jpg', '.png']:
        scan_image = 0.2125 * scan[:,:,0] + 0.7154 * scan[:,:,1] + 0.0721 * scan[:,:,2]
        scan_image_adjusted = scan_image_adjusted
    else:
        scan_image = scan
        scan_image_adjusted = skimage.exposure.adjust_gamma(scan, 0.1) # for visualization only

    xsize = scan_image.shape[1]
    ysize = scan_image.shape[0]
    print(f"Scan size: {xsize} x {ysize}")
    
    horizontal_sum =  scan_image.sum(axis=1)    
    bg_avg = horizontal_sum[50]/xsize

    # Smooth image using a butterworth low-pass filter
    mask = (scan_image > bg_avg)
    smoothed_mask = skimage.filters.butterworth(mask, high_pass=False, cutoff_frequency_ratio=0.03)
    mask_2 = np.where(smoothed_mask < 0.1, 0, 1)    

    print(f"Average Background Value: {bg_avg:.2f}")
    print(f"Min = {scan_image.min():.1f}")

    if show_steps:
        fig, ax = plt.subplots(nrows=2, figsize=(7, 7))
        ax[0].imshow(smoothed_mask, cmap='jet')
        ax[0].set_title("2. Low-pass")
        ax[1].imshow(mask_2, cmap='jet')
        ax[1].set_title("3. Contour > 0.1")
        plt.tight_layout()

    # Assume mask is a binary image (numpy array)
    mask_uint8 = (mask_2 * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rectangles = []
    mask_rgb = cv2.cvtColor((mask_2 * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    min_width = 800  # Width of a HXRD/PHC IP 
    min_height = 400  # Height of a eSpec IP 

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w >= min_width and h >= min_height:
            rectangles.append((x, y, w, h))
            cv2.rectangle(mask_rgb, (x, y), (x + w, y + h), (255, 0, 0), 50)

    # Downsample scan_image for faster display
    downsample_factor = 10  # Adjust as needed for speed/quality tradeoff
    scan_image_small = cv2.resize(scan_image_adjusted, (scan_image.shape[1] // downsample_factor, scan_image.shape[0] // downsample_factor), interpolation=cv2.INTER_AREA)

    # Draw rectangles on the downsampled image
    scan_image_small_rgb = cv2.cvtColor((scan_image_small / scan_image_small.max() * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    for (x, y, w, h) in rectangles:
        x_s, y_s, w_s, h_s = x // downsample_factor, y // downsample_factor, w // downsample_factor, h // downsample_factor
        cv2.rectangle(scan_image_small_rgb, (x_s, y_s), (x_s + w_s, y_s + h_s), (255, 0, 0), 2)

    fig, ax = plt.subplots()
    ax.imshow(scan_image_small_rgb)
    ax.set_title("Detected Rectangles (Downsampled 10x, Gamma Adjusted)")

if __name__ == "__main__":
    scan_file_path = Path("data/shot06-[Phosphor].tif")
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
    process_scan(scan_file_path, filenames, show_steps=True)
    plt.show()