import skimage
import numpy as np
import cv2

from pathlib import Path
from matplotlib import pyplot as plt

def downsample(image: np.ndarray, factor: int) -> np.ndarray:
    """Downsample an image by an integer factor using area interpolation."""
    new_size = (image.shape[1] // factor, image.shape[0] // factor)
    downsampled_image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    return downsampled_image

def crop_and_save(imagepath: Path, rect: tuple, filepath) -> Path:
    """Crop `image` to rectangle `rect`=(x,y,w,h) and save to `filepath`.

    - Clips rectangle to image bounds.
    - Converts floating images to uint8 for saving.
    - Creates parent directories for `filepath` if necessary.

    Returns the `Path` to the saved file.
    """
    image = skimage.io.imread(imagepath)

    x, y, w, h = map(int, rect)
    if w <= 0 or h <= 0:
        raise ValueError("Width and height must be positive")

    img_h, img_w = image.shape[:2]
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(img_w, x0 + w)
    y1 = min(img_h, y0 + h)

    if x0 >= x1 or y0 >= y1:
        raise ValueError("Crop rectangle is outside image bounds")

    cropped = image[y0:y1, x0:x1].copy()

    out_path = Path(filepath)
    if out_path.parent:
        out_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert floats to uint8 for safe saving
    if np.issubdtype(cropped.dtype, np.floating):
        arr = cropped
        amin = float(arr.min())
        amax = float(arr.max())
        if amax > amin:
            norm = (arr - amin) / (amax - amin)
        else:
            norm = arr - amin
        to_save = (norm * 255).astype(np.uint8)
    else:
        to_save = cropped

    skimage.io.imsave(str(out_path), to_save)
    return out_path

def get_rectangles(scan_path: Path, show_steps = False):
    """
    The scanned data should be a .tif file. The scan consists of image plates that appear as rectangular sections in the scan. This function will crop the scan to individual image plates based on OpenCV contour detection. The displayed image is downscaled and gamma adjusted for speed and visualization purposes.

    1. Load the scanned image.
    2. Take background value - this is done by averaging over the the 50th row of pixels (assumes there is blank space at the top of the scan).
    3. Create a binary mask where pixel values greater than the background are set to 1, else 0.
    4. Smooth the binary mask using a Butterworth low-pass filter to reduce noise.
    5. Create another mask that only takes values above a threshold (0.1).
    6. Use OpenCV on this second mask to find rectangles that correspond to image plates. The minimum width and height of the rectangles are defined based on expected image plate sizes.

    """

    if scan_path.suffix.lower() in ['.jpg', '.png']:
        scan = skimage.io.imread(scan_path, as_gray=True)
        expadjusted = scan
    else:
        rawdata = skimage.io.imread(scan_path).astype(np.uint8)
        print(f"Raw data type: {rawdata.dtype}, min: {rawdata.min()}, max: {rawdata.max()}")
        smoothed = cv2.bilateralFilter(rawdata, 10, 30, 30) # spatial window, color tolerance window, coordinate tolerance window
        expadjusted = np.power((smoothed - smoothed.min()) / (smoothed.max() - smoothed.min()), 0.3)

    xsize = rawdata.shape[1]
    ysize = rawdata.shape[0]
    print(f"Scan size: {xsize} x {ysize}")

    # Smooth image using a low-pass filter
    bg = np.percentile(expadjusted, 0.005)
    mask = (expadjusted > bg)
    # smoothed_mask = cv2.boxFilter(mask.astype(np.float64), ddepth=-1, ksize=(30, 30))
    smoothed_mask = cv2.bilateralFilter(mask.astype(np.float32), 10, 30, 30)
    contour = 0.1
    contour_mask = np.where(smoothed_mask < contour, 0, 1)    

    print(f"Average Background Value: {bg:.2g}")
    print(f"Min = {expadjusted.min():.2g}")

    if show_steps:
        fig, ax = plt.subplots(nrows=3, figsize=(7, 7))
        ax[0].imshow(downsample(rawdata,8), cmap='jet')
        ax[0].set_title("1. Raw Data")
        ax[1].imshow(downsample(smoothed_mask,8), cmap='jet')
        ax[1].set_title("2. Mask + Low-pass")
        ax[2].imshow(downsample(contour_mask.astype(np.uint8),8), cmap='jet')
        ax[2].set_title(f"3. Contour > {contour}")
        plt.tight_layout()

    # Assume mask is a binary image (numpy array)
    mask_uint8 = (contour_mask ).astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rectangles = []
    mask_rgb = cv2.cvtColor((contour_mask * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    min_width = 800  # Width of a HXRD/PHC IP 
    min_height = 400  # Height of a eSpec IP 

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)  # (x,y) is the top-left coordinates of the rectangle and (w,h) is width and height.
        if w >= min_width and h >= min_height:
            rectangles.append((x, y, w, h))
            cv2.rectangle(mask_rgb, (x, y), (x + w, y + h), (255, 0, 0), 50)

    # Downsample scan_image for faster display
    downsample_factor=8
    scan_image_small = downsample(expadjusted, factor=downsample_factor)

    # Draw rectangles on the downsampled image
    scan_image_small_rgb = cv2.cvtColor((scan_image_small / scan_image_small.max() * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    for (x, y, w, h) in rectangles:
        x_s, y_s, w_s, h_s = x // downsample_factor, y // downsample_factor, w // downsample_factor, h // downsample_factor
        cv2.rectangle(scan_image_small_rgb, (x_s, y_s), (x_s + w_s, y_s + h_s), (255, 0, 0), 1)

    fig, ax = plt.subplots()
    ax.imshow(scan_image_small_rgb)
    ax.set_title(f"Detected Rectangles (Downsampled {downsample_factor}x, Gamma Adjusted)")

    return rectangles

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
    rectangles = get_rectangles(scan_file_path, show_steps=True)
    plt.show()
    crop_and_save(scan_file_path, rectangles[0], Path("output/test.tif"))