"""vessel_segment.py

This script contains the code to
segment the vessel network in the 
fundus.

Usage:
    vessel_segment.py PATH

Arguments:
    PATH  The path to the image of the eye to segment.
"""

# Import the files stuff
import os, sys
sys.path.append(os.path.abspath("../"))
from utils import files

# The frontentd image processing stuff
from frontend import analysis as an
from frontend import utilities as ut
from frontend import filtering as ft
from frontend.utilities import cprint
from frontend.blob_detector import BlobDetector

# CLI and argparse stuff
from docopt import docopt
from colorama import Fore, Style, init

# Other random imports
import numpy as np
from matplotlib import pyplot as plt
import cv2 # For the image processing stuff

def masked_array_func(func, array, mask):
    """
    This function will apply the given function
    on the given array, only considering the 
    elements.

    Arguments:
        func  - Function that takes (array) as the argument.
        array - Array containing the data.
        mask  - Array of booleans representing the mask.

    Returns:
        func(array[mask]).    
    """
    #assert array.shape == mask.shape, "Dimensions do not match."
    assert mask.dtype == np.bool, "Mask must be of type boolean."

    return func(array[mask])

def main():
    """
    Main script here. This is where all the 
    logic for the stuff occurs.
    """

    ###############################
    ## Parsing the cmd line args ##
    ###############################

    args = docopt(__doc__)
    path = files.abspath(args["PATH"])

    # Check if path exists, if not exit
    if not files.exists(path):
        cprint("File path does not exist. Exiting...", Fore.RED)
        sys.exit()

    # Open the image
    image = ut.read_image(path)[:,:,1]
    

    #######################
    ## Vessel Extraction ##
    #######################

    # Dilate the image and erode it
    kernel = np.ones((5,5))
    dilated = cv2.dilate(image, kernel, iterations=5)
    eroded = cv2.erode(dilated, kernel, iterations=5)

    # Get the difference of the images here
    diff = (eroded - image)
    
    # Apply the circular mask to the fundus
    mask = ft.circular_mask(diff.shape, 450, 451)
    diff = (diff*mask).astype(np.uint8)
    diff = np.dstack((diff, diff, diff))

    # Apply the mean shift to smoothen the noise
    diff_shifted = cv2.pyrMeanShiftFiltering(diff, 10, 15)
    diff_shifted = diff_shifted[:,:,0].astype(np.uint8)

    # Now calculate the mask with basic mean comparissons
    colour_mean = masked_array_func(np.mean, diff_shifted, mask>0)
    colour_var = masked_array_func(np.var, diff_shifted, mask>0) 
    print(colour_mean, colour_var)

    # Get the vessel network mask
    thresh_type = cv2.THRESH_BINARY_INV
    thresh_val = colour_mean + np.sqrt(colour_var)
    deduct = 5; size = 9 # Mean deduction and block size
    ret_val, vessel_mask = cv2.threshold(diff_shifted, thresh_val,
                           255, thresh_type)
    

    
    ##########################
    ## Blob Detection Stuff ##
    ##########################
    
    # Create the detector 
    detector = BlobDetector()
    detector.set_colour(0, 10, 1)
    detector.filter_inertia(True, 0.25, 100)
    #detector.filter_convexity(True, 0.5, 2)
    detector.filter_area(True, 10, 1000)
    
    # Get the blobs keypoint stuff
    points, sizes = detector.detect(cv2.erode(vessel_mask,kernel, iterations=1))

    # Draw the blobs detected
    blob_img = diff_shifted.copy()
    for i in range(len(points)):
        cv2.circle(blob_img, (points[i][0], points[i][1]), int(sizes[i]/2), 255, 4)

    # Display the image (for testing only)
    fig = plt.figure(figsize=(17,5))

    ax1 = plt.subplot(1,3,1) 
    ax1.set_axis_off()
    ax1.set_title("Original Fundus")
    ax1.imshow(image, cmap="gray")
    
    ax2 = plt.subplot(1,3,2) 
    ax2.set_axis_off()
    ax2.set_title("Vessel Network Mask")
    ax2.imshow(diff_shifted, cmap="gray")

    ax3 = plt.subplot(1,3,3)
    ax2.set_axis_off()
    ax3.set_title("Detected Outside Network")
    ax3.imshow(blob_img, cmap="gray")
    fig.tight_layout(); plt.show();

if __name__ == "__main__":

    # Initialise Ansi encoding
    init()

    # Run the main script
    main()
