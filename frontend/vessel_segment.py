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
    mask = ft.circular_mask(diff.shape, 600, 601)
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

    # Create the parameters and filtering options
    params = cv2.SimpleBlobDetector_Params()
    params.filterByInertia = True
    params.minInertiaRatio = 0.025
    params.blobColor = 0

    # Creat the detector with the parameters
    detector = cv2.SimpleBlobDetector_create(params)

    # Get the keypoints with the detector
    keypoints = detector.detect(vessel_mask)
    kpt = keypoints[2]

    print("Point: ", kpt.pt)
    print("Octave: ", kpt.octave)
    print("Size: ", kpt.size)
    print("Angle: ", kpt.angle)
    print("Overlap: ", kpt.overlap)

    # Draw the blobs detected
    flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS # To match size
    out_img = np.zeros(vessel_mask.shape).astype(np.uint8) # Empty (but it is required)
    blob_img = cv2.drawKeypoints(out_img, keypoints, np.array([]), 255, flags)

    # Display the image (for testing only)
    fig = plt.figure(figsize=(17,5))

    ax1 = plt.subplot(1,3,1) 
    ax1.set_axis_off()
    ax1.set_title("Original Fundus")
    ax1.imshow(image, cmap="gray")
    
    ax2 = plt.subplot(1,3,2) 
    ax2.set_axis_off()
    ax2.set_title("Vessel Network Mask")
    ax2.imshow(vessel_mask, cmap="gray")

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
