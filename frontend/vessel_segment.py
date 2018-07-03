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

def breadth_first(image_original, loc):
    """
    This function will run breadth first 
    search on a mask, given a starting pixel.
    The visited (white) pixels are turned to
    black.

    Arguments:
        image_original - The np array representing the mask.
        loc            - The initial location to start.

    Returns:
        Nump array with the searched image, white 
        pixels represent the unvisited pixels.
    """

    def get_neighbours(image, loc):
        """
        Inner function that will get the neighbouring 
        pixels of the given pixel in the image.
        """

    
        # Get  the locations to look
        left, right = loc[1]-1, loc[1]+1
        up, down = loc[0]-1, loc[0]+1

        # Placeholder for the location queue
        locs = []

        # Adding the locations
        if left >= 0 and image[loc[0], left] > 0:
            locs.append([loc[0], left])
        if right <= image.shape[1] and image[loc[0], right] > 0:
            locs.append([loc[0], right])
        if up >= 0 and image[up, loc[1]] > 0:
            locs.append([up, loc[1]])
        if down <= image.shape[0] and image[down, loc[1]] > 0:
            locs.append([down, loc[1]])

        return locs

    # Get a copy of the image
    image = np.copy(image_original)

    # Create a queue (each row is a position to visit)
    queue = np.zeros((1,2))
    queue[0,:] = loc # First index is the location passed

    # Now just perform breadth first
    while queue.shape[0] > 0:

        # Visit the pixel in the image
        current_loc = queue[0,:].astype(int)

        if image[tuple(current_loc)] > 0:

            # Get the neighbours and add to the queue
            ns = get_neighbours(image, current_loc)
            if len(ns) > 0: queue = np.vstack((queue, ns))

        queue = np.delete(queue, (0), axis=0)
        image[tuple(current_loc)] = 0

    return image

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

    # Dilate the image and erode it
    kernel = np.ones((5,5))
    dilated = cv2.dilate(image, kernel, iterations=5)
    eroded = cv2.erode(dilated, kernel, iterations=5)

    # Get the difference of the images here
    diff = (eroded - image)
    
    # Apply the circular mask to the fundus
    mask = ft.circular_mask(diff.shape, 400, 450)
    diff = (diff*mask).astype(np.uint8)
    diff = np.dstack((diff, diff, diff))

    # Apply the mean shift to smoothen the noise
    diff_shifted = cv2.pyrMeanShiftFiltering(diff, 10, 15)

    # Now calculate the mask with basic mean comparissons
    colour_mean = masked_array_func(np.mean, diff_shifted, mask>0)
    colour_var = masked_array_func(np.var, diff_shifted, mask>0) 
    print(colour_mean, colour_var)

    # Get the vessel network mask
    vessel_mask = (diff_shifted[:,:,0] >= colour_mean+np.sqrt(colour_var))*255

    # Do the segmenting on the image
    bad_areas = breadth_first(vessel_mask, (493,996))
    
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
    ax3.imshow(bad_areas, cmap="gray")
    fig.tight_layout(); plt.show();

if __name__ == "__main__":

    # Initialise Ansi encoding
    init()

    # Run the main script
    main()
