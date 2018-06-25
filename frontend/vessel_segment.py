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
from colorama import Fore, init

# Other random imports
import numpy as np
from matplotlib import pyplot as plt
import cv2 # For the image processing stuff

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
    diff = diff*mask

    # Display the image (for testing only)
    fig = plt.figure(figsize=(10,5))

    ax1 = plt.subplot(1,2,1) 
    ax1.set_axis_off()
    ax1.set_title("Original Fundus")
    ax1.imshow(image, cmap="gray")
    
    ax2 = plt.subplot(1,2,2) 
    ax2.set_axis_off()
    ax2.set_title("Modified Fundus")
    ax2.imshow(diff, cmap="gray")

    fig.tight_layout(); plt.show();

if __name__ == "__main__":

    # Initialise Ansi encoding
    init()

    # Run the main script
    main()
