""" normalise.py

This script contains the code for standardising
the images sent from the user. This script will 
be changed once the server is sorted, for now it
is only for testing.

Usage:
    normalise.py -i PATH

Options:
    -i PATH --img=PATH    Indicates the path of the image to use.
"""

# Impor tthe files package
import sys, os
sys.path.append(os.path.abspath("../"))
from utils import files

# Import for the CLI and argparse stuff
from docopt import docopt
from colorama import Fore,Back, Style, init

# The imports from the frontend 
import filtering as ft
import analysis as an
import utilities as ut

# Import for maths and display (mockup)
import numpy as np
from matplotlib import pyplot as plt


################################
## USEFUL FUNCTION DEFINITINS ##
################################

def resize(image, width, height):
    """
    Function to reize a given image. Have to 
    make sure OpenCV is properly installed.
    """
    # Interpolation choice (area for minimisation)
    inter = cv2.CV_INTER_AREA

    # Use the chosen interpolation to resize 
    return cv2.resize(image, (width, heigh), inter)



##################
## TESTING CODE ##
##################

if __name__ == "__main__":
    print("Hello there, imports have worked!")
    
    # Init the window ANSI output
    init() # from the colorama library
    
    
    # Parse the CLI args
    args = docopt(__doc__)
    img_path = files.abspath(args["--img"])

    # Check whether the path exists and its image
    if not files.exists(img_path):
        print("File does not exist. Check the string.")
        sys.exit()
    if not files.is_filetype(img_path):
        print("File given is not an image.")
        sys.exit()


    # Load the image (in greyscale)
    image = ut.read_image(img_path, "BGR2GRAY")

    # Blur measures
    err = an.get_err(image)
    lap = an.get_laplacian(image)
    lap_var = np.var(lap)
    
    # Histogram calculation
    histogram = an.get_histogram(image)


    # Now plot the stuff
    fig = plt.figure(figsize=(10,5))
    fig.suptitle("Laplacian: {:.2f}, Err: {:.2f}".format(lap_var, err))

    ax1 = plt.subplot(1,3,1)
    ax1.set_axis_off()
    ax1.set_title("Orginal image")
    ax1.imshow(image, cmap="gray")

    ax2 = plt.subplot(1,3,2)
    ax2.set_title("Histogram")
    ax2.plot(histogram)

    ax3 = plt.subplot(1,3,3)
    ax3.set_axis_off()
    ax3.set_title("Laplacian Image")
    plt.imshow(lap, cmap="gray")

    fig.tight_layout()
    plt.show()

