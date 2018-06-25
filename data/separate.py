"""separate.py 

Script to separate the image files
into different labels/folders 
according to the CVS file passed.

Usage:
    separate.py FILE -n NAME -c COL

Options:
    -n NAME --name==NAME  The image name column.
    -c COL --column=COL   The column index for the labels.

Arguments:
    FILE  the path to the cvs file to use
"""

# File handling imports
import os, sys
sys.path.append(os.path.abspath("../"))
from utils import files
from frontend import utilities as ut
from fronend import utilities.cprint as cprint

# Imports for the CLI stuff
from docopt import docopt
from colorama import Fore, Style, init

# Random imports
import pandas as pd
import numpy as np

# CLI definition stuff here
c = {"RED": Fore.RED, "GREEN": Fore.GREEN, "BLUE": Fore.BLUE}

def loading_bar(current, N, size=20):
    """
    Loading bar, same as the one in the model
    histogram script.
    """

    perc = (current+1)/N
    bar = int(np.round(perc*size))
    line = "Copying ["
    line += "="*bar + " "*(size-bar)
    line += "] {:d}%".format(int(np.round(perc*100)))
    ut.update_line(line) # Use the line update stuff from utils


def main():
    """
    Main process function.
    Where the actual file moving stuff is.
    """

    #########################
    ## PARSE THE ARGUMENTS ##
    #########################
    args = docopt(__doc__)
    name = args["--name"]
    column = args["--column"]
    fpath = files.abspath(args["FILE"])
    

    # Print out the arguments passed
    cprint("File passed: {}".format(fpath), c["BLUE"])
    cprint("Image name column: {}. Label column: {}".format(name,column),c["BLUE"])

    # Check files exist
    if not files.exists(fpath):
        cprint("Feature file does not exist. Exiting...", c["RED"])
        sys.exit()


    #########################
    ## SEGMENTING THE FILE ##
    #########################
    features = pd.read_csv(fpath) # Load the data

    # Get the labels (to create the folder) and total
    labels = features[column].unique()
    print("Labels are: " + str(labels))
    total_rows = features.shape[0]
    current = 0

    # Create each folder to save the images
    for l in labels:
       
        # Check if folder exists, and create dir
        img_dir = os.path.dirname(fpath) # Images directory path
        lpath = files.append_path(img_dir, str(l)) # Label path

        if not files.exists(lpath): files.mkdir(lpath)

        # Now use iterrows to save each image
        for i, row in features[features[column]==l].iterrows():

            # Get the source and destination path strings for images
            source = files.append_path(img_dir, str(row["name"]))
            dst = files.append_path(lpath, str(row["name"]))
            
            # Finally copy this image to the label folder
            files.copy_image(source, dst)

            # Print the loading bar
            loading_bar(current, total_rows, 40)
            current += 1
            

    # Finished processing stuff
    cprint("\nFinished!", c["GREEN"])


if __name__ == "__main__":

    # Initialise the Ansi stuff
    init()

    # Run the main code
    main()
