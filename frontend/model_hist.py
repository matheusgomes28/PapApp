"""model_hist.py

Python script for creating a model histogram. Given 
a image directory, it will load all the images and 
calculate the mean histogram of the loaded images.
If givne a file instead, it will use the function in 
the file as the generator for the histogram.


Usage:
    model_hist.py --dir=DPATH
    model_hist.py --file=FPATh
    model_hist.py -h | --help

Options:
    -d DPATH --dir=DPATH    Indicate the image directory to use.
    -f FPATH --file=FPATH   Indicate the file path to use.
"""

# Impor the files package
import os, sys
sys.path.append(os.path.abspath("../"))
from utils import files

# Import for the CLI
from docopt import docopt
from colorama import Fore, Style, init

# The standard imports from fronend package
from frontend import analysis as an 
from frontend import utilities as ut

# For the maths stuff
import numpy as np
from matplotlib import pyplot as plt


def loading_bar(curent, N, size=20):
    """
    Simple loading bar for command lines.
    Should work well on unix systems.

    Args:
        current - Current index in iteration.
        N       - Last possible index.
        size    - Size of loading bar in chars.
    """

    perc = (current+1)/N
    bar  = int(np.round(perc*size))
    line = "Processing ["
    line += "="*bar + " "*(size-bar)
    line += "] {:d}%".format(int(np.round(perc*100)))
    ut.update_line(line) # This will deal with the carriage return stuff
    # Note everything is printed to sys.out

    
def main():
    """
    Main method, where all the actual logic goes into.
    CLI args parsing is done with docopt and colorama,
    then the Numpy packages are used to get the histogram.
    """

    ######################
    ## ARGUMENT PARSING ##
    ######################
    args = docopt(__doc__)

    # Print out the passed args
    init() # Init windows ANSI
    print("Arguments passed")

    dir_text = "Use directory? "
    if args["--dir"]: 
        dir_text += "YES"
        dir_text  += ", " + args["--dir"]
    else: dir_text += "NO"

    file_text = "Use file? "
    if args["--file"]: 
        file_text += "YES"
        file_text += ", " + args["--file"]
    else: file_text += "NO"
    
    # Now just print out the results
    print(Fore.GREEN + dir_text); print(file_text)


    #############################
    ## IMAGE DIRECTORY LOADING ##
    #############################
    if args["--dir"]:
        
        # Parse path and make sure it exists
        path = files.abspath(args["--dir"])
        if not files.exists(path):
            print(Fore.RED + "Path does not exists. Exiting.." + Style.RESET_ALL)
            sys.exit()
       
 
        # Create the histogram accumulator
        path = files.abspath(args["--dir"])
        img_paths = files.get_images(path)
        num_images = len(img_paths)
        hist_acc = np.zeros((num_images, 256)) # N_IMAGES x INTENSITIES
        print(Fore.BLUE + "Number of images ot load: {}".format(num_images))
        
        for i, path in enumerate(img_paths):
            # Load image in greyscale
            image = ut.read_image(path, "BGR2GRAY")

            # Update Nth row with the current histogram
            hist = an.get_histogram(image)
            hist /= np.sum(hist)
            hist_acc[i,:] = np.ravel(hist)

        # Now that we have accumulated the hist, take the mean
        model_hist = np.mean(hist_acc, axis=0)

        # Plot and save the results
        fig = plt.figure(figsize=(5, 2))
        ax1 = plt.gca()
        ax1.set_title("Histogram obtained")
        ax1.plot(model_hist)
        fig.tight_layout(); plt.show();
        np.savetxt("model_hist.txt", model_hist)

        


    ###########################
    ## FILE FUNCTION LOADING ##
    ###########################
    if args["--file"]:
        pass


# Init method
if __name__ == "__main__":
    main()
