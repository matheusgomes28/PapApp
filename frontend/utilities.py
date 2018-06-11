"""
This file will contain some utility code
that will be useful when processing images.

Most of the functions here will be used in 
Jupyter notebook in order to load, show and 
analyse images in the correct format 
(Matplotlib), and convert between colour 
spaces (E.g BGR -> RGB).

Some other code may be dumped here if 
its being used a lot in multiple scripts.
"""
# Version 0.1 - Python 3.x

import cv2, sys, os
import numpy as np
import cv2
import files 

# This holds the strings for the conversion
# for the conversion codes. Feel free to add 
# more strings if you need a different space
# for analysis (format = "{SPACE1}2{SPACE2})
convert_strs = {
				 "BGR2RGB" : cv2.COLOR_BGR2RGB,
				 "BGR2GRAY": cv2.COLOR_BGR2GRAY,
				 "RGB2BGR" : cv2.COLOR_RGB2BGR,
				 "RGB2GRAY": cv2.COLOR_RGB2GRAY,
			   }

def convert_spaces(img:np.ndarray, spaces:str) -> np.ndarray:

	# Return a new image in the converted space
	return cv2.cvtColor(img, convert_strs[spaces])


def read_image(path:str, spaces:str="BGR2RGB") -> np.ndarray:
	"""
	This funtion will use OpenCV to load an image. 
	Note that the default colour space is BGR. To 
	convert between spcaes, just change the default 
	'key' argument (see convert_strs). 

	path - string representing the path of the image
	spaces - string representing key in the convert_dir
	"""

	# Load image (BGR)
	img = cv2.imread(path)

	# Returns the converted image 
	return convert_spaces(img, spaces)
def save_image(filename:str, image:np.ndarray):
	"""
	Uses OpenCV to write an image to the disk. Make
	sure the path given in filename already exists,
	OpenCV does NOT create new directories.

	Args:
		filename - String representing the path + filename + extension.
		image - Np array with the image data.
	"""

	# For more info, view the OpenCV documentation
	cv2.imwrite(filename, image)

# Function for the loading bars
def update_line(text, chars=["\033[F","\r"]): 
    """
    This function will output text on
    the same line. I.e update the line
    with the new text using ANSII.

    Only use this for CLIs as info is printed to stdout.
    """

    # If windows is being used, flush explicitly or 
    # cmd won't output properly.
    if os.name == 'nt':
        # Print text and update cursor
        sys.stdout.write(text)
        sys.stdout.flush()

        sys.stdout.write(chars[1])
        sys.stdout.flush()	

    else:
        sys.stdout.write(text + "\n")
        sys.stdout.write(chars[0])


## Any questions on this code, please just
## ask me (Mat)
