""" File Management Python Cript

This script is supposed to contain all the necessary functions
to manage files and paths in the OS. Note that this code was
only tested in Windows and Linux environments.
"""

import sys, os, cv2
from colorama import Fore, Style


def abspath(path):
    """
    This function is designed to fix cross platform 
    errors with os's abspath function (windows uses
    \ while linux / ).

	Params:
    path - the path stirng of the file/directory.

    returns -> str representing absolute path in curr os.
    """

    return os.path.abspath(path).replace("\\", "/")


def relpath(path):
    """
    This function is design to fix cross platform 
    errors with os's relpath function (windows uses 
	\ while linux uses /).

	Params:
	path - the path stirng of fie/folder.

	returns -> str representing relative path.
    """

    return os.path.relpath(path).replace("\\", "/")


def get_images(path):
    """
    This function will return a list with all the
    images names in a given path.

    Params:
    path - the path string (rel or abs) representing folder path.

    returns -> list of string containing all the images.
    """

    # Cast path to absolute path
    absolute = abspath(path)

    img_lis = []  # Holds images in a folder
    file_lis = get_files(absolute)

    # Now get the images within file list
    img_lis = [f for f in file_lis if is_filetype(f)]

    return img_lis


def get_files(path, formats=[]):
    """
    This function will return a list of all files in 
    the given directory.

    Params:
    path - the path (rel or abs) of the directory to get files.

    returns -> list of strings containing all the files in a dir.
    """

    # Uses abs path as the directory
    absolute = abspath(path)
    all_files = os.listdir(absolute)

    # Get the absolute path of each file
    absolute_files = ["/".join([absolute, i]) for i in all_files]

    # Filter out non-files and return
    filtered_files = [f for f in absolute_files if os.path.isfile(f)]

    # Filter out unwanted file types (if requested)
    if formats:
        filtered_files = [f for f in filtered_files if is_filetype(f, formats)]
    
    return filtered_files


def get_directories(path):
    """
    This function will return all folders in
    a given path. Only returns directory*

    Params:
    path - string representing directory to search.

    returns -> list of strings containing all directories.
    """

    # Uses abspath as the directory
    absolute = os.path.dirname(abspath(path))
    all_files = os.listdir(absolute)

    # Get the absolute path of each file
    absolute_files = ["/".join([absolute, d]) for d in all_files]

    # Here we filter all non-directires out and return
    return [i for i in absolute_files if os.path.isdir(i)]


def get_relative_dir(path1, path2):
    """
    This function will return the relative location 
    (wihout filenames) of the second file to the 
    first file.

    Examples: 
    get_relative_dir("parent/file.txt", "parent/child/pic.jpg") -> "child/"
    get_relative_dir("parent/child/file.txt", "parent/pic.jpg") -> "../"

    Params: 
    path1 - string representing the first dir (absolute dir)
    path2 - string representing the second dir.

    returns -> string representing path2 relative to path1.
    """

    originalwd = os.getcwd()  # Get original working directory

    # Get directories if files given
    if os.path.isdir(path1):
        dir1 = path1
    else:
        dir1 = os.path.dirname(path1)

    if os.path.isdir(path2):
        dir2 = path2
    else:
        dir2 = os.path.dirname(path2)

    # Change working dir
    os.chdir(dir1)
    rel_dir = relpath(dir2)

    os.chdir(originalwd)  # switch back to wd

    # return the relative path
    return "/".join([rel_dir, os.path.basename(path2)])


def is_filetype(img_path, formats=["jpg", "png", "gif", "pgm", "tif", "ppm"]):
    """
    Determines whether or not a
    given file is an image.

    Params:
    img_path - string representing path of image.
    formats - list os formats (strings).

    returns -> True if file is image, False if not.
    """
    # formats = ["jpg", "png", "gif", "pgm"]
    end = img_path[-3:]
    return os.path.isfile(img_path) and (end in formats)

    
def get_filename(file_path):
    """
    This function uses the os library to return 
    only the filename, without the extension.

    Args:
        file_path - string representing the path of the file.

    Retrns:
        string representing just the name, no extension.
    """

    # Get rid of directories and etc
    just_file = os.path.basename(file_path)

    # Now we return just the base name
    return os.path.splitext(just_file)[0]

def get_fileext(file_path):
    """
    This function uses the os library to return 
    only the extension, without the filename.

    Args:
        file_path - string representing the path of the file.

    Retrns:
        string representing just the extension (with the .), no filename.
    """

    # Get rid of directories and etc
    just_file = os.path.basename(file_path)

    # Now we return just the base name
    return os.path.splitext(just_file)[1]

def append_path(path1, path2):
    """
    This function will append path2 to path1. This
    allows for the creation of file paths within
    directories.

    Args:
        path1 - str representing first path
        path2 - str representing REL path from path1, file or folder.

    Returns:
        String with the absolute joined path.
    """

    # Get the first absolute path
    abs_path1 = abspath(path1)

    # Return the joined paths
    return os.path.join(abs_path1, path2).replace("\\", "/")

def exists(path):
    """
    Function to check whether or not
    a given path exists in disc.

    Params:
    path - the string representing the path

    Returns:
    True if path exists, False otherwise.
    """

    # Use the OS function with the abspath
    return os.path.lexists(abspath(path))

def copy_image(source, dest):
    """
    This function will simply copy the image in
    the source location to the destinatio
    location.

    Params:
    source - image source string.
    dest - image destinationq string.

    returns -> Null.
    """

    # Cast to abs path 
    abs_src = abspath(source)
    abs_dst = abspath(dest)

    # OpenCV to open and save image
    img = cv2.imread(abs_src)
    cv2.imwrite(abs_dst, img)


if __name__ == "__main__":  # testing code here
    print(os.getcwd())
    print(get_files(os.getcwd()))
