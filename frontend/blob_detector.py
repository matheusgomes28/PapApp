"""blob_detector.py

Python class which performs the detection 
of blobs in a given image. This class wraps 
the SimpleBlobDetector OpenCV object.

More doc on this later..
"""

# Relevant imports
import cv2
import numpy as np


class BlobDetector(object):
    """
    The main class of the file. Inherits the 
    SimpleBlobDetector class. There are differences
    in the class depending on which OpenCV version,
    which this class was made to resulve.

    In addition, the class does not inherit the 
    SimpleBlobDetector class due to ambiguity in
    the initialisation between different version, 
    instead it just contains a detector object.
    """

   

    def __init__(self):

        # Set the filter stuff to default
        self.filterArea = False
        self.filterCircularity = False
        self.filterConvexity = False
        self.filterInertia = False
        
        # The parameter obj 
        self._params = cv2.SimpleBlobDetector_Params()
        
        # Relax all the default param values in OpenCV
        self._params.filterByArea = False
        self._params.minArea = 10
        self._params.maxArea = 2000
        
        self._params.filterByInertia = False
        self._params.minInertiaRatio = 0.1
        self._params.maxInertiaRatio = 1000

        self._params.filterByCircularity = False
        self._params.minCircularity = 0
        self._params.maxCircularity = 1000

        self._params.filterByConvexity = False
        self._params.minConvexity = 0
        self._params.maxConvexity = 10

        self._params.filterByColor = False
        self._params.blobColor = 0

        self._params.thresholdStep = 2
        self._params.minThreshold = 0
        self._params.maxThreshold = 10
        self._params.minRepeatability = 2
        self._params.minDistBetweenBlobs = 5

    def detect(self,img):
        """
        Instance method that actually performs the detection
        on an image, returning the list of points representing
        the blob location (centre) and a list of sizes.
        
        Arguments:
            img - Numpy array representing the image (greyscale, 1 channel)

        Returns:
            points - List containing (y,x) position of blobs.
            sizes  - List containing sizes of each blob.
        """

        # Create the detector (and check which version is installed)
        cv_ver = (cv2.__version__).split('.')[0]

        if int(cv_ver) < 3:
            detector = cv2.SimpleBlobDetector(self._params)
        else:
            detector = cv2.SimpleBlobDetector_create(self._params)

        # Get the keypoints
        keypoints = detector.detect(img)
        
        # Lambda to round the coordinates to integers
        r_tup = lambda y,x: (int(np.round(y)), int(np.round(x)))

        # Now split the keypoints into coordinates (y,x)
        # and the sizes of the blobs
        coords = [r_tup(*kpt.pt) for kpt in keypoints]
        sizes = [r_tup(0, kpt.size)[1] for kpt in keypoints]

        # return the tuple
        return coords, sizes


    ################################
    ## Parameter Setter Functions ##
    ################################

    def set_colour(self, min_thresh, max_thresh, step_thresh):
        """
        Method to set the parameters for the colour
        settings of the simple blob detector.

        Args:
            min_thresh  - Minimum threshold colour.
            max_thresh  - Maximum threshold colour.
            step_thresh - Step for threshold binirisation.
        """

        # Set the settings 
        self._params.minThreshold = min_thresh
        self._params.maxThreshold = max_thresh
        self._params.thresholdStep = step_thresh

    def filter_convexity(self, flag, min_c, max_c):
        """
        Set the convexity filtering settings.

        Args:
            flag -  Boolean, whether to filter by convexity or not.
            min_c - Float for the minimum convexity value.
            max_c - Float for the maximum convexity value.
        """

        # Set the param attributes
        self._params.filterByConvexity= flag
        self._params.minConvexity = min_c
        self._params.maxConvexity = max_c


    def filter_area(self, flag, min_a, max_a):
        """
        Set the area filtering settings.

        Args:
            flag  - Boolean, whether to filter by area or not.
            min_a - Float for the minimum area value.
            max_a - Float for the maximum area value.
        """

        # Set the param attributes
        self._params.filterByArea= flag
        self._params.minArea = min_a
        self._params.maxArea = max_a


    def filter_inertia(self, flag, min_i, max_i):
        """
        Set the inertia filtering settings.

        Args:
            flag  - Boolean, whether to filter by inertia or not.
            min_i - Float for the minimum inertia value.
            max_i - Float for the maximum inertia value.
        """

        # Set the param attributes
        self._params.filterByInertia = flag
        self._params.minInertiaRatio = min_i
        self._params.maxInertiaRatio = max_i

    
    def filter_circularity(self, fla, min_c, max_c):
        """
        Set the circularity filtering settings.

        Args:
            flag  -  Boolean, whether to filter by circularity or not.
            min_c - Float for the minimum circularity value.
            max_c - Float for the maximum circularity value.
        """

        # Set the param attributes
        self._params.filterByArea = flag
        self._params.minCircularity = min_c
        self._params.maxCircularity = max_c
