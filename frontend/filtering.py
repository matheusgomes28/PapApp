import cv2
import numpy as np
from scipy.stats import multivariate_normal


def assert_odd(v):
    # Just make sure we have number and odd
    if (type(v) is int):
        return v%2==1
    return False

def gaussian_kernel(N, mu, sigma):
    """
    This function will generate the gaussian blur
    kernel given the matrix size N and the 
    variance sigma.
    """
    # Asserting N is odd and sigma is number
    assert assert_odd(N)
    
    # Create the normal here (with ID covariance) 
    normal = multivariate_normal(mean=mu, cov=sigma*np.identity(2))
    
    # Create the position matries (x_1,x_2 in 2D)
    X_1 = np.ones((N,N))*np.arange(N) # x_1 pos
    X_2 = X_1.T #x_2 pos, just transpose the above
    
    # Shift the positions so center is at middle
    s = np.floor(N/2) #shift value
    X_1, X_2 = X_1-s, X_2-s # shifted matrices
    
    # Create holder matrix
    X = np.zeros((N,N)) # Below we have the iterator 
    for (i,j) in [(i,j) for i in range(N) for j in range(N)]:
        X[i,j] = normal.pdf([X_1[i,j], X_2[i,j]]) # Normal values
        
    # Finally just return the normalized kernel
    return X*(1/np.sum(X))

def circular_mask(dims, r1, r2):
    """
    Creates a circular mask around the centre of
    the image. It uses Numpy arrays to represent the mask,
    with floats varying in range [0,1].

    Function used for decay = (R2-D)/(R2-R1)

    Arguments:
        dims - Tuple representing array object.
        r1   - The inner radius, all pixels within this radius
               have value 1.
        r2   - The outer radius (cutoff), this is the linear cutoff.

    Returns:
        Numpy array representing the object.
    """

    # Just making some basic error checks
    assert r2>r1, isinstance(dims, tuple)

    # Create the matrix for the distance 
    offsetx, offsety = dims[1]//2, dims[0]//2
    Xs = np.tile(np.arange(dims[1])-offsetx, (dims[0], 1))
    Ys = np.tile(np.arange(dims[0])-offsety, (dims[1],1)).T
    Ds = np.sqrt(Xs*Xs + Ys*Ys)
    
    # Apply the linear cutoff
    linear = (r2-Ds)/(r2-r1)
    
    # Get inner mask
    mask = np.maximum(0, linear)
    mask[Ds<=r1] = 1

    return mask

def delta(N):
    """
    Generates the delta function given 
    an odd square dimension N by 
    creating zero matrix.
    """
    assert assert_odd(N) # Make sure kernel is odd
    X = np.zeros((N,N)) # Square matrix with all 0s
    middle = int(N/2) # Get the middle cell
    X[middle, middle] = 1
    return X

def convolution(matrix, kernel):
    """
    This function will perform convolution
    with the given matrix and kernel. This
    function uses 0 padding on the outside
    borders.
    """
    assert assert_odd(kernel.shape[0])
    
    # Padded matrix (0s on the outsides)
    N = kernel.shape[0] # Get the dim for the kernel
    I = np.pad(matrix, int(N/2), "constant")
    
    # Now do the convolution
    C = np.zeros(matrix.shape) # This is the convolved image
    h, w = C.shape # Get width and height
    s = int(N/2) # Spacing of the matrix
    positions = [(i,j) for i in range(h) for j in range(w)]
    for (i,j) in positions:
        y, x = i+s,j+s # Shift the center to the right position
        
        # Calc the convolution at each pixel
        C[i,j] = np.sum(np.multiply(kernel, I[y-s:y+s+1,x-s:x+s+1]))
        
    # Return the clipped array as uint8
    return C

def image_conv(image, kernel):
    """
    This function will convolute the image
    with the kernel given. It differs from 
    the previous convolution as it uses
    OpenCV's filter2D function.
    """
    
    # Filter2D used for performance
    return cv2.filter2D(image, -1, kernel)
