# File containing all the
# analysis code such as the
# Fourier transform.
import cv2
import numpy as np
import frontend.filtering as fil


def fourier(image):
	"""
	This function will use OpenCV to 
	calculate the fourier transform of 
	an image.
	"""

	# Calculate FFT (amplitude and phase)
	dft = cv2.dft(np.float32(image), flags= cv2.DFT_COMPLEX_OUTPUT)

	# Shift DFT so frequency 0 is at center
	shift = np.fft.fftshift(dft)

	# Amplitude and Phase spectrum
	m = cv2.magnitude(shift[:,:,0], shift[:,:,1])
	m[np.isclose(m, 0)] = np.finfo(float).eps
	amp = 20*np.log(m) # Log everything ()
	phase = cv2.phase(shift[:,:,0], shift[:,:,1])

	# Return both sprectra 
	return (amp, phase)


def get_laplacian(image_grey):
	"""
	This function will calculate the laplacian
	given the grey scale image.
	"""
	assert len(image_grey.shape) < 3
	
	# Approximation kernel
	kernel = np.array([[0,1,0], [1,-4,1], [0,1,0]])
	
	# Retrn convoluted image 
	return fil.image_conv(image_grey, kernel)


def get_err(grey_image):
	"""
	This function will get the ERR of an image
	using the method explained in the "Fourier
	Blurring" notebook. Note the the image 
	must be in grayscale.
	"""

	# Get the fourier spectrum with the above
	amp, phase = fourier(grey_image)

	# Get the f(0) value (or maximum)
	f0 = np.max(amp)

	# Get the energy of the amp spectrum
	E_f = np.sum(np.multiply(amp, amp))

	# Return tau = E_f/|f(0)|^2
	return E_f/np.power(np.abs(f0), 2) 

def get_rect(grey_image):
	"""
	This function will return the bounding rectangle 
	coordinates. It uses canny edge detection to identify
	the important edges, then a simple where search to 
	find the min and maxima points where the edges occur.

	Args:
		gre_image - np array representing the image data.

	Returns:
		(x1, y1, x2, y2) tuple representing the diagonal 
		vertices f the bounding rectangle.
	"""

	# Pre filter to remove the noise in black areas
	gaussian_n = np.round(np.max(grey_image.shape)*0.004) # Gaussian size
	if gaussian_n % 2 == 0: gaussian_n -= 1 # so its odd
	if gaussian_n < 3: gaussian_n = 3 # At least 3

	gaussian = fil.gaussian_kernel(int(gaussian_n), [0,0], 0.429*gaussian_n)
	grey_image = fil.image_conv(grey_image, gaussian) # Apply by convolution

	# Canny edge and image ditalion stuf
	canny_img = cv2.Canny(grey_image, 10, 60, 10) # Simple canny detection
	dilate_size = int(np.round(np.max(grey_image.shape)*0.025)) # Size of rel dilation 
	canny_img = cv2.dilate(canny_img, np.ones((dilate_size, dilate_size)))


	# Get the edge colour (assuming 0s are not edges)
	edge_c = np.max(canny_img)

	# Get the ags where the edges occur (rows and cols)
	rows, cols = np.where(canny_img == edge_c)

	# Simply return out tuple of coords now
	return (np.min(cols), np.min(rows), np.max(cols), np.max(rows))

def equalise(grey_image):
	"""
	This function will transfrom a grayscale 
	image through histogram equalisation using
	the methods described in the notebook
	"Histogram Equalisation" 

	Args: 
		image - numpy array representing the image.
	"""
	
	# Generate the intensity histogram for the image
	hist = cv2.calcHist([grey_img],[0],None,[256],[0,256])
	
	# Calculate the cumulative probability for the image
	cf = np.cumsum(hist/(grey_img.shape[0] * grey_img.shape[1]))
	
	# Transform the intensities in the image
	for row in range(grey_img.shape[0]):
		for col in range(grey_img.shape[1]):
				grey_img[row, col] = cf[grey_img[row, col]] * 256
		
	# Generate the intensity histogram for the altered image
	new_hist = cv2.calcHist([grey_img],[0],None,[256],[0,256])

	# Calculate the cumulative probability of the altered image
	n_cf = np.cumsum(new_hist/(grey_img.shape[0] * grey_img.shape[1]))
	
	# Return the transformed image
	return grey_image

## Funciton definitions to get the histograms ##
def get_histogram(image):
	"""
	This function will return the histogram
	of the given image using the OpenCV
	calcHist function.
	
	Args: 
		image - numpy array representing the image.
		
	Returns:
		Numpy matrix representing the histogram of image.
		Note each row is the histogram of each components.
	"""
	# Get all the channels
	dims = image.shape # dimension of image (H, W, D)
	n_channels = 1 if len(dims) < 3 else dims[-1]
	n_pixels = dims[0]*dims[1]
	
	# Set the parameters accordignly
	channels = [0]
	sizes = [256]
	ranges = [0, 256]
	mask = None
	
	# Separate image into different components
	if n_channels > 1:
		components = [image[:,:,i] for i in range(n_channels)]
	else:
		components = [image]
		
	# Helper function to apply calcHist to a given image
	helper = lambda x : cv2.calcHist([x], channels, mask, sizes, ranges)
	
	# Now return the stacked results
	return np.hstack([helper(comp) for comp in components])/n_pixels

def match_histogram(image, hist):
	"""
	This funcion will take an image and the 
	average histogram generated from a dataset
	and manipulate the image's histogram to that
	of the average one.

	Args: 
		image - numpy array representing the image.
		hist - numpy array representing a histogram.

	Returns: 
		Numpy array representing altered image.
	"""


	# Get the number numper of rows and columns in the image
	# for calculating how many pixels there are later on..
	rows, cols = image.shape

	# Normalise the histogram
	hist = (rows*cols*hist)/np.sum(hist)

	# The next array contains a list of 2D matrices (256), 
	# containing False and True values representing whether 
	# or not the current pixel has the corresponding intensity
	I_array = [image==i for i in range(256)] # 256 = #of intensities

	# In the next few lines the actual final form of I is
	# obtained using np.where. Note that each row in I
	# corresponds to an index in the image array.
	I = [np.where(locs) for locs in I_array]
	I = np.hstack(I).T # Get the primitive locations
					 # Note I.shape = (rows*cols, 2)

	# Copy image 
	altered = image.copy()
	
	# For loop doing the transformation
	counter = 0 # Keep track of current index in I
	for i in range(256):
		N = int(hist[i])
	  
		# Assign N first pixels to corresponding 
		# intensity        
		if N > 0: # Check whether assignments are actually made here
			altered[np.split(I[counter:counter+N], 2, axis=1)] = i
			
		# Update counter so we start from correct place in I 
		counter += N
		
	# Returned the modified image 
	return altered
