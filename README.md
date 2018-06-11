# Papilloedema App
Git repository for the Papilloedema mobile application. This repository will contain most of the  frontend/backend Python code for the image processing & ML parts, as well as planning documents and code documentation. 

## Documentation & updates
The directory `docs` will contain pdf documents regarding the updates made to project, as well as important code documentation.  Please let me know if anything needs improving, or if extra information is needed on methods/plans.

## Directory Tree & Plan
Initially, I thought it would be a good idea to mimic the plan I made for the Darwin project, where the whole project was split into three main parts: image pre-processing, image segmentation and machine learning. The image pre-processing and segmentation are kind of like the frontend, where the features of images will be extracted and irrelevant information will be removed (at the moment we mainly focus on the OD). Once the OD has been extracted, the data is then fed into the machine learning component, where the diagnosis is produced.

I have already added some boiler code I had written previously to help with OpenCV image processing, OS independent file management and some other utility code. The list below will describe the current directory tree and what each directory consists of:

 - `docs` - This is where the documentation and planning will be found. 
 - `frontend` - This directory contains the code I wrote previously to help with image analysis. At the moment it only contains image analysis and filtering stuff, but I will also add the relevant feature extraction stuff here. 
 - `ml` - Machine Learning code will be located here. At the moment this folder is empty, but over the next week I will add the relevant tensorflow models here.
- `od` - Directory created to store the optic disc segmentation stuff. Again, this is currently empty but this is what I am planning on working with in the upcoming week.
- `utils` - Miscellaneous Python stuff. So far this directory contains OS ind. file management stuff. I believe most boiler code for the server side stuff will be located here.

### Some Other Notes
I (Mat) plan to be in most days, from 9am to at least 1pm, however, if I am not in feel free to contact me via my personal email `matheusgarcia28@gmail.com`. 
