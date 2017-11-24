# SOM
A lenna image is compressed by vector quantization using Kohonen's Self organizing map. The image is divided into blocks of size `4 x 4` and the corresponding vectors are fed to the SOM. This generates a codebook of a predetermined size which is used to generate the reconstructed image.  
## Getting Started
### Prerequisites
```
Anaconda for Python 2.7
```
```
OpenCV for Python 2.7
```  
### Installing
```
Anaconda for Python 2.7
```
Go to the [downloads page of Anaconda](https://www.anaconda.com/download/) and select the installer for Python 2.7. Once downloaded, installing it should be a straightforward process. Anaconda has along with it most of the packages we need.  
```
OpenCV for Python 2.7
```
This [page](https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_setup/py_setup_in_windows/py_setup_in_windows.html) explains it quite well. 
## Running
Before running `SOM.py`, a few parameters are to be set.    
```python
image_location
bits_per_codevector
block_width
block_height
epochs
initial_learning_rate
```  
`image_location` is set to the relative location of the image from the current directory.  
`bits_per_codevector` is set based on the size of the codebook you desire. For e.g., for a 256-vector codebook, this value should be `8` as `2^8 = 256`.    
`block_width` and `block_height` are set to the size of the blocks the image is divided into. Make sure the blocks cover the the entire image.   
`epochs` is the number of epochs this algorithm is to be run.  
`initial_learning_rate` is the learning rate at `t = 0`.    

Once the parameters are decided, enter the following command to run the script.  
`python [name of the script] [image_location] [bits_per_codevector] [block_width] [block_height] [epochs] [initial_learning_rate]`  

**Please read the [wiki](https://github.com/droidadroit/SOM/wiki/SOM) for an understanding of the above terms.**  
## Results
