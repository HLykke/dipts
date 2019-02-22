# dipts
Practical classes and functions for FYS-2010 and STA-2003 at UiT.

## Classes (dipts.classes)

### TimeSeries
Inherits from numpy.ndarray. Adds additional functions:
#### acvf
Autocovariance function.
#### acf
Autocorrelation function.
#### linreg
Simple polynomial regression of specified degree. Utilizes numpy.polyfit.

### DipImage
Inherits from numpy.ndarray. Adds additional functions:
#### im2double
changes range of image defined in intensityinterval [0, N-1] to intensityinterval [0, 1].
#### im2uint8
changes range of image defined in intensityinterval [0, 1] to intensityinterval [0, 255].
#### histogram
Creates histogram of image and returns bins and counts of each bin. Shows plot if show=True.
#### histogram_equalization
Histogram equalization of an image. shows histograms on top of each other if show_hist=True.
#### hist_conversion
Not implemented yet
#### gamma_transform
Brightens or darkens image using gamma-transform. Needs argument gamma.
#### intensity_stretch
Takes an image from any intensity interval and converts to spesified parameters [minimum, L-1].
Default values are L=256 and minimum=0.
#### Blur
Blurs image using average filter. Uses square homogeneous filter as default. 
Specify parameter sfilter if preferred.
#### unsharpen
#### laplace_sharpen
