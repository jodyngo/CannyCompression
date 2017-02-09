# CannyCompression

## Setup
- Clone the repository
```
git clone https://github.com/olwyk/CannyCompression.git
```
- Install OpenCV for Linux
Follow the instructions listed on the OpenCV website.
```
http://docs.opencv.org/2.4/doc/tutorials/introduction/linux_install/linux_install.html#linux-installation
```
- Compile with CMake
```
cmake .
make
```

## Usage
CannyCompression can be run on individual files, are all files in a given directory.
- To specify a single file
```
./CannyCompression -f:<file_name>
```
- To specify a directory
```
./CannyCompression -d:<directory_name>/
```
*Note: Directories must be supplied with a trailing forward slash.*

Optionally, you may also select whether you would like the processed images to be displayed using the second command.
Images are displayed until a keypress is logged or the window is closed.
```
./CannyCompression -d:SamplePictures/ -disp
```

## Configuration
A configuration file called "config.conf" is used to control several parameters.

### JPEG Compression Level
The amount of compression to be used when writing the processed image should be specified. The value should be between 0 and 100, with lower values producing larger levels of compression. Values between 80 - 90 work well with few noticeable changes.

### Number of Images To Process
When processing the images in a directory, you may select how many to read before the program end. If this parameter is set to 0, the program will continually process images until it is stopped.

### Size Thresholding
Select whether to filter out regions of interest that aren't the correct size for a given target size. If this is enabled, the following parameters will be used.
- Target size: The size of the target ojbect in metres.
- Focal length: The focal length of the camera being used.
- Pixel size: The size of each pixel on the camera's sensor.
- Altitude: The height of the camera above the target.

### Default Values
If the parameters above are not specified, they default to the following values:
- size_thresholding =  true
- target_size  =        2.0
- focal_length =        8.5
- altitude     =      100.0
- pixel_size   =       3.69
- compression_level =    80
- num_images_to_process = 0

## Logging
CannyCompression creates a log file to record its progress, which is named in the scheme LOG_YYYY-MM-DD.txt.
Within the log, entries are made recording:
- the current configuration parameters, 
- the number of images processed as a fraction of the total, and
- the average processing time.
