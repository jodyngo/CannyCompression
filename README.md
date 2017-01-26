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

The level of JPEG compression to be used must also be specified as an integer between 0 and 100, where 0 represents maximum compression.
```
./CannyCompression -f:SamplePictures/1.jpg 80
```
*Note: Values between 80 to 95 seem to work best, with lower values offering diminishing returns.*

Optionally, you may also select whether you would like the processed images to be displayed using the third command.
Images are display for 2 seconds before closing.
```
./CannyCompression -d:SamplePictures/ 90 -disp
```
