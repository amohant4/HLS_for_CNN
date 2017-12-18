# High Level Synthesis for Deep Learning.
This repo has codes for hardware accelerator design for Deep CNNs using high level synthesis from Altera. 

### Requirements: 
* Visual Studio 2013 (the project was developed using windows host). Porting to linux should be straight forward. 
* OpenCV 2 was used in the host program for image capture and processing. 
* Intel OpenCL SDK 14.0
* The binary files provided in this repo are for DE5-Net board (Stratix V FPGA board from Altera). But if you want you can use the same codes to compile for any other board supporting OpenCL SDK. 


### Setup: 
* Clone this repo. 
* Download the test data (in binary format) from [here](https://drive.google.com/file/d/1aYYCH0x7Z752CkHfBN90ArDBsYcA-vMK/view?usp=sharing) and put it in the project directory.
* Un-zip the downloaded `alexNet_de5v2_data.zip` file. 
* Folder structure should look like: 
	- HLS_for_CNN
		- alexNet_de5v2
		- common
		- Model
		- Data
		- README.md